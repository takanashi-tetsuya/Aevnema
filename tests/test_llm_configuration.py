from __future__ import annotations

import asyncio
import os
from pathlib import Path
import tempfile
import unittest
import time
from unittest.mock import patch

from src.bot.chat_service import ConversationCoordinator
from src.bot.persona import apply_model_setting
from src.llm.engine import (
    EngineFactory,
    GenerationOptions,
    LLMConfig,
    LLMEngine,
    Message,
)
from src.llm.user_settings import UserModelSettingsStore
from src.memory.conversation import ConversationSessionBuffer
from src.memory.identity import PlatformIdentity
from src.memory.service import RetrievedMemory


class GenerationConfigurationTests(unittest.TestCase):
    def test_per_call_options_override_without_mutating_base_config(self):
        base = LLMConfig(
            provider="openai",
            model_name="test",
            api_key="secret",
            enable_thinking=False,
            temperature=0.8,
            extra_body={"vendor_flag": "base"},
        )
        effective = base.with_options(
            {
                "enable_thinking": True,
                "thinking_budget": 2048,
                "temperature": 0.2,
                "extra_body": {"vendor_flag": "override"},
            }
        )
        self.assertFalse(base.enable_thinking)
        self.assertEqual(0.8, base.temperature)
        self.assertTrue(effective.enable_thinking)
        self.assertEqual(2048, effective.thinking_budget)
        self.assertEqual(0.2, effective.temperature)
        self.assertEqual("override", effective.extra_body["vendor_flag"])

    def test_explicit_auto_can_remove_a_model_thinking_default(self):
        base = LLMConfig(
            provider="openai",
            model_name="test",
            api_key="secret",
            enable_thinking=False,
        )
        effective = base.with_options(GenerationOptions(enable_thinking=None))
        self.assertIsNone(effective.enable_thinking)

    def test_factory_merges_global_task_and_model_layers(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "models.toml"
            config.write_text(
                """
[runtime]
max_retries = 2
retry_delay = 0.25
[defaults]
provider = "openai"
base_url = "https://example.invalid/v1"
api_key_env = "TEST_MODEL_KEY"
temperature = 0.7
max_tokens = 1000
[task_defaults.chat]
temperature = 0.3
max_tokens = 1200
[[engines.chat]]
model = "primary"
[[engines.chat]]
model = "fallback"
max_tokens = 600
enable_thinking = false
""".strip(),
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {
                    "MODEL_CONFIG_PATH": str(config),
                    "TEST_MODEL_KEY": "test-key",
                },
                clear=False,
            ):
                engine = EngineFactory.create("chat")
        self.assertEqual(2, engine.max_retries)
        self.assertEqual(0.25, engine.retry_delay)
        self.assertEqual(0.3, engine.configs[0].temperature)
        self.assertEqual(1200, engine.configs[0].max_tokens)
        self.assertEqual(600, engine.configs[1].max_tokens)
        self.assertFalse(engine.configs[1].enable_thinking)
        self.assertNotIn("api_key", str(engine.describe()))
        self.assertNotIn("test-key", str(engine.describe()))


class RequestDeadlineTests(unittest.IsolatedAsyncioTestCase):
    async def test_engine_enforces_one_deadline_across_provider_attempts(self):
        engine = LLMEngine(
            [
                LLMConfig(
                    provider="openai",
                    model_name="slow",
                    api_key="test",
                )
            ],
            max_retries=1,
            request_deadline_seconds=0.03,
        )

        def slow(*_args, **_kwargs):
            time.sleep(1.0)
            return "late"

        engine._call_openai = slow
        with self.assertRaises(RuntimeError) as raised:
            await engine.generate_response([Message(role="user", content="hi")])
        self.assertIn("TimeoutError", str(raised.exception))

    async def test_deadline_does_not_wait_for_cancellation_resistant_sdk(self):
        engine = LLMEngine(
            [
                LLMConfig(
                    provider="openai",
                    model_name="stuck",
                    api_key="test",
                )
            ],
            max_retries=1,
            request_deadline_seconds=0.03,
        )
        from threading import Event

        release = Event()

        def cancellation_resistant(*_args, **_kwargs):
            release.wait(timeout=1.0)
            return "late"

        engine._call_openai = cancellation_resistant
        loop = asyncio.get_running_loop()
        started = loop.time()
        with self.assertRaises(RuntimeError):
            await engine.generate_response([Message(role="user", content="hi")])
        self.assertLess(loop.time() - started, 0.2)
        release.set()


class UserSettingsTests(unittest.TestCase):
    def test_settings_are_persistent_and_isolated_by_platform_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            first = UserModelSettingsStore(path)
            telegram_a = PlatformIdentity("telegram", "1001", "Teacher")
            telegram_b = PlatformIdentity("telegram", "1002", "Teacher")
            discord_a = PlatformIdentity("discord", "1001", "Teacher")
            first.update(
                telegram_a,
                thinking="on",
                thinking_budget=1536,
                temperature=0.4,
            )

            restored = UserModelSettingsStore(path)
            self.assertEqual("on", restored.get(telegram_a).thinking)
            self.assertEqual(1536, restored.get(telegram_a).thinking_budget)
            self.assertEqual("inherit", restored.get(telegram_b).thinking)
            self.assertEqual("inherit", restored.get(discord_a).thinking)

    def test_model_command_updates_only_current_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            store = UserModelSettingsStore(Path(directory) / "settings.json")
            first = PlatformIdentity("telegram", "1")
            second = PlatformIdentity("telegram", "2")
            response = apply_model_setting(store, first, ["thinking", "on"])
            self.assertIn("设置已保存", response)
            self.assertTrue(store.request_options(first)["enable_thinking"])
            self.assertEqual({}, store.request_options(second))
            apply_model_setting(store, first, ["reset"])
            self.assertEqual({}, store.request_options(first))


class CoordinatorGenerationOptionsTests(unittest.IsolatedAsyncioTestCase):
    async def test_coordinator_passes_identity_specific_chat_options(self):
        class EmptyMemory:
            async def recall(self, identity, question, route=None):
                return RetrievedMemory("")

        class CapturingChat:
            async def generate_response(
                self,
                messages,
                system_prompt,
                task_context="",
                request_options=None,
            ):
                self.task_context = task_context
                self.request_options = request_options
                return "reply"

        chat = CapturingChat()
        identity = PlatformIdentity("matrix", "stable-7")
        coordinator = ConversationCoordinator(
            chat_engine=chat,
            memory_system=EmptyMemory(),
            system_prompt_factory=lambda context: context,
            sessions=ConversationSessionBuffer(),
            generation_options_factory=lambda current: {
                "enable_thinking": current.key == identity.key,
                "temperature": 0.4,
            },
        )
        await coordinator.handle(identity=identity, text="你好")
        self.assertEqual("matrix:stable-7:chat", chat.task_context)
        self.assertEqual(
            {"enable_thinking": True, "temperature": 0.4},
            chat.request_options,
        )


if __name__ == "__main__":
    unittest.main()
