from __future__ import annotations

import asyncio
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import bot as bot_entrypoint

from src.bot.adapter_registry import AdapterSpec, validate_enabled_adapters
from src.bot.adapter_support import (
    DebouncedDispatcher,
    require_access_policy,
    whitelist_values,
)
from src.bot.adapter_support import AdapterRuntime
from src.bot.chat_service import ConversationCoordinator
from src.bot.event_dedup import PersistentEventDeduplicator
from src.bot.telegram_adapter import (
    _telegram_allowed,
    _telegram_group_allowed,
    telegram_conversation_key,
    telegram_identity,
)
from src.memory import PlatformIdentity
from src.memory.conversation import ConversationSessionBuffer


class PersistentEventDeduplicatorTests(unittest.TestCase):
    def test_completed_event_survives_process_restart(self) -> None:
        with TemporaryDirectory() as temp:
            path = Path(temp) / "events.db"
            first = PersistentEventDeduplicator(path)
            claim = first.claim("matrix", "@bot:example.org", "$event")
            self.assertIsNotNone(claim)
            first.complete(claim)  # type: ignore[arg-type]
            first.close()

            second = PersistentEventDeduplicator(path)
            self.assertIsNone(
                second.claim("matrix", "@bot:example.org", "$event")
            )
            second.close()

    def test_released_event_can_be_retried(self) -> None:
        with TemporaryDirectory() as temp:
            ledger = PersistentEventDeduplicator(Path(temp) / "events.db")
            claim = ledger.claim("slack", "T1", "Ev1")
            self.assertIsNotNone(claim)
            ledger.release(claim)  # type: ignore[arg-type]
            self.assertIsNotNone(ledger.claim("slack", "T1", "Ev1"))
            ledger.close()

    def test_account_scope_prevents_cross_bot_collisions(self) -> None:
        with TemporaryDirectory() as temp:
            ledger = PersistentEventDeduplicator(Path(temp) / "events.db")
            self.assertIsNotNone(ledger.claim("telegram", "bot-a", "42"))
            self.assertIsNotNone(ledger.claim("telegram", "bot-b", "42"))
            ledger.close()


class DispatcherPersistenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_dispatcher_completes_claim_only_after_success(self) -> None:
        with TemporaryDirectory() as temp:
            ledger = PersistentEventDeduplicator(Path(temp) / "events.db")
            calls: list[dict] = []

            class Runtime:
                event_deduplicator = ledger

                async def process_message(self, **kwargs):
                    calls.append(kwargs)
                    return True

            dispatcher = DebouncedDispatcher(Runtime(), wait_seconds=0.01)  # type: ignore[arg-type]
            identity = PlatformIdentity("discord", "u1", "Teacher")

            async def send_text(_value: str) -> None:
                return None

            accepted = dispatcher.submit(
                conversation_key="discord:room-a:u1",
                identity=identity,
                text="hello",
                send_text=send_text,
                event_id="m1",
                event_scope="bot-a",
            )
            self.assertTrue(accepted)
            await asyncio.sleep(0.04)
            self.assertEqual(1, len(calls))
            self.assertEqual(
                "discord:room-a:u1", calls[0]["conversation_key"]
            )
            self.assertFalse(
                dispatcher.submit(
                    conversation_key="discord:room-a:u1",
                    identity=identity,
                    text="duplicate",
                    send_text=send_text,
                    event_id="m1",
                    event_scope="bot-a",
                )
            )
            await dispatcher.close()
            ledger.close()


class SideEffectOrderingTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _runtime(events: list[str]) -> AdapterRuntime:
        class Reply:
            text = "reply"
            memory = SimpleNamespace(error="")

            async def finalize(self) -> None:
                events.append("finalize")

        class Coordinator:
            @staticmethod
            def route(_text: str):
                return SimpleNamespace(user=False, public=False, knowledge=False)

            async def handle(self, **_kwargs):
                events.append("generate")
                return Reply()

        runtime = AdapterRuntime.__new__(AdapterRuntime)
        runtime.coordinator = Coordinator()
        runtime.semaphore = asyncio.Semaphore(1)
        return runtime

    async def test_transport_send_precedes_local_memory_commit(self) -> None:
        events: list[str] = []
        runtime = self._runtime(events)

        async def send(_text: str) -> None:
            events.append("send")

        completed = await runtime.process_message(
            identity=PlatformIdentity("test", "u1", "Teacher"),
            conversation_key="test:room:u1",
            text="hello",
            send_text=send,
        )
        self.assertTrue(completed)
        self.assertEqual(["generate", "send", "finalize"], events)

    async def test_failed_send_does_not_commit_local_memory(self) -> None:
        events: list[str] = []
        runtime = self._runtime(events)

        async def send(_text: str) -> None:
            events.append("send")
            raise ConnectionError("delivery failed")

        completed = await runtime.process_message(
            identity=PlatformIdentity("test", "u1", "Teacher"),
            conversation_key="test:room:u1",
            text="hello",
            send_text=send,
        )
        self.assertFalse(completed)
        self.assertEqual("generate", events[0])
        self.assertNotIn("finalize", events)


class ConversationAddressTests(unittest.TestCase):
    def test_retrieval_history_is_isolated_by_conversation(self) -> None:
        coordinator = ConversationCoordinator.__new__(ConversationCoordinator)
        coordinator.sessions = ConversationSessionBuffer(20)
        identity = PlatformIdentity("matrix", "@teacher:example.org", "Teacher")
        coordinator.sessions.add("matrix:room-a:teacher", "user", "星野去了哪里")
        coordinator.sessions.add("matrix:room-a:teacher", "assistant", "她去了沙漠")

        same_room = coordinator._retrieval_question(
            identity,
            "为什么？",
            conversation_key="matrix:room-a:teacher",
        )
        other_room = coordinator._retrieval_question(
            identity,
            "为什么？",
            conversation_key="matrix:room-b:teacher",
        )
        self.assertIn("星野去了哪里", same_room)
        self.assertEqual("为什么？", other_room)

    def test_telegram_chat_and_thread_are_part_of_conversation_key(self) -> None:
        user = SimpleNamespace(id=42, username="teacher", full_name="Teacher")
        first = SimpleNamespace(
            effective_user=user,
            effective_chat=SimpleNamespace(id=-100, type="supergroup"),
            effective_message=SimpleNamespace(message_thread_id=7),
        )
        second = SimpleNamespace(
            effective_user=user,
            effective_chat=SimpleNamespace(id=-100, type="supergroup"),
            effective_message=SimpleNamespace(message_thread_id=8),
        )
        identity = telegram_identity(first)  # type: ignore[arg-type]
        self.assertNotEqual(
            telegram_conversation_key(first, identity),  # type: ignore[arg-type]
            telegram_conversation_key(second, identity),  # type: ignore[arg-type]
        )


class TelegramPolicyTests(unittest.TestCase):
    @staticmethod
    def _update(*, text: str = "hello", chat_type: str = "private"):
        return SimpleNamespace(
            effective_user=SimpleNamespace(id=42),
            effective_chat=SimpleNamespace(id=-100, type=chat_type),
            effective_message=SimpleNamespace(
                text=text,
                caption="",
                reply_to_message=None,
            ),
        )

    def test_access_is_open_by_default_and_stable_id_allowlist_is_optional(self) -> None:
        update = self._update()
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(_telegram_allowed(update))  # type: ignore[arg-type]
        with patch.dict(
            os.environ,
            {
                "TELEGRAM_WHITELIST_ENABLED": "true",
                "TELEGRAM_ALLOWED_USERS": "7",
            },
            clear=True,
        ):
            self.assertFalse(_telegram_allowed(update))  # type: ignore[arg-type]
        with patch.dict(
            os.environ,
            {
                "TELEGRAM_WHITELIST_ENABLED": "true",
                "TELEGRAM_ALLOWED_USERS": "42",
            },
            clear=True,
        ):
            self.assertTrue(_telegram_allowed(update))  # type: ignore[arg-type]

    def test_group_defaults_to_mentions(self) -> None:
        context = SimpleNamespace(bot=SimpleNamespace(id=99, username="AlonaBot"))
        plain = self._update(chat_type="group")
        mentioned = self._update(text="@AlonaBot hello", chat_type="group")
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(
                _telegram_group_allowed(plain, context)  # type: ignore[arg-type]
            )
            self.assertTrue(
                _telegram_group_allowed(mentioned, context)  # type: ignore[arg-type]
            )


class AdapterPreflightTests(unittest.TestCase):
    SPEC = AdapterSpec(
        name="sample",
        enable_env="ENABLE_SAMPLE",
        module="sample.module",
        starter="start",
        dependency_modules=(),
        required_env_groups=(("SAMPLE_TOKEN",),),
        allowlist_envs=("SAMPLE_ALLOWED_USERS",),
    )

    def test_preflight_allows_startup_when_whitelist_is_disabled(self) -> None:
        with patch.dict(os.environ, {"SAMPLE_TOKEN": "token"}, clear=True):
            validate_enabled_adapters([self.SPEC])

    def test_disabled_whitelist_ignores_stale_allowlist_values(self) -> None:
        with patch.dict(
            os.environ,
            {
                "SAMPLE_WHITELIST_ENABLED": "false",
                "SAMPLE_ALLOWED_USERS": "stale-id",
            },
            clear=True,
        ):
            self.assertEqual(set(), whitelist_values("sample", "SAMPLE_ALLOWED_USERS"))

    def test_enabled_empty_whitelist_fails_preflight(self) -> None:
        with patch.dict(
            os.environ,
            {"SAMPLE_TOKEN": "token", "SAMPLE_WHITELIST_ENABLED": "true"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "whitelist is enabled but empty"):
                validate_enabled_adapters([self.SPEC])

    def test_enabled_populated_whitelist_passes_preflight(self) -> None:
        with patch.dict(
            os.environ,
            {
                "SAMPLE_TOKEN": "token",
                "SAMPLE_WHITELIST_ENABLED": "true",
                "SAMPLE_ALLOWED_USERS": "stable-id",
            },
            clear=True,
        ):
            validate_enabled_adapters([self.SPEC])

    def test_malformed_whitelist_flag_is_rejected(self) -> None:
        with patch.dict(
            os.environ, {"SAMPLE_WHITELIST_ENABLED": "perhaps"}, clear=True
        ):
            with self.assertRaisesRegex(ValueError, "must be true or false"):
                require_access_policy("sample", "SAMPLE_ALLOWED_USERS")

    def test_listener_collisions_are_reported_before_startup(self) -> None:
        line = AdapterSpec(
            "line",
            "ENABLE_LINE",
            "line.module",
            "start",
            (),
            (("LINE_TOKEN",),),
            ("LINE_ALLOWED_USERS",),
        )
        whatsapp = AdapterSpec(
            "whatsapp",
            "ENABLE_WHATSAPP",
            "whatsapp.module",
            "start",
            (),
            (("WHATSAPP_TOKEN",),),
            ("WHATSAPP_ALLOWED_USERS",),
        )
        with patch.dict(
            os.environ,
            {
                "LINE_TOKEN": "token",
                "WHATSAPP_TOKEN": "token",
                "LINE_WHITELIST_ENABLED": "false",
                "WHATSAPP_WHITELIST_ENABLED": "false",
                "LINE_WEBHOOK_HOST": "127.0.0.1",
                "WHATSAPP_WEBHOOK_HOST": "0.0.0.0",
                "LINE_WEBHOOK_PORT": "8081",
                "WHATSAPP_WEBHOOK_PORT": "8081",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "listener collision"):
                validate_enabled_adapters([line, whatsapp])


class ProcessOrchestrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_one_adapter_failure_cancels_peers_and_closes_runtime_once(self) -> None:
        peer_cancelled = asyncio.Event()

        async def failing(_runtime: object) -> None:
            await asyncio.sleep(0)
            raise ConnectionError("gateway lost")

        async def peer(_runtime: object) -> None:
            try:
                await asyncio.Event().wait()
            finally:
                peer_cancelled.set()

        class Spec:
            def __init__(self, name: str, starter) -> None:
                self.name = name
                self._starter = starter

            def load_starter(self):
                return self._starter

        runtime = SimpleNamespace(close=AsyncMock())
        with (
            patch.object(
                bot_entrypoint,
                "enabled_adapters",
                return_value=[Spec("broken", failing), Spec("peer", peer)],
            ),
            patch.object(bot_entrypoint, "validate_enabled_adapters"),
            patch.object(bot_entrypoint, "load_dotenv"),
            patch.object(
                bot_entrypoint.AdapterRuntime,
                "create",
                new=AsyncMock(return_value=runtime),
            ),
        ):
            with self.assertRaises(ExceptionGroup):
                await bot_entrypoint.main()

        self.assertTrue(peer_cancelled.is_set())
        runtime.close.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
