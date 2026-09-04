from __future__ import annotations

import asyncio
import unittest

from src.bot.adapter_support import AdapterRuntime, DebouncedDispatcher, split_message
from src.bot.matrix_adapter import matrix_identity, matrix_room_allows
from src.bot.xmpp_adapter import (
    XMPPAdapter,
    _allowed_jids,
    _positive_int_env,
    xmpp_identity,
)
from src.memory import PlatformIdentity


class _FakeCoordinator:
    def __init__(self) -> None:
        self.cleared: list[str] = []

    def clear_recent_context(self, identity: PlatformIdentity) -> None:
        self.cleared.append(identity.key)


class AdapterPolicyTests(unittest.TestCase):
    def test_xmpp_identity_uses_bare_jid(self) -> None:
        identity = xmpp_identity("Teacher@Example.org/Phone", "Teacher")
        self.assertEqual("xmpp:teacher@example.org", identity.key)
        self.assertEqual("Teacher", identity.display_name)

    def test_matrix_identity_keeps_fully_qualified_user_id(self) -> None:
        identity = matrix_identity("@teacher:example.org", "Teacher")
        self.assertEqual("matrix:@teacher:example.org", identity.key)
        self.assertEqual("Teacher", identity.display_name)

    def test_matrix_room_modes(self) -> None:
        bot_id = "@alona:example.org"
        self.assertTrue(
            matrix_room_allows(
                room_is_group=False, body="hello", bot_user_id=bot_id, mode="direct"
            )
        )
        self.assertFalse(
            matrix_room_allows(
                room_is_group=True, body="hello", bot_user_id=bot_id, mode="direct"
            )
        )
        self.assertTrue(
            matrix_room_allows(
                room_is_group=True,
                body=f"{bot_id} hello",
                bot_user_id=bot_id,
                mode="mentions",
            )
        )
        self.assertTrue(
            matrix_room_allows(
                room_is_group=True, body="hello", bot_user_id=bot_id, mode="all"
            )
        )
        with self.assertRaises(ValueError):
            matrix_room_allows(
                room_is_group=False,
                body="hello",
                bot_user_id=bot_id,
                mode="invalid",
            )

    def test_split_message_preserves_content(self) -> None:
        chunks = split_message("first paragraph\nsecond paragraph\nthird", 20)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(0 < len(chunk) <= 20 for chunk in chunks))
        self.assertEqual(
            "first paragraphsecond paragraphthird", "".join(chunks).replace("\n", "")
        )


class AdapterRuntimeCommandTests(unittest.IsolatedAsyncioTestCase):
    async def test_identity_and_clear_commands(self) -> None:
        runtime = object.__new__(AdapterRuntime)
        coordinator = _FakeCoordinator()
        runtime.coordinator = coordinator
        identity = PlatformIdentity("matrix", "@teacher:example.org", "Teacher")

        identity_reply = await runtime.command_reply(identity, "/identity")
        self.assertIn("matrix", identity_reply or "")
        self.assertIn("@teacher:example.org", identity_reply or "")

        clear_reply = await runtime.command_reply(identity, "/clear")
        self.assertIn("已清空", clear_reply or "")
        self.assertEqual([identity.key], coordinator.cleared)

        self.assertIsNone(await runtime.command_reply(identity, "ordinary text"))

    async def test_debounce_merges_messages(self) -> None:
        calls: list[dict] = []

        class FakeRuntime:
            async def process_message(self, **kwargs) -> None:
                calls.append(kwargs)

        dispatcher = DebouncedDispatcher(FakeRuntime(), wait_seconds=0.01)  # type: ignore[arg-type]
        identity = PlatformIdentity("xmpp", "teacher@example.org")

        async def send_text(_text: str) -> None:
            return None

        dispatcher.submit(
            conversation_key=identity.key,
            identity=identity,
            text="first",
            send_text=send_text,
        )
        dispatcher.submit(
            conversation_key=identity.key,
            identity=identity,
            text="second",
            send_text=send_text,
        )
        await asyncio.sleep(0.04)
        self.assertEqual(1, len(calls))
        self.assertEqual("first\nsecond", calls[0]["text"])
        await dispatcher.close()


class XMPPAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_message_uses_bare_identity_but_resource_specific_conversation(
        self,
    ) -> None:
        submissions: list[dict] = []

        class FakeDispatcher:
            def submit(self, **kwargs) -> None:
                submissions.append(kwargs)

        class BareJID:
            bare = "alona@example.org"

        class Harness:
            boundjid = BareJID()
            allowed_jids = {"teacher@example.org"}
            message_limit = 5
            dispatcher = FakeDispatcher()

            def send_message(self, **_kwargs) -> None:
                return None

        message = {
            "type": "chat",
            "body": " hello ",
            "from": "Teacher@Example.org/Phone",
        }
        await XMPPAdapter._on_message(Harness(), message)  # type: ignore[arg-type]

        self.assertEqual(1, len(submissions))
        self.assertEqual("xmpp:teacher@example.org", submissions[0]["identity"].key)
        self.assertEqual("hello", submissions[0]["text"])
        self.assertEqual(
            "xmpp:teacher@example.org/Teacher@Example.org/Phone",
            submissions[0]["conversation_key"],
        )

    async def test_message_rejects_missing_or_disallowed_sender(self) -> None:
        submissions: list[dict] = []

        class FakeDispatcher:
            def submit(self, **kwargs) -> None:
                submissions.append(kwargs)

        class BareJID:
            bare = "alona@example.org"

        class Harness:
            boundjid = BareJID()
            allowed_jids = {"teacher@example.org"}
            message_limit = 8000
            dispatcher = FakeDispatcher()

        harness = Harness()
        await XMPPAdapter._on_message(  # type: ignore[arg-type]
            harness, {"type": "chat", "body": "hello", "from": ""}
        )
        await XMPPAdapter._on_message(  # type: ignore[arg-type]
            harness,
            {"type": "chat", "body": "hello", "from": "other@example.org/pc"},
        )
        self.assertEqual([], submissions)

    async def test_close_cancels_connection_attempt_and_is_idempotent(self) -> None:
        calls: list[str] = []

        class FakeDispatcher:
            async def close(self) -> None:
                calls.append("dispatcher")

        class Harness:
            _adapter_closed = False
            dispatcher = FakeDispatcher()

            def cancel_connection_attempt(self) -> None:
                calls.append("cancel")

            def disconnect(self):
                calls.append("disconnect")

                async def finished() -> None:
                    calls.append("disconnected")

                return finished()

        harness = Harness()
        await XMPPAdapter.close_adapter(harness)  # type: ignore[arg-type]
        await XMPPAdapter.close_adapter(harness)  # type: ignore[arg-type]
        self.assertEqual(
            ["dispatcher", "cancel", "disconnect", "disconnected"], calls
        )

    def test_positive_integer_environment_validation(self) -> None:
        import os
        from unittest.mock import patch

        with patch.dict(os.environ, {"XMPP_PORT": "65535"}):
            self.assertEqual(
                65535, _positive_int_env("XMPP_PORT", "5222", maximum=65535)
            )
        with patch.dict(os.environ, {"XMPP_PORT": "65536"}):
            with self.assertRaises(ValueError):
                _positive_int_env("XMPP_PORT", "5222", maximum=65535)
        with patch.dict(os.environ, {"XMPP_MESSAGE_LIMIT": "0"}):
            with self.assertRaises(ValueError):
                _positive_int_env("XMPP_MESSAGE_LIMIT", "8000")

    def test_allowed_jids_are_validated_and_normalized_to_bare_jids(self) -> None:
        import os
        from unittest.mock import patch

        with patch.dict(
            os.environ,
            {"XMPP_ALLOWED_JIDS": " Teacher@Example.org/Phone, other@example.org "},
        ):
            self.assertEqual(
                {"teacher@example.org", "other@example.org"}, _allowed_jids()
            )
        with patch.dict(os.environ, {"XMPP_ALLOWED_JIDS": "not a jid"}):
            with self.assertRaises(ValueError):
                _allowed_jids()

    async def test_message_arriving_during_processing_is_not_lost(self) -> None:
        calls: list[str] = []
        processing = asyncio.Event()
        release = asyncio.Event()

        class FakeRuntime:
            async def process_message(self, **kwargs) -> None:
                calls.append(kwargs["text"])
                if len(calls) == 1:
                    processing.set()
                    await release.wait()

        dispatcher = DebouncedDispatcher(FakeRuntime(), wait_seconds=0.01)  # type: ignore[arg-type]
        identity = PlatformIdentity("matrix", "@teacher:example.org")

        async def send_text(_text: str) -> None:
            return None

        dispatcher.submit(
            conversation_key=identity.key,
            identity=identity,
            text="first",
            send_text=send_text,
        )
        await asyncio.wait_for(processing.wait(), timeout=0.2)
        dispatcher.submit(
            conversation_key=identity.key,
            identity=identity,
            text="second",
            send_text=send_text,
        )
        release.set()
        await asyncio.sleep(0.04)
        self.assertEqual(["first", "second"], calls)
        await dispatcher.close()


if __name__ == "__main__":
    unittest.main()
