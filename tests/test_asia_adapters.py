from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.bot import dingtalk_adapter as dingtalk_module
from src.bot import line_adapter as line_module
from src.bot.dingtalk_adapter import (
    DingTalkAdapter,
    dingtalk_identity,
    dingtalk_message_allows,
)
from src.bot.lark_adapter import LarkAdapter, lark_identity, lark_message_allows
from src.bot.line_adapter import (
    LineAdapter,
    line_identity,
    line_message_allows,
    verify_line_signature,
)
from src.bot.qq_adapter import QQAdapter, qq_identity, qq_message_allows
from src.bot.wecom_adapter import (
    WeComAdapter,
    parse_wecom_frame,
    wecom_identity,
    wecom_message_allows,
)


class _CapturingDispatcher:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.closed = False
        self._events: set[tuple[object, object]] = set()

    def submit(self, **kwargs: object) -> bool:
        key = (kwargs.get("event_scope"), kwargs.get("event_id"))
        if key[1] and key in self._events:
            return False
        if key[1]:
            self._events.add(key)
        self.calls.append(kwargs)
        return True

    async def close(self) -> None:
        self.closed = True


class _FakeLarkChannel:
    def __init__(self) -> None:
        self.handler = None
        self.sent: list[tuple[str, dict, dict]] = []
        self.unsubscribed = False

    def on(self, _name: str, handler: object):
        self.handler = handler

        def unsubscribe() -> None:
            self.unsubscribed = True

        return unsubscribe

    async def send(self, to: str, message: dict, opts: dict) -> object:
        self.sent.append((to, message, opts))
        return SimpleNamespace(success=True)


class _FakeWeComClient:
    def __init__(self) -> None:
        self.handlers: dict[str, object] = {}
        self.sent: list[tuple[str, dict]] = []

    def on(self, event: str, handler: object) -> None:
        self.handlers[event] = handler

    async def send_message(self, target: str, body: dict) -> dict:
        self.sent.append((target, body))
        return {}

    async def download_file(self, _url: str, _key: str | None):
        return b"image", "photo.png"


class AsiaIdentityAndPolicyTests(unittest.TestCase):
    def test_stable_identity_namespaces(self) -> None:
        self.assertEqual("ou_1", lark_identity("ou_1", "Old Name").platform_user_id)
        self.assertEqual("zhangsan", wecom_identity("zhangsan").platform_user_id)
        self.assertEqual(
            "corp-a:staff-1",
            dingtalk_identity("corp-a", "staff-1", "opaque").platform_user_id,
        )
        self.assertEqual(
            "group:member-openid", qq_identity("member-openid", "group").platform_user_id
        )
        self.assertEqual("U123", line_identity("U123", "Mutable").platform_user_id)

    def test_group_policies_require_platform_mentions(self) -> None:
        self.assertFalse(
            lark_message_allows(
                chat_type="group", mentioned_bot=False, group_mode="mentions"
            )
        )
        self.assertFalse(
            wecom_message_allows(
                chat_type="group", mentioned_bot=False, group_mode="mentions"
            )
        )
        self.assertFalse(
            dingtalk_message_allows(
                conversation_type="2", is_in_at_list=False, group_mode="mentions"
            )
        )
        self.assertTrue(
            qq_message_allows(
                chat_scope="group",
                event_type="GROUP_AT_MESSAGE_CREATE",
                group_mode="mentions",
            )
        )
        self.assertFalse(
            line_message_allows(
                source_type="group", mentioned_bot=False, group_mode="mentions"
            )
        )

    def test_direct_messages_are_not_subject_to_group_mention_mode(self) -> None:
        self.assertTrue(
            lark_message_allows(
                chat_type="p2p", mentioned_bot=False, group_mode="mentions"
            )
        )
        self.assertTrue(
            dingtalk_message_allows(
                conversation_type="1", is_in_at_list=False, group_mode="mentions"
            )
        )
        self.assertTrue(
            line_message_allows(
                source_type="user", mentioned_bot=False, group_mode="mentions"
            )
        )


class LarkAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_structured_message_is_queued_and_sent_with_reply_reference(self) -> None:
        channel = _FakeLarkChannel()
        with patch.dict(
            os.environ,
            {
                "LARK_GROUP_MODE": "mentions",
                "LARK_ALLOWED_USERS": "",
                "LARK_ALLOWED_CHATS": "",
            },
        ):
            adapter = LarkAdapter(channel, SimpleNamespace())  # type: ignore[arg-type]
        adapter.dispatcher = _CapturingDispatcher()  # type: ignore[assignment]
        message = SimpleNamespace(
            id="om_1",
            sender=SimpleNamespace(
                open_id="ou_1", display_name="Teacher", is_bot=False
            ),
            conversation=SimpleNamespace(chat_id="oc_1", chat_type="group"),
            mentioned_bot=True,
            safe_content_text="hello",
            body_text="",
            content_text="@bot hello",
        )
        await adapter._on_message(message)
        self.assertEqual("hello", adapter.dispatcher.calls[0]["text"])  # type: ignore[attr-defined]
        await adapter.dispatcher.calls[0]["send_text"]("reply")  # type: ignore[attr-defined]
        self.assertEqual(
            ("oc_1", {"text": "reply"}, {"reply_target_gone": "fresh", "reply_to": "om_1"}),
            channel.sent[0],
        )

    async def test_allowlist_and_dedup_are_checked_before_dispatch(self) -> None:
        channel = _FakeLarkChannel()
        with patch.dict(
            os.environ,
            {
                "LARK_GROUP_MODE": "all",
                "LARK_WHITELIST_ENABLED": "true",
                "LARK_ALLOWED_USERS": "ou_allowed",
                "LARK_ALLOWED_CHATS": "oc_allowed",
            },
        ):
            adapter = LarkAdapter(channel, SimpleNamespace())  # type: ignore[arg-type]
        adapter.dispatcher = _CapturingDispatcher()  # type: ignore[assignment]
        base = dict(
            id="om_1",
            conversation=SimpleNamespace(chat_id="oc_allowed", chat_type="group"),
            mentioned_bot=False,
            safe_content_text="hello",
        )
        await adapter._on_message(
            SimpleNamespace(
                **base,
                sender=SimpleNamespace(
                    open_id="ou_denied", display_name="", is_bot=False
                ),
            )
        )
        allowed = SimpleNamespace(
            **base,
            sender=SimpleNamespace(
                open_id="ou_allowed", display_name="", is_bot=False
            ),
        )
        await adapter._on_message(allowed)
        await adapter._on_message(allowed)
        self.assertEqual(1, len(adapter.dispatcher.calls))  # type: ignore[attr-defined]


class WeComAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_official_frame_shape_is_parsed_and_queued(self) -> None:
        frame = {
            "headers": {"req_id": "req-1"},
            "body": {
                "msgid": "msg-1",
                "msgtype": "text",
                "chattype": "group",
                "chatid": "chat-1",
                "from": {"userid": "teacher", "name": "Teacher"},
                "text": {"content": "hello"},
            },
        }
        parsed = parse_wecom_frame(frame)
        self.assertIsNotNone(parsed)
        self.assertEqual(("teacher", "chat-1", "hello"), (parsed.user_id, parsed.chat_id, parsed.text))  # type: ignore[union-attr]

        client = _FakeWeComClient()
        with patch.dict(
            os.environ,
            {
                "WECOM_GROUP_MODE": "mentions",
                "WECOM_ALLOWED_USERS": "",
                "WECOM_ALLOWED_CHATS": "",
            },
        ):
            adapter = WeComAdapter(client, SimpleNamespace())  # type: ignore[arg-type]
        adapter.dispatcher = _CapturingDispatcher()  # type: ignore[assignment]
        client.handlers["message.text"](frame)  # type: ignore[operator]
        await __import__("asyncio").sleep(0)
        self.assertEqual("hello", adapter.dispatcher.calls[0]["text"])  # type: ignore[attr-defined]
        await adapter.dispatcher.calls[0]["send_text"]("reply")  # type: ignore[attr-defined]
        self.assertEqual("chat-1", client.sent[0][0])

    async def test_mixed_image_uses_official_download_decryption_boundary(self) -> None:
        client = _FakeWeComClient()
        with patch.dict(os.environ, {"WECOM_GROUP_MODE": "all"}):
            adapter = WeComAdapter(client, SimpleNamespace())  # type: ignore[arg-type]
        adapter.dispatcher = _CapturingDispatcher()  # type: ignore[assignment]
        frame = {
            "body": {
                "msgid": "mixed-1",
                "msgtype": "mixed",
                "chattype": "single",
                "from": {"userid": "teacher"},
                "mixed": {
                    "msg_item": [
                        {"msgtype": "text", "text": {"content": "look"}},
                        {
                            "msgtype": "image",
                            "image": {"url": "https://media", "aeskey": "key"},
                        },
                    ]
                },
            }
        }
        client.handlers["message.mixed"](frame)  # type: ignore[operator]
        await __import__("asyncio").sleep(0)
        self.assertEqual([(b"image", "image/png")], adapter.dispatcher.calls[0]["images"])  # type: ignore[attr-defined]


@unittest.skipUnless(dingtalk_module._DINGTALK_AVAILABLE, "dingtalk-stream not installed")
class DingTalkAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_process_returns_ack_without_awaiting_runtime(self) -> None:
        with patch.dict(
            os.environ,
            {
                "DINGTALK_GROUP_MODE": "mentions",
                "DINGTALK_ALLOWED_USERS": "",
                "DINGTALK_ALLOWED_CONVERSATIONS": "",
            },
        ):
            adapter = DingTalkAdapter(SimpleNamespace())  # type: ignore[arg-type]
        adapter.dispatcher = _CapturingDispatcher()  # type: ignore[assignment]
        callback = SimpleNamespace(
            data={
                "msgId": "msg-1",
                "msgtype": "text",
                "text": {"content": "hello"},
                "senderId": "opaque",
                "senderStaffId": "staff-1",
                "senderCorpId": "corp-1",
                "senderNick": "Teacher",
                "conversationId": "cid-1",
                "conversationType": "2",
                "isInAtList": True,
                "sessionWebhook": "https://example.invalid/session",
                "sessionWebhookExpiredTime": 9999999999999,
            }
        )
        code, text = await adapter.process(callback)
        self.assertEqual((200, "OK"), (code, text))
        self.assertEqual("hello", adapter.dispatcher.calls[0]["text"])  # type: ignore[attr-defined]
        self.assertEqual(
            "corp-1:staff-1",
            adapter.dispatcher.calls[0]["identity"].platform_user_id,  # type: ignore[attr-defined]
        )

    async def test_group_without_at_is_acked_but_not_dispatched(self) -> None:
        with patch.dict(os.environ, {"DINGTALK_GROUP_MODE": "mentions"}):
            adapter = DingTalkAdapter(SimpleNamespace())  # type: ignore[arg-type]
        adapter.dispatcher = _CapturingDispatcher()  # type: ignore[assignment]
        callback = SimpleNamespace(
            data={
                "msgId": "msg-2",
                "msgtype": "text",
                "text": {"content": "ambient"},
                "senderId": "opaque",
                "conversationId": "cid-1",
                "conversationType": "2",
                "isInAtList": False,
                "sessionWebhook": "https://example.invalid/session",
            }
        )
        self.assertEqual((200, "OK"), await adapter.process(callback))
        self.assertEqual([], adapter.dispatcher.calls)  # type: ignore[attr-defined]


class _FakeQQApi:
    def __init__(self) -> None:
        self.sent: list[tuple[tuple, dict]] = []
        self.typing: list[tuple[str, str]] = []

    async def send_text(self, *args: object, **kwargs: object) -> dict:
        self.sent.append((args, kwargs))
        return {}

    async def send_typing(self, chat_id: str, message_id: str) -> None:
        self.typing.append((chat_id, message_id))


class _FakeQQParser:
    def __init__(self, event: object) -> None:
        self.event = event

    def parse(self, _event_type: str, _raw: dict) -> object:
        return self.event


class QQAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_group_at_event_is_queued_with_scene_scoped_identity(self) -> None:
        event = SimpleNamespace(
            event_type="GROUP_AT_MESSAGE_CREATE",
            chat_scope="group",
            chat_id="group-openid",
            user_id="member-openid",
            user_name=None,
            content="hello",
            message_id="msg-1",
        )
        api = _FakeQQApi()
        with patch.dict(
            os.environ,
            {"QQ_GROUP_MODE": "mentions", "QQ_ALLOWED_SCOPES": "c2c,group,guild"},
        ):
            adapter = QQAdapter(api, SimpleNamespace(), _FakeQQParser(event))  # type: ignore[arg-type]
        adapter.dispatcher = _CapturingDispatcher()  # type: ignore[assignment]
        await adapter.on_message_event(event.event_type, {})
        call = adapter.dispatcher.calls[0]  # type: ignore[attr-defined]
        self.assertEqual("group:member-openid", call["identity"].platform_user_id)
        await call["send_text"]("reply")
        self.assertEqual(("group", "group-openid", "reply"), api.sent[0][0])

    async def test_plain_guild_event_is_rejected_in_mentions_mode(self) -> None:
        event = SimpleNamespace(
            event_type="GUILD_MESSAGE_CREATE",
            chat_scope="guild",
            chat_id="channel-1",
            user_id="user-1",
            content="ambient",
            message_id="msg-2",
        )
        with patch.dict(os.environ, {"QQ_GROUP_MODE": "mentions"}):
            adapter = QQAdapter(_FakeQQApi(), SimpleNamespace(), _FakeQQParser(event))  # type: ignore[arg-type]
        adapter.dispatcher = _CapturingDispatcher()  # type: ignore[assignment]
        await adapter.on_message_event(event.event_type, {})
        self.assertEqual([], adapter.dispatcher.calls)  # type: ignore[attr-defined]


class _FakeLineParser:
    def __init__(self, events: list[object]) -> None:
        self.events = events

    def parse(self, _body: str, _signature: str) -> list[object]:
        return self.events


class _FakeLineApi:
    def __init__(self) -> None:
        self.replies: list[object] = []
        self.pushes: list[object] = []

    async def reply_message(self, request: object) -> object:
        self.replies.append(request)
        return SimpleNamespace()

    async def push_message(self, request: object) -> object:
        self.pushes.append(request)
        return SimpleNamespace()


def _line_event(*, group: bool = False, mentioned: bool = False) -> object:
    source = (
        SimpleNamespace(type="group", group_id="G1", room_id=None, user_id="U1")
        if group
        else SimpleNamespace(type="user", user_id="U1")
    )
    mention = SimpleNamespace(
        mentionees=[SimpleNamespace(index=0, length=4, is_self=True)] if mentioned else []
    )
    return SimpleNamespace(
        type="message",
        webhook_event_id="evt-1",
        reply_token="reply-token",
        source=source,
        message=SimpleNamespace(type="text", id="m1", text="@bot hello" if mentioned else "hello", mention=mention),
    )


class LineAdapterTests(unittest.IsolatedAsyncioTestCase):
    def test_signature_uses_raw_body_hmac(self) -> None:
        body = b'{"events":[]}'
        signature = base64.b64encode(
            hmac.new(b"secret", body, hashlib.sha256).digest()
        ).decode("ascii")
        self.assertTrue(verify_line_signature(body, signature, "secret"))
        self.assertFalse(verify_line_signature(body + b" ", signature, "secret"))

    @unittest.skipUnless(line_module._LINE_AVAILABLE, "line-bot-sdk not installed")
    async def test_valid_webhook_is_queued_then_reply_token_falls_back_to_push(self) -> None:
        event = _line_event()
        api = _FakeLineApi()
        body = json.dumps({"events": []}).encode()
        signature = base64.b64encode(
            hmac.new(b"secret", body, hashlib.sha256).digest()
        ).decode("ascii")
        with patch.dict(
            os.environ,
            {
                "LINE_GROUP_MODE": "mentions",
                "LINE_ALLOWED_USERS": "",
                "LINE_ALLOWED_CHATS": "",
            },
        ):
            adapter = LineAdapter(
                "secret",
                api,
                SimpleNamespace(),
                SimpleNamespace(),  # type: ignore[arg-type]
                parser=_FakeLineParser([event]),
            )
        adapter.dispatcher = _CapturingDispatcher()  # type: ignore[assignment]
        self.assertEqual(1, adapter.accept_webhook(body, signature))
        send = adapter.dispatcher.calls[0]["send_text"]  # type: ignore[attr-defined]
        await send("first")
        await send("second")
        self.assertEqual(1, len(api.replies))
        self.assertEqual(1, len(api.pushes))

    async def test_group_requires_structured_self_mention(self) -> None:
        body = b"{}"
        signature = base64.b64encode(
            hmac.new(b"secret", body, hashlib.sha256).digest()
        ).decode("ascii")
        events = [_line_event(group=True, mentioned=False)]
        with patch.dict(os.environ, {"LINE_GROUP_MODE": "mentions"}):
            adapter = LineAdapter(
                "secret",
                _FakeLineApi(),
                SimpleNamespace(),
                SimpleNamespace(),  # type: ignore[arg-type]
                parser=_FakeLineParser(events),
            )
        adapter.dispatcher = _CapturingDispatcher()  # type: ignore[assignment]
        self.assertEqual(0, adapter.accept_webhook(body, signature))


if __name__ == "__main__":
    unittest.main()
