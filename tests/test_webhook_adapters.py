from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from src.bot.google_chat_adapter import (
    GoogleChatAdapter,
    GoogleChatApiSender,
    create_google_chat_sender,
    google_chat_identity,
    google_chat_space_allows,
    start_google_chat_adapter,
)
from src.bot.messenger_adapter import (
    MessengerAdapter,
    messenger_chat_allows,
    messenger_identity,
    start_messenger_adapter,
)
from src.bot.webhook_support import stop_webhook_components, verify_meta_signature
from src.bot.whatsapp_adapter import (
    WhatsAppAdapter,
    start_whatsapp_adapter,
    whatsapp_chat_allows,
    whatsapp_identity,
)


class _FakeRuntime:
    def __init__(self) -> None:
        self.closed = False

    async def process_message(self, **_kwargs: object) -> None:
        return None

    async def close(self) -> None:
        self.closed = True


class _FakePoster:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.closed = 0

    async def post_json(self, url: str, **kwargs: object) -> dict:
        self.calls.append({"url": url, **kwargs})
        return {"ok": True}

    async def close(self) -> None:
        self.closed += 1


class _FakeChatSender:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.closed = 0

    async def send_message(self, **kwargs: object) -> dict:
        self.calls.append(dict(kwargs))
        return {"name": "sent"}

    async def close(self) -> None:
        self.closed += 1


class _CapturingDispatcher:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.closed = 0
        self._events: set[tuple[object, object]] = set()

    def submit(self, **kwargs: object) -> bool:
        key = (kwargs.get("event_scope"), kwargs.get("event_id"))
        if key[1] and key in self._events:
            return False
        if key[1]:
            self._events.add(key)
        self.calls.append(dict(kwargs))
        return True

    async def close(self) -> None:
        self.closed += 1


class _FakeServer:
    def __init__(self) -> None:
        self.closed = 0

    async def close(self) -> None:
        self.closed += 1


class _FakeServerStarter:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.server = _FakeServer()

    async def __call__(self, **kwargs: object) -> _FakeServer:
        self.calls.append(dict(kwargs))
        return self.server


def _json_bytes(value: dict) -> bytes:
    return json.dumps(value, separators=(",", ":")).encode("utf-8")


def _meta_signature(secret: str, raw: bytes) -> str:
    return "sha256=" + hmac.new(
        secret.encode("utf-8"), raw, hashlib.sha256
    ).hexdigest()


def _whatsapp_payload(*, event_id: str = "wamid.1", text: str = "hello") -> dict:
    return {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "waba-1",
                "changes": [
                    {
                        "field": "messages",
                        "value": {
                            "metadata": {"phone_number_id": "phone-1"},
                            "contacts": [
                                {
                                    "wa_id": "15551234567",
                                    "profile": {"name": "Teacher"},
                                }
                            ],
                            "messages": [
                                {
                                    "id": event_id,
                                    "from": "15551234567",
                                    "type": "text",
                                    "text": {"body": text},
                                }
                            ],
                        },
                    }
                ],
            }
        ],
    }


def _messenger_payload(*, event_id: str = "mid.1", text: str = "hello") -> dict:
    return {
        "object": "page",
        "entry": [
            {
                "id": "page-1",
                "messaging": [
                    {
                        "sender": {"id": "psid-1"},
                        "recipient": {"id": "page-1"},
                        "message": {"mid": event_id, "text": text},
                    }
                ],
            }
        ],
    }


def _google_payload(*, message_name: str = "spaces/s1/messages/m1") -> dict:
    return {
        "type": "MESSAGE",
        "user": {
            "name": "users/123",
            "displayName": "Teacher",
            "domainId": "customer-1",
            "type": "HUMAN",
        },
        "space": {"name": "spaces/s1", "type": "DIRECT_MESSAGE"},
        "message": {
            "name": message_name,
            "text": "@Alona hello",
            "argumentText": "hello",
            "thread": {"name": "spaces/s1/threads/t1"},
        },
    }


class WebhookSecurityAndIdentityTests(unittest.TestCase):
    def test_meta_signature_covers_exact_raw_body(self) -> None:
        raw = b'{"same":"value"}'
        signature = _meta_signature("secret", raw)
        self.assertTrue(verify_meta_signature(raw, signature, "secret"))
        self.assertFalse(
            verify_meta_signature(b'{ "same":"value"}', signature, "secret")
        )
        self.assertFalse(verify_meta_signature(raw, "sha256=bad", "secret"))

    def test_stable_ids_include_business_or_tenant_scope(self) -> None:
        self.assertEqual(
            "whatsapp:phone-1:15551234567",
            whatsapp_identity("phone-1", "15551234567").key,
        )
        self.assertEqual(
            "messenger:page-1:psid-1",
            messenger_identity("page-1", "psid-1").key,
        )
        self.assertEqual(
            "google_chat:project-1:customer-1:users/123",
            google_chat_identity(
                "project-1", "customer-1", "users/123"
            ).key,
        )

    def test_group_policies_are_conservative(self) -> None:
        self.assertTrue(whatsapp_chat_allows(is_group=False, mode="direct"))
        self.assertFalse(whatsapp_chat_allows(is_group=True, mode="direct"))
        self.assertFalse(messenger_chat_allows(is_group=True, mode="direct"))
        self.assertFalse(
            google_chat_space_allows(is_direct=False, mode="direct")
        )
        self.assertTrue(
            google_chat_space_allows(is_direct=False, mode="mentions")
        )
        with self.assertRaises(ValueError):
            google_chat_space_allows(is_direct=False, mode="all")


class MetaWebhookAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_whatsapp_verifies_before_json_then_acks_in_background(self) -> None:
        poster = _FakePoster()
        adapter = WhatsAppAdapter(
            _FakeRuntime(),  # type: ignore[arg-type]
            app_secret="secret",
            verify_token="verify",
            access_token="access",
            graph_api_version="v25.0",
            phone_number_id="phone-1",
            poster=poster,
            allowed_users={"15551234567"},
            message_limit=5,
        )
        dispatcher = _CapturingDispatcher()
        adapter.dispatcher = dispatcher  # type: ignore[assignment]

        unauthorized = await adapter.handle_webhook(
            "POST", {"X-Hub-Signature-256": "bad"}, {}, b"not-json"
        )
        self.assertEqual(401, unauthorized.status)

        raw = _json_bytes(_whatsapp_payload())
        response = await adapter.handle_webhook(
            "POST",
            {"x-hub-signature-256": _meta_signature("secret", raw)},
            {},
            raw,
        )
        self.assertEqual(200, response.status)
        await adapter.wait_background()
        self.assertEqual(1, len(dispatcher.calls))
        call = dispatcher.calls[0]
        self.assertEqual("whatsapp:phone-1:15551234567", call["identity"].key)
        self.assertEqual("hello", call["text"])

        duplicate = await adapter.handle_webhook(
            "POST",
            {"X-Hub-Signature-256": _meta_signature("secret", raw)},
            {},
            raw,
        )
        self.assertEqual(200, duplicate.status)
        await adapter.wait_background()
        self.assertEqual(1, len(dispatcher.calls))

        await call["send_text"]("abcdefghij")
        self.assertEqual(["abcde", "fghij"], [
            item["payload"]["text"]["body"] for item in poster.calls
        ])
        self.assertTrue(all("access" not in item["url"] for item in poster.calls))
        self.assertTrue(all(
            item["headers"]["Authorization"] == "Bearer access"
            for item in poster.calls
        ))
        await adapter.close()
        await adapter.close()
        self.assertEqual(1, poster.closed)

    async def test_whatsapp_subscription_verification(self) -> None:
        adapter = WhatsAppAdapter(
            _FakeRuntime(),  # type: ignore[arg-type]
            app_secret="secret",
            verify_token="verify",
            access_token="access",
            graph_api_version="v25.0",
            phone_number_id="phone-1",
            poster=_FakePoster(),
        )
        accepted = await adapter.handle_webhook(
            "GET",
            {},
            {
                "hub.mode": "subscribe",
                "hub.verify_token": "verify",
                "hub.challenge": "challenge",
            },
            b"",
        )
        rejected = await adapter.handle_webhook(
            "GET",
            {},
            {"hub.mode": "subscribe", "hub.verify_token": "wrong"},
            b"",
        )
        self.assertEqual(b"challenge", accepted.body)
        self.assertEqual(403, rejected.status)
        await adapter.close()

    async def test_messenger_signature_scope_dedupe_send_and_typing(self) -> None:
        poster = _FakePoster()
        adapter = MessengerAdapter(
            _FakeRuntime(),  # type: ignore[arg-type]
            app_secret="secret",
            verify_token="verify",
            page_access_token="page-token",
            graph_api_version="v25.0",
            page_id="page-1",
            poster=poster,
            allowed_users={"psid-1"},
            message_limit=5,
        )
        dispatcher = _CapturingDispatcher()
        adapter.dispatcher = dispatcher  # type: ignore[assignment]
        raw = _json_bytes(_messenger_payload())

        bad = await adapter.handle_webhook(
            "POST", {"X-Hub-Signature-256": "bad"}, {}, raw
        )
        self.assertEqual(401, bad.status)
        good = await adapter.handle_webhook(
            "POST",
            {"X-Hub-Signature-256": _meta_signature("secret", raw)},
            {},
            raw,
        )
        self.assertEqual(200, good.status)
        await adapter.wait_background()
        self.assertEqual(1, len(dispatcher.calls))
        call = dispatcher.calls[0]
        self.assertEqual("messenger:page-1:psid-1", call["identity"].key)

        await call["send_text"]("abcdefghij")
        await call["set_typing"](True)
        self.assertEqual(["abcde", "fghij"], [
            item["payload"]["message"]["text"] for item in poster.calls[:2]
        ])
        self.assertEqual("typing_on", poster.calls[2]["payload"]["sender_action"])
        self.assertTrue(all("page-token" not in item["url"] for item in poster.calls))
        await adapter.close()


class GoogleChatWebhookAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_sdk_request_shape_and_transport_close(self) -> None:
        try:
            from google.apps import chat_v1
        except ModuleNotFoundError:
            self.skipTest("google-apps-chat is not installed")

        class FakeTransport:
            def __init__(self) -> None:
                self.closed = 0

            def close(self) -> None:
                self.closed += 1

        class ProtoCheckingClient:
            def __init__(self) -> None:
                self.transport = FakeTransport()
                self.request = None

            async def create_message(self, *, request: dict) -> object:
                self.request = chat_v1.CreateMessageRequest(request)
                return chat_v1.Message(name="spaces/s1/messages/sent")

        client = ProtoCheckingClient()
        sender = GoogleChatApiSender(client)
        await sender.send_message(
            space_name="spaces/s1",
            thread_name="spaces/s1/threads/t1",
            text="hello",
        )

        self.assertEqual("spaces/s1", client.request.parent)
        self.assertEqual("hello", client.request.message.text)
        self.assertEqual(
            "spaces/s1/threads/t1", client.request.message.thread.name
        )
        self.assertEqual(
            chat_v1.CreateMessageRequest.MessageReplyOption.REPLY_MESSAGE_OR_FAIL,
            client.request.message_reply_option,
        )
        await sender.close()
        await sender.close()
        self.assertEqual(1, client.transport.closed)

    async def test_oidc_precedes_json_and_reply_is_async_chat_api(self) -> None:
        verifier_tokens: list[str] = []

        async def reject(token: str) -> bool:
            verifier_tokens.append(token)
            return False

        sender = _FakeChatSender()
        adapter = GoogleChatAdapter(
            _FakeRuntime(),  # type: ignore[arg-type]
            oidc_verifier=reject,
            sender=sender,
            app_scope="project-1",
        )
        unauthorized = await adapter.handle_webhook(
            "POST", {"Authorization": "Bearer forged"}, {}, b"not-json"
        )
        self.assertEqual(401, unauthorized.status)
        self.assertEqual(["forged"], verifier_tokens)
        await adapter.close()

        async def accept(_token: str) -> bool:
            return True

        sender = _FakeChatSender()
        adapter = GoogleChatAdapter(
            _FakeRuntime(),  # type: ignore[arg-type]
            oidc_verifier=accept,
            sender=sender,
            app_scope="project-1",
            allowed_users={"users/123"},
            allowed_spaces={"spaces/s1"},
            allowed_domains={"customer-1"},
            message_limit=5,
        )
        dispatcher = _CapturingDispatcher()
        adapter.dispatcher = dispatcher  # type: ignore[assignment]
        raw = _json_bytes(_google_payload())
        response = await adapter.handle_webhook(
            "POST", {"authorization": "Bearer valid"}, {}, raw
        )
        self.assertEqual(200, response.status)
        self.assertEqual(b"{}", response.body)
        self.assertEqual([], sender.calls)
        await adapter.wait_background()
        self.assertEqual(1, len(dispatcher.calls))
        call = dispatcher.calls[0]
        self.assertEqual(
            "google_chat:project-1:customer-1:users/123", call["identity"].key
        )
        self.assertIn("spaces/s1/threads/t1", call["conversation_key"])

        await call["send_text"]("abcdefghij")
        self.assertEqual(["abcde", "fghij"], [
            item["text"] for item in sender.calls
        ])
        self.assertTrue(all(
            item["thread_name"] == "spaces/s1/threads/t1"
            for item in sender.calls
        ))

        await adapter.handle_webhook(
            "POST", {"Authorization": "Bearer valid"}, {}, raw
        )
        await adapter.wait_background()
        self.assertEqual(1, len(dispatcher.calls))
        await adapter.close()

    async def test_direct_policy_rejects_group_interaction(self) -> None:
        async def accept(_token: str) -> bool:
            return True

        adapter = GoogleChatAdapter(
            _FakeRuntime(),  # type: ignore[arg-type]
            oidc_verifier=accept,
            sender=_FakeChatSender(),
            app_scope="project-1",
            space_mode="direct",
        )
        dispatcher = _CapturingDispatcher()
        adapter.dispatcher = dispatcher  # type: ignore[assignment]
        payload = _google_payload()
        payload["space"]["type"] = "SPACE"
        raw = _json_bytes(payload)
        await adapter.handle_webhook(
            "POST", {"Authorization": "Bearer valid"}, {}, raw
        )
        await adapter.wait_background()
        self.assertEqual([], dispatcher.calls)
        await adapter.close()

    def test_missing_service_account_fails_before_optional_sdk_import(self) -> None:
        missing = Path(os.getcwd()) / "definitely-missing-service-account.json"
        with self.assertRaisesRegex(ValueError, "service-account file is missing"):
            create_google_chat_sender(str(missing))


class WebhookLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_cleanup_continues_when_server_close_fails(self) -> None:
        events: list[str] = []

        class FailingServer:
            async def close(self) -> None:
                events.append("server")
                raise RuntimeError("server close failed")

        class Closeable:
            def __init__(self, label: str) -> None:
                self.label = label

            async def close(self) -> None:
                events.append(self.label)

        with self.assertRaisesRegex(RuntimeError, "server close failed"):
            await stop_webhook_components(
                server=FailingServer(),
                adapter=Closeable("adapter"),
                runtime=Closeable("runtime"),
                owns_runtime=True,
            )
        self.assertEqual(["server", "adapter", "runtime"], events)

    async def test_all_three_start_and_close_with_fakes(self) -> None:
        stop = asyncio.Event()
        stop.set()

        runtime = _FakeRuntime()
        poster = _FakePoster()
        server = _FakeServerStarter()
        with patch.dict(
            os.environ,
            {
                "WHATSAPP_APP_SECRET": "secret",
                "WHATSAPP_VERIFY_TOKEN": "verify",
                "WHATSAPP_ACCESS_TOKEN": "access",
                "WHATSAPP_GRAPH_API_VERSION": "v25.0",
                "WHATSAPP_PHONE_NUMBER_ID": "phone-1",
                "WHATSAPP_WHITELIST_ENABLED": "false",
            },
            clear=True,
        ):
            await start_whatsapp_adapter(
                runtime, shutdown_event=stop, server_starter=server, poster=poster
            )  # type: ignore[arg-type]
        self.assertEqual(1, server.server.closed)
        self.assertEqual(1, poster.closed)
        self.assertFalse(runtime.closed)

        runtime = _FakeRuntime()
        poster = _FakePoster()
        server = _FakeServerStarter()
        with patch.dict(
            os.environ,
            {
                "MESSENGER_APP_SECRET": "secret",
                "MESSENGER_VERIFY_TOKEN": "verify",
                "MESSENGER_PAGE_ACCESS_TOKEN": "page-token",
                "MESSENGER_GRAPH_API_VERSION": "v25.0",
                "MESSENGER_PAGE_ID": "page-1",
                "MESSENGER_WHITELIST_ENABLED": "false",
            },
            clear=True,
        ):
            await start_messenger_adapter(
                runtime, shutdown_event=stop, server_starter=server, poster=poster
            )  # type: ignore[arg-type]
        self.assertEqual(1, server.server.closed)
        self.assertEqual(1, poster.closed)

        async def accept(_token: str) -> bool:
            return True

        runtime = _FakeRuntime()
        sender = _FakeChatSender()
        server = _FakeServerStarter()
        with patch.dict(
            os.environ,
            {
                "GOOGLE_CHAT_OIDC_AUDIENCE": "https://example.test/chat",
                "GOOGLE_CHAT_WHITELIST_ENABLED": "false",
            },
            clear=True,
        ):
            await start_google_chat_adapter(
                runtime,
                shutdown_event=stop,
                server_starter=server,
                oidc_verifier=accept,
                sender=sender,
            )  # type: ignore[arg-type]
        self.assertEqual(1, server.server.closed)
        self.assertEqual(1, sender.closed)
        self.assertEqual("/webhooks/google-chat", server.calls[0]["path"])


if __name__ == "__main__":
    unittest.main()
