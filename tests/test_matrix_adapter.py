from __future__ import annotations

import os
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    from nio import MatrixRoom, RoomEncryptedImage, RoomMessageImage, RoomMessageText
    from nio.crypto.attachments import encrypt_attachment

    NIO_AVAILABLE = True
except ModuleNotFoundError:
    MatrixRoom = RoomEncryptedImage = RoomMessageImage = RoomMessageText = object
    encrypt_attachment = None
    NIO_AVAILABLE = False

from src.bot.matrix_adapter import (
    MatrixAdapter,
    _authenticate_matrix_client,
    start_matrix_adapter,
)


class _FakeClient:
    def __init__(self) -> None:
        self.user_id = "@alona:example.org"
        self.callbacks: list[tuple[object, type]] = []
        self.download_response = SimpleNamespace(
            body=b"image", content_type="image/png"
        )
        self.sent: list[tuple[str, dict]] = []
        self.restored: tuple[str, str, str] | None = None
        self.sync_kwargs: dict | None = None
        self.closed = False

    def add_event_callback(self, callback: object, event_type: type) -> None:
        self.callbacks.append((callback, event_type))

    def restore_login(self, user_id: str, device_id: str, token: str) -> None:
        self.restored = (user_id, device_id, token)

    async def download(self, _url: str) -> object:
        return self.download_response

    async def room_send(
        self, *, room_id: str, message_type: str, content: dict
    ) -> object:
        self.sent.append((room_id, content))
        return SimpleNamespace()

    async def sync_forever(self, **kwargs: object) -> None:
        self.sync_kwargs = kwargs

    async def close(self) -> None:
        self.closed = True


class _CapturingDispatcher:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def submit(self, **kwargs: object) -> None:
        self.calls.append(kwargs)

    async def close(self) -> None:
        return None


def _room(member_count: int) -> MatrixRoom:
    room = MatrixRoom("!room:example.org", "@alona:example.org")
    room.add_member("@alona:example.org", "Alona", None)
    if member_count >= 2:
        room.add_member("@teacher:example.org", "Teacher", None)
    for index in range(2, member_count):
        room.add_member(f"@student{index}:example.org", f"Student {index}", None)
    return room


def _text_event(body: str, timestamp: int | None = None, *, mention=False):
    content: dict[str, object] = {"msgtype": "m.text", "body": body}
    if mention:
        content["m.mentions"] = {"user_ids": ["@alona:example.org"]}
    return RoomMessageText.from_dict(
        {
            "type": "m.room.message",
            "event_id": "$event",
            "sender": "@teacher:example.org",
            "origin_server_ts": timestamp or int(time.time() * 1000),
            "content": content,
        }
    )


@unittest.skipUnless(NIO_AVAILABLE, "matrix-nio is not installed")
class MatrixAdapterTests(unittest.IsolatedAsyncioTestCase):
    def _adapter(self, mode: str = "direct") -> tuple[MatrixAdapter, _FakeClient]:
        client = _FakeClient()
        with patch.dict(
            os.environ,
            {
                "MATRIX_ROOM_MODE": mode,
                "MATRIX_ALLOWED_ROOMS": "",
                "MATRIX_ALLOWED_USERS": "",
                "MATRIX_AUTO_JOIN_INVITES": "false",
                "MATRIX_REPLAY_OLD_MESSAGES": "false",
            },
        ):
            adapter = MatrixAdapter(client, SimpleNamespace())  # type: ignore[arg-type]
        adapter.dispatcher = _CapturingDispatcher()  # type: ignore[assignment]
        return adapter, client

    async def test_access_token_uses_nio_restore_login(self) -> None:
        client = _FakeClient()
        with patch.dict(
            os.environ,
            {"MATRIX_ACCESS_TOKEN": "secret", "MATRIX_DEVICE_ID": "DEVICE"},
        ):
            await _authenticate_matrix_client(client)
        self.assertEqual(
            ("@alona:example.org", "DEVICE", "secret"), client.restored
        )

    async def test_start_registers_callbacks_syncs_and_closes_client(self) -> None:
        client = _FakeClient()
        runtime = SimpleNamespace()
        with (
            patch.dict(
                os.environ,
                {
                    "MATRIX_HOMESERVER": "https://matrix.example.org",
                    "MATRIX_USER_ID": "@alona:example.org",
                    "MATRIX_ACCESS_TOKEN": "secret",
                    "MATRIX_DEVICE_ID": "DEVICE",
                    "MATRIX_ROOM_MODE": "direct",
                    "MATRIX_AUTO_JOIN_INVITES": "false",
                    "MATRIX_WHITELIST_ENABLED": "false",
                },
            ),
            patch("src.bot.matrix_adapter.AsyncClient", return_value=client),
        ):
            await start_matrix_adapter(runtime)  # type: ignore[arg-type]
        callback_types = {event_type for _callback, event_type in client.callbacks}
        self.assertIn(RoomMessageText, callback_types)
        self.assertIn(RoomMessageImage, callback_types)
        self.assertIn(RoomEncryptedImage, callback_types)
        self.assertEqual({"timeout": 30000, "full_state": True}, client.sync_kwargs)
        self.assertTrue(client.closed)

    async def test_direct_mode_accepts_unnamed_two_member_room(self) -> None:
        adapter, _client = self._adapter("direct")
        await adapter._on_text(_room(2), _text_event("hello"))
        self.assertEqual(1, len(adapter.dispatcher.calls))  # type: ignore[attr-defined]

    async def test_direct_mode_rejects_unnamed_multi_member_room(self) -> None:
        adapter, _client = self._adapter("direct")
        await adapter._on_text(_room(3), _text_event("hello"))
        self.assertEqual([], adapter.dispatcher.calls)  # type: ignore[attr-defined]

    async def test_mentions_mode_understands_structured_matrix_mention(self) -> None:
        adapter, _client = self._adapter("mentions")
        await adapter._on_text(
            _room(3), _text_event("Alona: hello", mention=True)
        )
        self.assertEqual(1, len(adapter.dispatcher.calls))  # type: ignore[attr-defined]

    async def test_old_timeline_event_is_not_dispatched(self) -> None:
        adapter, _client = self._adapter("all")
        await adapter._on_text(_room(3), _text_event("old", timestamp=1))
        self.assertEqual([], adapter.dispatcher.calls)  # type: ignore[attr-defined]

    async def test_plain_image_is_downloaded_and_dispatched(self) -> None:
        adapter, client = self._adapter("direct")
        source = {
            "type": "m.room.message",
            "event_id": "$image",
            "sender": "@teacher:example.org",
            "origin_server_ts": int(time.time() * 1000),
            "content": {
                "msgtype": "m.image",
                "body": "image.png",
                "url": "mxc://example.org/image",
                "info": {"mimetype": "image/png", "size": 5},
            },
        }
        event = RoomMessageImage(source, "mxc://example.org/image", "image.png")
        await adapter._on_image(_room(2), event)
        images = adapter.dispatcher.calls[0]["images"]  # type: ignore[attr-defined]
        self.assertEqual([(b"image", "image/png")], images)
        self.assertEqual([], client.sent)

    async def test_encrypted_image_is_decrypted_before_dispatch(self) -> None:
        adapter, client = self._adapter("direct")
        plaintext = b"private image bytes"
        ciphertext, file_info = encrypt_attachment(plaintext)
        client.download_response = SimpleNamespace(
            body=ciphertext, content_type="application/octet-stream"
        )
        source = {
            "type": "m.room.message",
            "event_id": "$encrypted-image",
            "sender": "@teacher:example.org",
            "origin_server_ts": int(time.time() * 1000),
            "content": {
                "msgtype": "m.image",
                "body": "image.png",
                "file": {"url": "mxc://example.org/image", **file_info},
                "info": {"mimetype": "image/png", "size": len(ciphertext)},
            },
        }
        event = RoomEncryptedImage(
            source,
            "mxc://example.org/image",
            "image.png",
            file_info["key"],
            file_info["hashes"],
            file_info["iv"],
            "image/png",
        )
        await adapter._on_image(_room(2), event)
        images = adapter.dispatcher.calls[0]["images"]  # type: ignore[attr-defined]
        self.assertEqual([(plaintext, "image/png")], images)


if __name__ == "__main__":
    unittest.main()
