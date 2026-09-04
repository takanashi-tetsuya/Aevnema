from __future__ import annotations

import asyncio
import os
import time
from typing import Any

from dotenv import load_dotenv

from src.bot.adapter_support import (
    AdapterRuntime,
    DebouncedDispatcher,
    PROJECT_ROOT,
    require_access_policy,
    split_message,
    whitelist_values,
)
from src.memory import PlatformIdentity
from src.utils.logger import setup_logger


logger = setup_logger(__name__)

try:
    from nio import (
        AsyncClient,
        DownloadError,
        InviteMemberEvent,
        LoginResponse,
        MatrixRoom,
        RoomEncryptedImage,
        RoomMessageImage,
        RoomMessageText,
        RoomSendError,
    )

    _NIO_AVAILABLE = True
except ModuleNotFoundError:  # Policy/identity helpers remain importable in tests.
    AsyncClient = Any  # type: ignore[assignment,misc]
    DownloadError = InviteMemberEvent = LoginResponse = Any  # type: ignore[misc]
    MatrixRoom = RoomMessageImage = RoomMessageText = Any  # type: ignore[misc]
    RoomEncryptedImage = Any  # type: ignore[misc]
    RoomSendError = Any  # type: ignore[assignment,misc]
    _NIO_AVAILABLE = False


def matrix_identity(user_id: str, display_name: str = "") -> PlatformIdentity:
    """Matrix's fully qualified user ID is stable across room display names."""

    return PlatformIdentity("matrix", user_id.strip(), display_name or user_id.strip())


def matrix_room_allows(
    *, room_is_group: bool, body: str, bot_user_id: str, mode: str
) -> bool:
    """Apply the configured group-room response policy."""

    normalized = mode.strip().casefold()
    if normalized not in {"direct", "mentions", "all"}:
        raise ValueError("MATRIX_ROOM_MODE must be direct, mentions, or all")
    if not room_is_group or normalized == "all":
        return True
    if normalized == "direct":
        return False
    return bot_user_id.casefold() in body.casefold()


def _csv_env(name: str) -> set[str]:
    return {
        value.strip()
        for value in os.getenv(name, "").split(",")
        if value.strip()
    }


def _matrix_room_is_group(room: Any) -> bool:
    """Treat only an established two-member room as a direct conversation.

    ``MatrixRoom.is_group`` does not mean "more than two members" in
    matrix-nio.  It means that the room has no explicit name/alias, which is
    also the normal shape of a direct-message room.  Member count is therefore
    the safer transport-level approximation until the application persists
    the account's ``m.direct`` mapping.
    """

    try:
        return int(room.joined_count) != 2
    except (AttributeError, TypeError, ValueError):
        # Unknown membership must not accidentally bypass the conservative
        # group-room policy.
        return True


class MatrixAdapter:
    """matrix-nio transport with conservative room and history policies."""

    def __init__(self, client: Any, runtime: AdapterRuntime):
        self.client = client
        self.runtime = runtime
        self.dispatcher = DebouncedDispatcher(runtime)
        self.started_at_ms = int(time.time() * 1000)
        self.room_mode = os.getenv("MATRIX_ROOM_MODE", "direct").strip().casefold()
        # Validate at startup instead of silently ignoring all rooms later.
        matrix_room_allows(
            room_is_group=False,
            body="",
            bot_user_id=str(client.user_id or ""),
            mode=self.room_mode,
        )
        self.allowed_rooms = whitelist_values("matrix", "MATRIX_ALLOWED_ROOMS")
        self.allowed_users = whitelist_values("matrix", "MATRIX_ALLOWED_USERS")
        client.add_event_callback(self._on_text, RoomMessageText)
        client.add_event_callback(self._on_image, RoomMessageImage)
        client.add_event_callback(self._on_image, RoomEncryptedImage)
        if os.getenv("MATRIX_AUTO_JOIN_INVITES", "false").casefold() == "true":
            client.add_event_callback(self._on_invite, InviteMemberEvent)

    def _accept_event(self, room: Any, event: Any, body: str) -> bool:
        sender = str(event.sender)
        if sender == str(self.client.user_id):
            return False
        if self.allowed_rooms and room.room_id not in self.allowed_rooms:
            return False
        if self.allowed_users and sender not in self.allowed_users:
            return False
        replay = os.getenv("MATRIX_REPLAY_OLD_MESSAGES", "false").casefold() == "true"
        timestamp = int(getattr(event, "server_timestamp", 0) or 0)
        startup_skew = int(os.getenv("MATRIX_STARTUP_EVENT_SKEW_MS", "5000"))
        if not replay and timestamp and timestamp < self.started_at_ms - startup_skew:
            return False
        room_is_group = _matrix_room_is_group(room)
        if (
            room_is_group
            and self.room_mode.strip().casefold() == "mentions"
            and self._has_structured_mention(event)
        ):
            return True
        return matrix_room_allows(
            room_is_group=room_is_group,
            body=body,
            bot_user_id=str(self.client.user_id),
            mode=self.room_mode,
        )

    def _has_structured_mention(self, event: Any) -> bool:
        source = getattr(event, "source", {}) or {}
        content = source.get("content", {}) if isinstance(source, dict) else {}
        mentions = content.get("m.mentions", {}) if isinstance(content, dict) else {}
        user_ids = mentions.get("user_ids", []) if isinstance(mentions, dict) else []
        return str(self.client.user_id or "") in user_ids

    def _identity(self, room: Any, event: Any) -> PlatformIdentity:
        sender = str(event.sender)
        try:
            display_name = str(room.user_name(sender) or sender)
        except Exception:
            display_name = sender
        return matrix_identity(sender, display_name)

    def _strip_mention(self, body: str) -> str:
        bot_user_id = str(self.client.user_id or "")
        if bot_user_id and bot_user_id.casefold() in body.casefold():
            start = body.casefold().find(bot_user_id.casefold())
            body = body[:start] + body[start + len(bot_user_id) :]
        return body.lstrip(" :,-\t")

    async def _send_text(self, room_id: str, text: str) -> None:
        limit = int(os.getenv("MATRIX_MESSAGE_LIMIT", "12000"))
        for chunk in split_message(text, limit):
            response = await self.client.room_send(
                room_id=room_id,
                message_type="m.room.message",
                content={"msgtype": "m.text", "body": chunk},
            )
            if isinstance(response, RoomSendError):
                raise RuntimeError(f"Matrix send failed: {response}")

    async def _set_typing(self, room_id: str, active: bool) -> None:
        await self.client.room_typing(
            room_id, typing_state=active, timeout=60_000
        )

    async def _on_text(self, room: Any, event: Any) -> None:
        body = str(event.body or "").strip()
        if not body or not self._accept_event(room, event, body):
            return
        if _matrix_room_is_group(room) and self.room_mode.casefold() == "mentions":
            body = self._strip_mention(body)
        identity = self._identity(room, event)
        room_id = str(room.room_id)
        self.dispatcher.submit(
            conversation_key=f"{room_id}:{identity.key}",
            identity=identity,
            text=body,
            send_text=lambda text: self._send_text(room_id, text),
            set_typing=lambda active: self._set_typing(room_id, active),
            event_id=str(getattr(event, "event_id", "") or "") or None,
            event_scope=str(getattr(self.client, "user_id", "") or "matrix-bot"),
        )

    async def _on_image(self, room: Any, event: Any) -> None:
        if not self._accept_event(room, event, ""):
            return
        source = getattr(event, "source", {}) or {}
        content = source.get("content", {}) if isinstance(source, dict) else {}
        info = content.get("info", {}) if isinstance(content, dict) else {}
        announced_size = int(info.get("size", 0) or 0) if isinstance(info, dict) else 0
        max_bytes = int(os.getenv("MATRIX_MAX_IMAGE_BYTES", str(10 * 1024 * 1024)))
        if announced_size and announced_size > max_bytes:
            await self._send_text(
                str(room.room_id),
                f"这张图片超过了 Matrix 适配器的 {max_bytes} 字节下载上限。",
            )
            return

        response = await self.client.download(str(event.url))
        if isinstance(response, DownloadError):
            await self._send_text(str(room.room_id), "Matrix 图片下载失败，暂时无法查看。")
            return
        payload = bytes(getattr(response, "body", b""))
        if isinstance(event, RoomEncryptedImage):
            try:
                from nio.crypto.attachments import decrypt_attachment

                payload = decrypt_attachment(
                    payload,
                    str(event.key["k"]),
                    str(event.hashes["sha256"]),
                    str(event.iv),
                )
            except Exception:
                logger.exception("Matrix 加密图片解密失败：%s", room.room_id)
                await self._send_text(
                    str(room.room_id), "Matrix 加密图片解密失败，暂时无法查看。"
                )
                return
        if not payload or len(payload) > max_bytes:
            await self._send_text(str(room.room_id), "Matrix 图片为空或超过下载上限。")
            return
        mime_type = str(
            (info.get("mimetype") if isinstance(info, dict) else "")
            or getattr(event, "mimetype", "")
            or getattr(response, "content_type", "")
            or "application/octet-stream"
        ).split(";", 1)[0]
        if not mime_type.startswith("image/"):
            await self._send_text(str(room.room_id), "收到的媒体不是受支持的图片类型。")
            return

        identity = self._identity(room, event)
        room_id = str(room.room_id)
        self.dispatcher.submit(
            conversation_key=f"{room_id}:{identity.key}",
            identity=identity,
            text="",
            images=[(payload, mime_type)],
            send_text=lambda text: self._send_text(room_id, text),
            set_typing=lambda active: self._set_typing(room_id, active),
            event_id=str(getattr(event, "event_id", "") or "") or None,
            event_scope=str(getattr(self.client, "user_id", "") or "matrix-bot"),
        )

    async def _on_invite(self, room: Any, event: Any) -> None:
        if str(getattr(event, "state_key", "")) != str(self.client.user_id):
            return
        if str(getattr(event, "membership", "")) != "invite":
            return
        if self.allowed_rooms and room.room_id not in self.allowed_rooms:
            logger.warning("拒绝自动加入未在 MATRIX_ALLOWED_ROOMS 中的房间：%s", room.room_id)
            return
        response = await self.client.join(room.room_id)
        logger.info("Matrix 邀请处理结果（%s）：%s", room.room_id, response)

    async def close(self) -> None:
        await self.dispatcher.close()


async def _authenticate_matrix_client(client: Any) -> None:
    access_token = os.getenv("MATRIX_ACCESS_TOKEN", "").strip()
    if access_token:
        device_id = os.getenv("MATRIX_DEVICE_ID", "").strip()
        client.restore_login(str(client.user_id), device_id, access_token)
        return

    password = os.getenv("MATRIX_PASSWORD", "")
    if not password:
        raise ValueError("configure MATRIX_ACCESS_TOKEN or MATRIX_PASSWORD")
    response = await client.login(
        password,
        device_name=os.getenv("MATRIX_DEVICE_NAME", "chat_bot matrix adapter"),
    )
    if not isinstance(response, LoginResponse):
        raise RuntimeError(f"Matrix login failed: {response}")


async def start_matrix_adapter(runtime: AdapterRuntime | None = None) -> None:
    """Start the Matrix adapter and sync until cancelled."""

    load_dotenv(PROJECT_ROOT / ".env")
    if not _NIO_AVAILABLE:
        raise RuntimeError("Matrix adapter requires matrix-nio>=0.26,<0.27")
    homeserver = os.getenv("MATRIX_HOMESERVER", "").strip()
    user_id = os.getenv("MATRIX_USER_ID", "").strip()
    if not homeserver or not user_id:
        raise ValueError("MATRIX_HOMESERVER and MATRIX_USER_ID must be configured")
    require_access_policy("matrix", "MATRIX_ALLOWED_USERS", "MATRIX_ALLOWED_ROOMS")

    owns_runtime = runtime is None
    active_runtime = runtime or await AdapterRuntime.create()
    client: Any | None = None
    adapter: MatrixAdapter | None = None
    try:
        client = AsyncClient(homeserver, user_id)
        adapter = MatrixAdapter(client, active_runtime)
        logger.info("Matrix Adapter 正在连接……")
        await _authenticate_matrix_client(client)
        logger.info("Matrix Adapter 已登录：%s", client.user_id)
        await client.sync_forever(
            timeout=int(os.getenv("MATRIX_SYNC_TIMEOUT_MS", "30000")),
            full_state=True,
        )
    finally:
        logger.info("正在停止 Matrix Adapter 并提交待处理对话批次……")
        if adapter is not None:
            await adapter.close()
        if client is not None:
            await client.close()
        if owns_runtime:
            await active_runtime.close()


if __name__ == "__main__":
    asyncio.run(start_matrix_adapter())
