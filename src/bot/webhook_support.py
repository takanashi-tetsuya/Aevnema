from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import json
import os
from typing import Any, Awaitable, Callable, Mapping, Protocol

from src.utils.logger import setup_logger


logger = setup_logger(__name__)

JsonObject = dict[str, Any]
WebhookHandler = Callable[
    [str, Mapping[str, str], Mapping[str, str], bytes],
    Awaitable["WebhookResponse"],
]


class JsonPoster(Protocol):
    async def post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        params: Mapping[str, str] | None = None,
    ) -> Any: ...

    async def close(self) -> None: ...


class WebhookServer(Protocol):
    async def close(self) -> None: ...


@dataclass(slots=True, frozen=True)
class WebhookResponse:
    status: int = 200
    body: bytes = b""
    content_type: str = "text/plain; charset=utf-8"

    @classmethod
    def text(cls, text: str, status: int = 200) -> "WebhookResponse":
        return cls(status, text.encode("utf-8"))

    @classmethod
    def json(
        cls, value: Mapping[str, Any], status: int = 200
    ) -> "WebhookResponse":
        return cls(
            status,
            json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            ),
            "application/json; charset=utf-8",
        )


def header_value(headers: Mapping[str, str], name: str) -> str:
    wanted = name.casefold()
    for key, value in headers.items():
        if str(key).casefold() == wanted:
            return str(value).strip()
    return ""


def bearer_token(headers: Mapping[str, str]) -> str:
    value = header_value(headers, "Authorization")
    scheme, separator, token = value.partition(" ")
    if not separator or scheme.casefold() != "bearer":
        return ""
    return token.strip()


def verify_meta_signature(raw_body: bytes, signature: str, app_secret: str) -> bool:
    """Verify Meta's X-Hub-Signature-256 over the untouched request bytes."""

    prefix = "sha256="
    if not app_secret or not signature.startswith(prefix):
        return False
    supplied = signature[len(prefix) :]
    if len(supplied) != hashlib.sha256().digest_size * 2:
        return False
    try:
        int(supplied, 16)
    except ValueError:
        return False
    expected = hmac.new(
        app_secret.encode("utf-8"), raw_body, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, supplied.casefold())


def parse_json_object(raw_body: bytes, *, max_bytes: int) -> JsonObject:
    if len(raw_body) > max_bytes:
        raise ValueError("webhook payload is too large")
    try:
        value = json.loads(raw_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("webhook payload is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("webhook payload must be a JSON object")
    return value


def csv_env(name: str) -> set[str]:
    return {part.strip() for part in os.getenv(name, "").split(",") if part.strip()}


def required_env(name: str, *fallback_names: str) -> str:
    for candidate in (name, *fallback_names):
        value = os.getenv(candidate, "").strip()
        if value:
            return value
    alternatives = ", ".join((name, *fallback_names))
    raise ValueError(f"required environment variable is missing: {alternatives}")


def integer_env(
    name: str, default: str, *, minimum: int = 1, maximum: int | None = None
) -> int:
    try:
        value = int(os.getenv(name, default))
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < minimum or (maximum is not None and value > maximum):
        suffix = f"..{maximum}" if maximum is not None else " or greater"
        raise ValueError(f"{name} must be {minimum}{suffix}")
    return value


def event_fingerprint(scope: str, event: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        event, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return f"{scope}:sha256:{hashlib.sha256(canonical).hexdigest()}"


class AiohttpJsonPoster:
    """Minimal optional HTTP transport shared by webhook adapters."""

    def __init__(self, *, timeout_seconds: float = 20.0):
        try:
            import aiohttp
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "aiohttp is required to run HTTP webhook adapters"
            ) from exc
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=timeout_seconds)
        )
        self._closed = False

    async def post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        params: Mapping[str, str] | None = None,
    ) -> Any:
        async with self._session.post(
            url, headers=dict(headers), json=dict(payload), params=params
        ) as response:
            if response.status < 200 or response.status >= 300:
                await response.read()
                raise RuntimeError(f"remote send API returned HTTP {response.status}")
            if response.content_type == "application/json":
                return await response.json()
            await response.read()
            return None

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._session.close()


class _AiohttpServer:
    def __init__(self, runner: Any):
        self._runner = runner
        self._closed = False

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._runner.cleanup()


async def start_aiohttp_server(
    *,
    handler: WebhookHandler,
    host: str,
    port: int,
    path: str,
    max_body_bytes: int,
) -> WebhookServer:
    try:
        from aiohttp import web
    except ModuleNotFoundError as exc:
        raise RuntimeError("aiohttp is required to run webhook adapters") from exc

    if not path.startswith("/"):
        raise ValueError("webhook path must start with /")

    async def route(request: Any) -> Any:
        try:
            raw_body = await request.read()
        except web.HTTPRequestEntityTooLarge:
            return web.Response(status=413)
        response = await handler(
            request.method,
            request.headers,
            request.query,
            raw_body,
        )
        return web.Response(
            status=response.status,
            body=response.body,
            headers={"Content-Type": response.content_type},
        )

    application = web.Application(client_max_size=max_body_bytes)
    application.router.add_route("*", path, route)
    runner = web.AppRunner(application, access_log=None)
    await runner.setup()
    try:
        site = web.TCPSite(runner, host=host, port=port)
        await site.start()
    except BaseException:
        await runner.cleanup()
        raise
    return _AiohttpServer(runner)


async def stop_webhook_components(
    *,
    server: WebhookServer | None,
    adapter: Any,
    runtime: Any,
    owns_runtime: bool,
) -> None:
    try:
        if server is not None:
            await server.close()
    finally:
        try:
            await adapter.close()
        finally:
            if owns_runtime:
                await runtime.close()
