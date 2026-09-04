from __future__ import annotations

import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.bot import discord_adapter as discord_module
from src.bot import slack_adapter as slack_module
from src.bot import teams_adapter as teams_module
from src.bot.discord_adapter import (
    DiscordAdapter,
    discord_conversation_key,
    discord_identity,
    discord_message_allows,
    start_discord_adapter,
    strip_discord_bot_mention,
)
from src.bot.slack_adapter import (
    SlackAdapter,
    slack_conversation_key,
    slack_event_allows,
    slack_identity,
    start_slack_adapter,
    strip_slack_bot_mention,
)
from src.bot.teams_adapter import (
    TeamsAdapter,
    start_teams_adapter,
    strip_teams_bot_mention,
    teams_bot_is_mentioned,
    teams_conversation_key,
    teams_identity,
    teams_message_allows,
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


class WesternPolicyTests(unittest.TestCase):
    def test_discord_identity_policy_mentions_and_conversation_key(self) -> None:
        user = SimpleNamespace(
            id=123456789,
            display_name="Teacher",
            global_name="Different",
            name="teacher",
        )
        identity = discord_identity(user)
        self.assertEqual("discord:123456789", identity.key)
        self.assertEqual("Teacher", identity.display_name)
        self.assertTrue(
            discord_message_allows(
                is_direct=True,
                mentioned=False,
                replied_to_bot=False,
                mode="direct",
            )
        )
        self.assertFalse(
            discord_message_allows(
                is_direct=False,
                mentioned=False,
                replied_to_bot=False,
                mode="mentions",
            )
        )
        self.assertTrue(
            discord_message_allows(
                is_direct=False,
                mentioned=False,
                replied_to_bot=True,
                mode="mentions",
            )
        )
        self.assertEqual(
            "hello", strip_discord_bot_mention("<@!999> hello", "999")
        )
        self.assertEqual(
            "discord:123456789/55", discord_conversation_key(identity, 55)
        )
        with self.assertRaises(ValueError):
            discord_message_allows(
                is_direct=True,
                mentioned=False,
                replied_to_bot=False,
                mode="invalid",
            )

    def test_slack_identity_policy_mentions_and_conversation_key(self) -> None:
        identity = slack_identity("T123", "U456", "Teacher")
        self.assertEqual("slack:T123:U456", identity.key)
        self.assertEqual("Teacher", identity.display_name)
        self.assertTrue(
            slack_event_allows(event_type="app_mention", channel_type="channel")
        )
        self.assertTrue(
            slack_event_allows(event_type="message", channel_type="im")
        )
        self.assertFalse(
            slack_event_allows(event_type="message", channel_type="channel")
        )
        self.assertEqual("hello", strip_slack_bot_mention("<@B1> hello", "B1"))
        self.assertEqual(
            "slack:T123:U456/C1:123.4",
            slack_conversation_key(identity, "C1", "123.4"),
        )
        with self.assertRaises(ValueError):
            slack_identity("", "U456")

    def test_teams_identity_policy_mentions_and_conversation_key(self) -> None:
        activity = _teams_activity(text="<at>Alona</at> hello", mentioned=True)
        identity = teams_identity(activity)
        self.assertEqual("teams:tenant-1:aad-teacher", identity.key)
        self.assertEqual("Teacher", identity.display_name)
        self.assertTrue(teams_bot_is_mentioned(activity))
        self.assertEqual("hello", strip_teams_bot_mention(activity["text"], activity))
        self.assertTrue(
            teams_message_allows(
                conversation_type="personal", mentioned=False, mode="direct"
            )
        )
        self.assertFalse(
            teams_message_allows(
                conversation_type="channel", mentioned=False, mode="mentions"
            )
        )
        self.assertTrue(
            teams_message_allows(
                conversation_type="channel", mentioned=True, mode="mentions"
            )
        )
        self.assertEqual(
            "teams:tenant-1:aad-teacher/conv-1:root-1",
            teams_conversation_key(identity, "conv-1", "root-1"),
        )
        with self.assertRaises(ValueError):
            teams_message_allows(
                conversation_type="personal", mentioned=False, mode="invalid"
            )

        activity["text"] = "<at>Alona</at> hello <at>Alice</at>"
        activity["entities"].append(
            {
                "type": "mention",
                "mentioned": {"id": "user-2", "name": "Alice"},
                "text": "<at>Alice</at>",
            }
        )
        self.assertEqual(
            "hello <at>Alice</at>",
            strip_teams_bot_mention(activity["text"], activity),
        )


class _DiscordChannel:
    def __init__(self, channel_id: str = "channel-1") -> None:
        self.id = channel_id
        self.sent: list[str] = []
        self.typing_enter_count = 0
        self.typing_exit_count = 0

    async def send(self, text: str, **_kwargs: object) -> None:
        self.sent.append(text)

    def typing(self) -> object:
        channel = self

        class TypingContext:
            async def __aenter__(self) -> object:
                channel.typing_enter_count += 1
                return self

            async def __aexit__(
                self, _exc_type: object, _exc: object, _traceback: object
            ) -> None:
                channel.typing_exit_count += 1

        return TypingContext()


def _discord_message(
    *,
    author_id: str = "teacher-1",
    content: str = "<@bot-1> hello",
    guild: bool = True,
    mentioned: bool = True,
    author_is_bot: bool = False,
) -> SimpleNamespace:
    channel = _DiscordChannel()
    return SimpleNamespace(
        author=SimpleNamespace(
            id=author_id,
            bot=author_is_bot,
            display_name="Teacher",
            name="teacher",
        ),
        channel=channel,
        guild=SimpleNamespace(id="guild-1") if guild else None,
        mentions=[SimpleNamespace(id="bot-1")] if mentioned else [],
        reference=None,
        content=content,
    )


class DiscordAdapterTests(unittest.IsolatedAsyncioTestCase):
    def _harness(self, *, allowed_users: set[str] | None = None) -> object:
        class Harness:
            user = SimpleNamespace(id="bot-1")
            guild_mode = "mentions"
            allowed_users: set[str] = set()
            allowed_channels: set[str] = set()
            allowed_guilds: set[str] = set()
            message_limit = 5
            dispatcher = _CapturingDispatcher()
            _is_reply_to_bot = DiscordAdapter._is_reply_to_bot

        harness = Harness()
        harness.allowed_users = allowed_users or set()
        return harness

    async def test_guild_mention_dispatches_with_key_splitting_and_typing(self) -> None:
        harness = self._harness()
        message = _discord_message()
        await DiscordAdapter.on_message(harness, message)  # type: ignore[arg-type]

        calls = harness.dispatcher.calls  # type: ignore[attr-defined]
        self.assertEqual(1, len(calls))
        self.assertEqual("discord:teacher-1", calls[0]["identity"].key)
        self.assertEqual("discord:teacher-1/channel-1", calls[0]["conversation_key"])
        self.assertEqual("hello", calls[0]["text"])

        await calls[0]["send_text"]("123456789")
        self.assertEqual(["12345", "6789"], message.channel.sent)
        await calls[0]["set_typing"](True)
        self.assertEqual(1, message.channel.typing_enter_count)
        await calls[0]["set_typing"](False)
        self.assertEqual(1, message.channel.typing_exit_count)

    async def test_self_echo_unmentioned_guild_and_whitelist_are_rejected(self) -> None:
        harness = self._harness(allowed_users={"teacher-1"})
        await DiscordAdapter.on_message(  # type: ignore[arg-type]
            harness, _discord_message(author_id="bot-1")
        )
        await DiscordAdapter.on_message(  # type: ignore[arg-type]
            harness, _discord_message(mentioned=False, content="hello")
        )
        await DiscordAdapter.on_message(  # type: ignore[arg-type]
            harness, _discord_message(author_id="stranger")
        )
        self.assertEqual([], harness.dispatcher.calls)  # type: ignore[attr-defined]

    async def test_start_closes_transport_after_client_returns(self) -> None:
        events: list[object] = []

        class FakeAdapter:
            def __init__(self, runtime: object) -> None:
                events.append(("init", runtime))

            async def start(self, token: str, *, reconnect: bool) -> None:
                events.append(("start", token, reconnect))

            async def close_adapter(self) -> None:
                events.append("close")

        runtime = SimpleNamespace()
        with (
            patch.dict(os.environ, {"DISCORD_BOT_TOKEN": "secret", "DISCORD_WHITELIST_ENABLED": "false"}),
            patch.object(discord_module, "_DISCORD_AVAILABLE", True),
            patch.object(discord_module, "DiscordAdapter", FakeAdapter),
        ):
            await start_discord_adapter(runtime)  # type: ignore[arg-type]
        self.assertEqual(
            [("init", runtime), ("start", "secret", True), "close"], events
        )

    async def test_start_error_still_closes_discord_client(self) -> None:
        events: list[str] = []

        class FakeAdapter:
            def __init__(self, _runtime: object) -> None:
                return None

            async def start(self, _token: str, *, reconnect: bool) -> None:
                self.assert_reconnect = reconnect
                events.append("start")
                raise ConnectionError("gateway failed")

            async def close_adapter(self) -> None:
                events.append("close")

        with (
            patch.dict(os.environ, {"DISCORD_BOT_TOKEN": "secret", "DISCORD_WHITELIST_ENABLED": "false"}),
            patch.object(discord_module, "_DISCORD_AVAILABLE", True),
            patch.object(discord_module, "DiscordAdapter", FakeAdapter),
        ):
            with self.assertRaisesRegex(ConnectionError, "gateway failed"):
                await start_discord_adapter(SimpleNamespace())  # type: ignore[arg-type]
        self.assertEqual(["start", "close"], events)


class _FakeSlackApp:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        self.handlers: dict[str, object] = {}
        self.client = SimpleNamespace()

    def event(self, name: str):
        def register(handler: object) -> object:
            self.handlers[name] = handler
            return handler

        return register


class _FakeSlackClient:
    def __init__(self) -> None:
        self.posts: list[dict] = []

    async def chat_postMessage(self, **kwargs: object) -> None:
        self.posts.append(kwargs)


class SlackAdapterTests(unittest.IsolatedAsyncioTestCase):
    def _adapter(
        self, *, allowed_users: str = "", message_limit: str = "5"
    ) -> SlackAdapter:
        with patch.dict(
            os.environ,
            {
                "SLACK_WHITELIST_ENABLED": "true" if allowed_users else "false",
                "SLACK_ALLOWED_USERS": allowed_users,
                "SLACK_ALLOWED_CHANNELS": "",
                "SLACK_ALLOWED_TEAMS": "",
                "SLACK_MESSAGE_LIMIT": message_limit,
                "SLACK_REPLY_IN_THREAD": "true",
            },
        ):
            adapter = SlackAdapter(
                _FakeSlackApp(), SimpleNamespace(), bot_user_id="B1"
            )  # type: ignore[arg-type]
        adapter.dispatcher = _CapturingDispatcher()  # type: ignore[assignment]
        return adapter

    async def test_app_mention_dispatches_deduplicates_and_splits_in_thread(self) -> None:
        adapter = self._adapter()
        client = _FakeSlackClient()
        event = {
            "user": "U1",
            "channel": "C1",
            "text": "<@B1> hello",
            "ts": "100.1",
        }
        body = {"team_id": "T1", "event_id": "Ev1"}
        await adapter.handle_event(
            event=event,
            body=body,
            client=client,
            event_type="app_mention",
        )
        await adapter.handle_event(
            event=event,
            body=body,
            client=client,
            event_type="app_mention",
        )

        self.assertEqual(1, len(adapter.dispatcher.calls))  # type: ignore[attr-defined]
        call = adapter.dispatcher.calls[0]  # type: ignore[attr-defined]
        self.assertEqual("slack:T1:U1", call["identity"].key)
        self.assertEqual("slack:T1:U1/C1:100.1", call["conversation_key"])
        self.assertEqual("hello", call["text"])
        await call["send_text"]("123456789")
        self.assertEqual(
            [
                {"channel": "C1", "text": "12345", "thread_ts": "100.1"},
                {"channel": "C1", "text": "6789", "thread_ts": "100.1"},
            ],
            client.posts,
        )

    async def test_dm_is_accepted_but_self_echo_and_whitelist_are_rejected(self) -> None:
        adapter = self._adapter(allowed_users="U1")
        client = _FakeSlackClient()
        await adapter.handle_event(
            event={
                "user": "U1",
                "channel": "D1",
                "channel_type": "im",
                "text": "hello",
                "ts": "1",
            },
            body={"team_id": "T1", "event_id": "Ev1"},
            client=client,
            event_type="message",
        )
        await adapter.handle_event(
            event={
                "user": "B1",
                "channel": "D1",
                "channel_type": "im",
                "text": "echo",
            },
            body={"team_id": "T1", "event_id": "Ev2"},
            client=client,
            event_type="message",
        )
        await adapter.handle_event(
            event={
                "user": "U2",
                "channel": "D1",
                "channel_type": "im",
                "text": "stranger",
            },
            body={"team_id": "T1", "event_id": "Ev3"},
            client=client,
            event_type="message",
        )
        self.assertEqual(1, len(adapter.dispatcher.calls))  # type: ignore[attr-defined]
        self.assertEqual(
            "slack:T1:U1/D1",
            adapter.dispatcher.calls[0]["conversation_key"],  # type: ignore[attr-defined]
        )

    async def test_enterprise_grid_id_is_used_when_team_id_is_absent(self) -> None:
        adapter = self._adapter()
        await adapter.handle_event(
            event={
                "user": "U1",
                "channel": "D1",
                "channel_type": "im",
                "text": "hello",
            },
            body={"enterprise_id": "E1", "event_id": "Ev1"},
            client=_FakeSlackClient(),
            event_type="message",
        )
        self.assertEqual(
            "slack:E1:U1",
            adapter.dispatcher.calls[0]["identity"].key,  # type: ignore[attr-defined]
        )

    async def test_start_and_close_use_official_socket_mode_lifecycle(self) -> None:
        events: list[object] = []

        class FakeApp(_FakeSlackApp):
            def __init__(self, *, token: str) -> None:
                super().__init__()
                events.append(("app", token))

        class FakeHandler:
            def __init__(self, app: object, token: str) -> None:
                events.append(("handler", app, token))

            async def start_async(self) -> None:
                events.append("start")

            async def close_async(self) -> None:
                events.append("close")

        runtime = SimpleNamespace()
        with (
            patch.dict(
                os.environ,
                {
                    "SLACK_BOT_TOKEN": "xoxb-test",
                    "SLACK_APP_TOKEN": "xapp-test",
                    "SLACK_BOT_USER_ID": "B1",
                    "SLACK_ALLOWED_USERS": "",
                    "SLACK_ALLOWED_CHANNELS": "",
                    "SLACK_ALLOWED_TEAMS": "",
                    "SLACK_WHITELIST_ENABLED": "false",
                },
            ),
            patch.object(slack_module, "_SLACK_AVAILABLE", True),
            patch.object(slack_module, "_AsyncApp", FakeApp),
            patch.object(slack_module, "_AsyncSocketModeHandler", FakeHandler),
        ):
            await start_slack_adapter(runtime)  # type: ignore[arg-type]
        self.assertEqual("start", events[-2])
        self.assertEqual("close", events[-1])

    async def test_socket_start_error_still_closes_handler(self) -> None:
        events: list[str] = []

        class FakeHandler:
            def __init__(self, _app: object, _token: str) -> None:
                return None

            async def start_async(self) -> None:
                events.append("start")
                raise ConnectionError("socket failed")

            async def close_async(self) -> None:
                events.append("close")

        with (
            patch.dict(
                os.environ,
                {
                    "SLACK_BOT_TOKEN": "xoxb-test",
                    "SLACK_APP_TOKEN": "xapp-test",
                    "SLACK_BOT_USER_ID": "U-BOT",
                    "SLACK_WHITELIST_ENABLED": "false",
                },
            ),
            patch.object(slack_module, "_SLACK_AVAILABLE", True),
            patch.object(slack_module, "_AsyncApp", _FakeSlackApp),
            patch.object(slack_module, "_AsyncSocketModeHandler", FakeHandler),
        ):
            with self.assertRaisesRegex(ConnectionError, "socket failed"):
                await start_slack_adapter(SimpleNamespace())  # type: ignore[arg-type]
        self.assertEqual(["start", "close"], events)


def _teams_activity(
    *,
    text: str = "hello",
    mentioned: bool = False,
    conversation_type: str = "channel",
    sender_id: str = "teams-user-1",
    aad_object_id: str = "aad-teacher",
    sender_type: str = "person",
) -> dict:
    entities = []
    if mentioned:
        entities.append(
            {
                "type": "mention",
                "mentioned": {"id": "bot-1", "name": "Alona"},
                "text": "<at>Alona</at>",
            }
        )
    return {
        "id": "root-1",
        "type": "message",
        "text": text,
        "from": {
            "id": sender_id,
            "aadObjectId": aad_object_id,
            "name": "Teacher",
            "type": sender_type,
        },
        "recipient": {"id": "bot-1", "name": "Alona"},
        "conversation": {
            "id": "conv-1",
            "conversationType": conversation_type,
        },
        "channelData": {"tenant": {"id": "tenant-1"}},
        "entities": entities,
    }


class _FakeTeamsApp:
    instances: list["_FakeTeamsApp"] = []

    def __init__(self, **options: object) -> None:
        self.options = options
        self.handler: object | None = None
        self.started_port: int | None = None
        self.stop_count = 0
        self.__class__.instances.append(self)

    def on_message(self, handler: object) -> object:
        self.handler = handler
        return handler

    async def start(self, *, port: int) -> None:
        self.started_port = port

    async def stop(self) -> None:
        self.stop_count += 1


class _FakeTeamsContext:
    def __init__(self, activity: object) -> None:
        self.activity = activity
        self.sent: list[str] = []

    async def send(self, text: str) -> None:
        self.sent.append(text)


class TeamsAdapterTests(unittest.IsolatedAsyncioTestCase):
    def _adapter(self, *, allowed_users: str = "") -> TeamsAdapter:
        app = _FakeTeamsApp()
        with patch.dict(
            os.environ,
            {
                "TEAMS_GROUP_MODE": "mentions",
                "TEAMS_WHITELIST_ENABLED": "true" if allowed_users else "false",
                "TEAMS_ALLOWED_USERS": allowed_users,
                "TEAMS_ALLOWED_TENANTS": "",
                "TEAMS_ALLOWED_CONVERSATIONS": "",
                "TEAMS_MESSAGE_LIMIT": "5",
            },
        ):
            adapter = TeamsAdapter(app, SimpleNamespace())  # type: ignore[arg-type]
        adapter.dispatcher = _CapturingDispatcher()  # type: ignore[assignment]
        return adapter

    async def test_channel_mention_dispatches_with_thread_key_and_splitting(self) -> None:
        adapter = self._adapter()
        activity = _teams_activity(text="<at>Alona</at> hello", mentioned=True)
        # JSON-shaped Activity payloads can explicitly carry a null replyToId;
        # the root Activity id must remain the channel thread scope.
        activity["replyToId"] = None
        ctx = _FakeTeamsContext(activity)
        await adapter._on_message(ctx)

        self.assertEqual(1, len(adapter.dispatcher.calls))  # type: ignore[attr-defined]
        call = adapter.dispatcher.calls[0]  # type: ignore[attr-defined]
        self.assertEqual("teams:tenant-1:aad-teacher", call["identity"].key)
        self.assertEqual(
            "teams:tenant-1:aad-teacher/conv-1:root-1",
            call["conversation_key"],
        )
        self.assertEqual("hello", call["text"])
        await call["send_text"]("123456789")
        self.assertEqual(["12345", "6789"], ctx.sent)

    async def test_personal_is_accepted_but_self_group_and_whitelist_are_rejected(self) -> None:
        adapter = self._adapter(allowed_users="aad-teacher")
        personal = _teams_activity(conversation_type="personal")
        await adapter._on_message(_FakeTeamsContext(personal))
        await adapter._on_message(
            _FakeTeamsContext(_teams_activity(sender_id="bot-1", mentioned=True))
        )
        await adapter._on_message(
            _FakeTeamsContext(
                _teams_activity(
                    sender_id="another-bot",
                    aad_object_id="",
                    sender_type="bot",
                    mentioned=True,
                )
            )
        )
        role_bot = _teams_activity(
            sender_id="role-bot", aad_object_id="", mentioned=True
        )
        role_bot["from"].pop("type")
        role_bot["from"]["userRole"] = "bot"
        await adapter._on_message(_FakeTeamsContext(role_bot))
        await adapter._on_message(
            _FakeTeamsContext(_teams_activity(mentioned=False))
        )
        await adapter._on_message(
            _FakeTeamsContext(
                _teams_activity(
                    sender_id="stranger",
                    aad_object_id="aad-stranger",
                    mentioned=True,
                )
            )
        )
        self.assertEqual(1, len(adapter.dispatcher.calls))  # type: ignore[attr-defined]
        self.assertEqual(
            "teams:tenant-1:aad-teacher/conv-1",
            adapter.dispatcher.calls[0]["conversation_key"],  # type: ignore[attr-defined]
        )

    async def test_start_passes_documented_options_and_stops_app(self) -> None:
        _FakeTeamsApp.instances.clear()
        runtime = SimpleNamespace()
        with (
            patch.dict(
                os.environ,
                {
                    "TEAMS_CLIENT_ID": "client-1",
                    "TEAMS_TENANT_ID": "tenant-1",
                    "TEAMS_CLIENT_SECRET": "secret-1",
                    "TEAMS_MANAGED_IDENTITY_CLIENT_ID": "",
                    "TEAMS_CLOUD": "",
                    "TEAMS_MESSAGING_ENDPOINT": "",
                    "TEAMS_SERVICE_URL": "",
                    "TEAMS_PORT": "4567",
                    "TEAMS_ALLOWED_USERS": "",
                    "TEAMS_ALLOWED_TENANTS": "",
                    "TEAMS_ALLOWED_CONVERSATIONS": "",
                    "TEAMS_WHITELIST_ENABLED": "false",
                },
            ),
            patch.object(teams_module, "_TEAMS_AVAILABLE", True),
            patch.object(teams_module, "_TeamsApp", _FakeTeamsApp),
        ):
            await start_teams_adapter(runtime)  # type: ignore[arg-type]

        app = _FakeTeamsApp.instances[-1]
        self.assertEqual(
            {
                "client_id": "client-1",
                "tenant_id": "tenant-1",
                "client_secret": "secret-1",
            },
            app.options,
        )
        self.assertEqual(4567, app.started_port)
        self.assertEqual(1, app.stop_count)
        self.assertIsNotNone(app.handler)

    async def test_cloud_name_is_resolved_to_official_sdk_environment(self) -> None:
        _FakeTeamsApp.instances.clear()
        runtime = SimpleNamespace()
        with (
            patch.dict(
                os.environ,
                {
                    "TEAMS_CLIENT_ID": "client-1",
                    "TEAMS_TENANT_ID": "tenant-1",
                    "TEAMS_CLIENT_SECRET": "secret-1",
                    "TEAMS_CLOUD": "USGov",
                    "TEAMS_PORT": "3978",
                    "TEAMS_ALLOWED_USERS": "",
                    "TEAMS_ALLOWED_TENANTS": "",
                    "TEAMS_ALLOWED_CONVERSATIONS": "",
                    "TEAMS_WHITELIST_ENABLED": "false",
                },
            ),
            patch.object(teams_module, "_TEAMS_AVAILABLE", True),
            patch.object(teams_module, "_TeamsApp", _FakeTeamsApp),
            patch.object(
                teams_module,
                "_teams_cloud_from_name",
                side_effect=lambda name: f"resolved:{name}",
            ) as resolver,
        ):
            await start_teams_adapter(runtime)  # type: ignore[arg-type]

        resolver.assert_called_once_with("USGov")
        self.assertEqual(
            "resolved:USGov", _FakeTeamsApp.instances[-1].options["cloud"]
        )

    async def test_real_sdk_activity_model_uses_supported_field_names(self) -> None:
        if not teams_module._TEAMS_AVAILABLE:
            self.skipTest("microsoft-teams-apps is not installed")
        from microsoft_teams.api.activities import MessageActivity

        payload = _teams_activity(text="<at>Alona</at> hello", mentioned=True)
        payload.update(
            {
                "serviceUrl": "https://smba.trafficmanager.net/amer/",
                "channelId": "msteams",
            }
        )
        activity = MessageActivity.model_validate(payload)
        self.assertEqual(
            "teams:tenant-1:aad-teacher", teams_identity(activity).key
        )
        self.assertTrue(teams_bot_is_mentioned(activity))
        self.assertEqual("hello", strip_teams_bot_mention(activity.text, activity))


class MissingDependencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_startup_errors_name_the_missing_extra(self) -> None:
        with patch.object(discord_module, "_DISCORD_AVAILABLE", False):
            with self.assertRaisesRegex(RuntimeError, "discord.py"):
                await start_discord_adapter(SimpleNamespace())  # type: ignore[arg-type]
        with patch.object(slack_module, "_SLACK_AVAILABLE", False):
            with self.assertRaisesRegex(RuntimeError, "slack-bolt"):
                await start_slack_adapter(SimpleNamespace())  # type: ignore[arg-type]
        with patch.object(teams_module, "_TEAMS_AVAILABLE", False):
            with self.assertRaisesRegex(RuntimeError, "microsoft-teams-apps"):
                await start_teams_adapter(SimpleNamespace())  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
