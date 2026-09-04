# 主流聊天平台 Adapter 评审与集成报告

> 后续状态（2026-08-31）：本报告列出的正式入口、共享 runtime、IdentityKey/ConversationKey、持久化去重、依赖分组、环境配置和 fail-closed P0 已完成。实现结果、测试和在线验收边界见 `ADAPTER_INTEGRATION_COMPLETION_REPORT_2026-08-31.md`。下文保留为改造前审计快照。

日期：2026-08-31  
评审对象：本轮新增的 Discord、Slack、Microsoft Teams、Lark/飞书、企业微信智能机器人、钉钉、QQ 官方机器人、LINE、WhatsApp、Messenger、Google Chat adapter，以及已有 Telegram、XMPP、Matrix adapter  
代码范围：`src/bot/*_adapter.py`、`src/bot/adapter_support.py`、`src/bot/webhook_support.py` 与 adapter 聚焦测试

## 1. 结论摘要

本轮已经为 11 个新增平台形成独立 adapter，并保留已有 Telegram、XMPP、Matrix，共计 14 个平台边界。新增 adapter 普遍具备：稳定平台身份、保守的私聊/群聊触发策略、白名单、长消息拆分、快速 webhook ACK 或官方长连接、可注入的 `AdapterRuntime`、关闭清理，以及适合 fake client 的协议测试边界。

但这不是“14 平台已经由正式入口同时上线”：

- **没有修改 `bot.py`**。当前 `python bot.py` 仍只识别 `ENABLE_TELEGRAM`，不会启动另外 13 个 adapter。
- **没有修改 `requirements.txt`**。本轮为开发和校验安装的精确依赖版本尚未进入可复现安装清单。
- **没有修改 `.env.example`**。新增平台的开关、凭据和安全默认值尚未形成正式部署契约。
- 11 个新增 adapter 与 XMPP/Matrix 可以接受外部共享 `AdapterRuntime`；Telegram 仍自行创建一整套 memory、coordinator、worker 和 semaphore。直接并行调用所有无参入口会产生多套 runtime 和同一数据目录的多 writer 风险。
- 多数平台只有**进程内去重**，重启、重投和崩溃恢复期间仍可能重复回复、重复写 journal 或重复触发长期记忆。
- adapter 的防抖会话键通常包含 chat/room/thread，但 `ConversationCoordinator` 的工作会话仍主要按 `identity.key` 使用；同一用户跨两个群或频道时，短期上下文仍可能串接。
- LINE、WhatsApp、Messenger、Google Chat、Teams 都需要可信公网 HTTPS 入口。当前本地监听器本身不终止公网 TLS，生产必须放在受控反向代理、负载均衡或云运行环境之后。

建议首批生产顺序：Telegram 基线 → Discord/Slack/Lark/企业微信/钉钉的显式 allowlist canary → LINE/Google Chat → Teams/Meta/QQ；XMPP/Matrix 作为明确受限的开放协议接入。QQ、Meta 和 Teams 的主要难点不是代码，而是平台审核、企业身份、权限和生产运维。

## 2. 评估口径与总览

“主流度”是目标用户覆盖和组织采用的定性判断，不代表精确月活排名；“接入难度”同时计入代码、平台控制台、审核、凭据、网络入口和生产运维，而不只是写一个 HTTP 请求的难度。

| 平台 | 主流度 / 典型市场 | 接入难度 | 入站传输 | 当前推荐状态 |
|---|---|---:|---|---|
| Telegram | 全球大众与社区，高 | 低 | Bot API long polling | 已有基线；需共享 runtime 与群策略改造 |
| Discord | 全球社区/游戏/开发者，高 | 低 | Gateway WSS | 适合首批 canary |
| Slack | 全球企业协作，高 | 低—中 | Socket Mode WSS | 适合首批 canary |
| Microsoft Teams | 全球大型组织，极高 | 中—高 | 认证 HTTPS Activity | 代码已具备；先完成 Entra/Teams 管理配置 |
| Lark/飞书 | 中国及国际企业协作，高 | 低—中 | 官方 Channel SDK WSS | 适合首批 canary |
| 企业微信智能机器人 | 中国企业与微信生态，高 | 低—中 | 官方智能机器人 WSS | 适合受控企业 canary |
| 钉钉 | 中国企业协作，高 | 中 | Stream Mode WSS | 可 canary；重点监控 SDK 生命周期 |
| QQ 官方机器人 | 中国大众/群社区，高 | 中—高 | QQ Gateway WSS + REST | 先 sandbox，审核后生产 |
| LINE | 日本、台湾、泰国等市场，高 | 中 | HTTPS webhook | 具备生产骨架；需公网 TLS 与额度评估 |
| WhatsApp | 全球大众，极高 | 中—高 | Meta HTTPS webhook | 技术可用；业务验证和消息窗口是门槛 |
| Messenger | 全球消费者与 Page 生态，高 | 中—高 | Meta HTTPS webhook | 技术可用；Page 权限和 App Review 是门槛 |
| Google Chat | Google Workspace 组织，高 | 中—高 | OIDC HTTPS interaction | 适合 Workspace 内部部署 |
| XMPP | 开放协议、联邦/自建场景，中低 | 中 | XMPP TLS 长连接 stanza | 仅单聊文本；受限部署 |
| Matrix | 开放协议、技术/组织自建，中 | 中；E2EE 为高 | Client-Server `/sync` | 明文房间可 canary；不能宣称完整 E2EE |

## 3. 经过校验的精确依赖版本

下列版本是 2026-08-31 在当前项目 `.venv` 中实际安装、inspect 并用于聚焦测试的版本。它们尚未写入 `requirements.txt`；生产应使用 lock/constraints 和哈希，而不是依赖开发机状态。

| 平台 | 直接依赖（精确版本） | Python 下限 / 说明 |
|---|---|---|
| Telegram | `python-telegram-bot[job-queue]==22.8` | 包要求 Python ≥3.10 |
| Discord | `discord.py==2.7.1` | Python ≥3.8 |
| Slack | `slack-bolt==1.30.0`、`slack-sdk==3.44.0`、`aiohttp==3.14.3` | Bolt ≥3.7；项目统一建议 ≥3.11 |
| Microsoft Teams | `microsoft-teams-apps==2.0.16`（连带 `microsoft-teams-api==2.0.16`、`microsoft-teams-common==2.0.16`） | Python ≥3.11 |
| Lark/飞书 | `lark-channel-sdk==1.3.0` | Python ≥3.8 |
| 企业微信智能机器人 | `wecom-aibot-python-sdk==1.0.2` | Python ≥3.8 |
| 钉钉 | `dingtalk-stream==0.24.3`、`aiohttp==3.14.3` | 官方包未在 wheel metadata 声明下限；本项目以 ≥3.11 验证 |
| QQ | `qqbot-agent-sdk==1.2.2`、`httpx==0.28.1` | Python ≥3.10 |
| LINE | `line-bot-sdk==3.25.0`、`aiohttp==3.14.3` | Python ≥3.10 |
| WhatsApp | `aiohttp==3.14.3`，直接调用官方 Graph API，无额外 Meta Python SDK | Python ≥3.10 |
| Messenger | `aiohttp==3.14.3`，直接调用官方 Graph API，无额外 Meta Python SDK | Python ≥3.10 |
| Google Chat | `google-apps-chat==0.10.5`、`google-auth==2.57.0`、`aiohttp==3.14.3` | Python ≥3.10 |
| XMPP | `slixmpp==1.17.0` | 当前版本要求 Python ≥3.11 |
| Matrix | `matrix-nio==0.26.0` | Python ≥3.10；完整 E2EE 还需经验证的 e2e extra/底层库 |

所有 adapter 还依赖项目公共的 `python-dotenv==1.2.3`。综合 Teams 和 Slixmpp 的要求，建议 all-adapters 环境统一锁定 Python 3.11+；当前开发环境为 Python 3.12。

## 4. 分平台评审

### 4.1 Telegram

- **事件和发送**：现有实现使用 Bot API long polling；通过 `python-telegram-bot` 发送文本、typing、下载图片和静态 sticker。
- **稳定身份**：`telegram:<numeric user_id>`。显示名仅展示，不参与私人记忆主键。
- **群聊策略**：当前 adapter 没有本地 `direct/mentions/all` 策略或 allowlist；它依赖 BotFather Privacy Mode 决定群内能收到哪些更新。一旦关闭 Privacy Mode，普通群消息也可能进入模型，这是当前风险。
- **必需环境变量**：`TELEGRAM_BOT_TOKEN`。`bot.py` 还读取 `ENABLE_TELEGRAM=true`。
- **控制台要求**：通过 BotFather 创建 bot、取得 token；按需要配置 Privacy Mode、群管理权限和命令。一般没有应用商店审核。
- **当前能力**：文本、图片、静态 sticker、命令、typing、防抖、长消息、视觉描述、完整现有 memory/coordinator 生命周期。
- **残余风险**：Telegram 自己拥有 runtime，尚不能由统一入口注入；没有平台 allowlist、群 mention guard、持久 `update_id` 去重和跨群短期会话隔离。正式多平台化时应先改造它，而不是复制其自建 runtime。
- **官方参考**：[Telegram Bot API](https://core.telegram.org/bots/api)。

### 4.2 Discord

- **事件和发送**：`discord.py` 通过 Discord Gateway 安全 WebSocket 接收消息，通过 Discord HTTP API 发送。
- **稳定身份**：Discord 全局 snowflake user ID，即 `discord:<user_id>`。
- **群聊策略**：DM 总是允许；guild 默认 `mentions`，只接受结构化 @bot 或回复 bot 的消息；可设 `direct` 或 `all`。`all` 才主动打开 Message Content intent。支持用户、频道、guild allowlist，发送时禁用意外 mentions。
- **必需环境变量**：`DISCORD_BOT_TOKEN`。
- **可选环境变量**：`DISCORD_GUILD_MODE=mentions|direct|all`、`DISCORD_ALLOWED_USERS`、`DISCORD_ALLOWED_CHANNELS`、`DISCORD_ALLOWED_GUILDS`、`DISCORD_MESSAGE_LIMIT`。
- **控制台/审核**：Discord Developer Portal 创建 Application/Bot、安装到 guild 并授予 View/Send 等权限。达到验证规模后需要 bot verification；`MESSAGE_CONTENT` 是 privileged intent，符合门槛时需要审批。当前 mentions 模式利用 DM/@mention 例外，可降低权限面。
- **当前能力**：DM 和 guild 文本、@mention/回复触发、typing、拆分、allowlist、自身/bot 回环过滤、官方 reconnect 生命周期。
- **残余风险**：没有附件、线程富语义、interaction/slash command、持久事件去重或限流预算；大规模部署还需 sharding 与 Gateway session start 管理。
- **官方参考**：[Discord Gateway 与 intents](https://docs.discord.com/developers/events/gateway)、[Bots & Companion Apps](https://docs.discord.com/developers/platform/bots)。

### 4.3 Slack

- **事件和发送**：官方 Bolt Python 的 Socket Mode WebSocket；回复使用 Slack Web API `chat.postMessage`。
- **稳定身份**：Slack user ID 只在 workspace 内解释，当前键为 `slack:<team_id>:<user_id>`。
- **群聊策略**：只接受 DM `message` 和显式 `app_mention`；忽略 subtype、hidden、bot/self 事件。频道消息默认线程内回复，conversation key 包含 channel/thread/user。
- **必需环境变量**：`SLACK_BOT_TOKEN`（`xoxb-...`）、`SLACK_APP_TOKEN`（启用 Socket Mode 的 `xapp-...`）。
- **可选环境变量**：`SLACK_BOT_USER_ID`、`SLACK_ALLOWED_USERS`、`SLACK_ALLOWED_CHANNELS`、`SLACK_ALLOWED_TEAMS`、`SLACK_REPLY_IN_THREAD`、`SLACK_MESSAGE_LIMIT`、`SLACK_EVENT_CACHE_SIZE`。
- **控制台/审核**：Slack App 控制台启用 Socket Mode，app token 至少需要 `connections:write`；订阅 `app_mention`、`message.im`，bot scopes 至少覆盖相应 history/mentions 和 `chat:write`。内部安装由 workspace 管理员批准；上架 App Directory 另需审核。
- **当前能力**：DM、频道 @mention、线程回复、allowlist、内存 event ID 去重、消息拆分、官方异步启停。
- **残余风险**：没有文件、Block Kit、reaction、持久 dedup；Socket Mode 的 app-level token 与 bot token 必须独立轮换。跨多个 Enterprise Grid workspace 时必须保留 team scope。
- **官方参考**：[Slack Bolt Python Socket Mode](https://docs.slack.dev/tools/bolt-python/concepts/socket-mode/)、[Bolt for Python](https://docs.slack.dev/tools/bolt-python/)。

### 4.4 Microsoft Teams

- **事件和发送**：官方 `microsoft-teams-apps` 提供认证 `/api/messages` HTTPS Activity server，SDK负责 JWT 验证；`ActivityContext.send()` 发送。
- **稳定身份**：优先 `tenant_id:aadObjectId`，退化到 `tenant_id:bot-scoped sender id`。不能用 display name 或 UPN 猜测身份。
- **群聊策略**：personal/direct 总是允许；频道/群默认必须有结构化 mention；支持 `direct/mentions/all`、tenant/user/conversation allowlist。频道 conversation key 包含 activity reply/thread ID。
- **必需环境变量**：`TEAMS_CLIENT_ID`、`TEAMS_TENANT_ID`；生产还应配置 `TEAMS_CLIENT_SECRET` 或 `TEAMS_MANAGED_IDENTITY_CLIENT_ID`。代码也接受无前缀兼容别名 `CLIENT_ID`、`TENANT_ID`、`CLIENT_SECRET`、`MANAGED_IDENTITY_CLIENT_ID`。
- **可选环境变量**：`TEAMS_PORT`/`PORT`、`TEAMS_CLOUD`、`TEAMS_MESSAGING_ENDPOINT`、`TEAMS_SERVICE_URL`、`TEAMS_GROUP_MODE`、`TEAMS_ALLOWED_USERS`、`TEAMS_ALLOWED_TENANTS`、`TEAMS_ALLOWED_CONVERSATIONS`、`TEAMS_MESSAGE_LIMIT`。
- **控制台/审核**：Entra 应用注册、凭据或 managed identity、Teams Developer Portal app/manifest、bot messaging endpoint 和组织安装策略；跨租户/商店分发可能需要管理员同意、Publisher Verification 和商店审核。公网 endpoint 必须 HTTPS。
- **当前能力**：personal 与频道/群文本、结构化 mention、租户作用域身份和 allowlist、频道线程 key、官方 SDK 鉴权与启停。
- **残余风险**：没有附件/card/SSO、无持久 dedup；当前 `ctx.send()` 对复杂 thread/reply 语义依赖 Teams transport。必须对目标 tenant 做真实 Activity、token refresh 与 sovereign cloud smoke test。
- **官方参考**：[Teams Python SDK `HttpServer`](https://learn.microsoft.com/en-us/python/api/microsoft-teams-apps/microsoft_teams.apps.httpserver?view=msteams-sdk-python-latest)、[Teams 应用平台](https://learn.microsoft.com/en-us/microsoftteams/platform/overview)。

### 4.5 Lark / 飞书

- **事件和发送**：官方 `lark-channel-sdk` 的 `FeishuChannel`，默认长连接；`channel.on("message")` 接收，`channel.send()` 发送和 reply。
- **稳定身份**：当前 `lark:<open_id>`；`open_id` 是应用作用域稳定 ID。若未来同一 runtime 托管多个 Lark app，应再显式加入 app ID scope。
- **群聊策略**：p2p 允许；group/topic 默认要求 SDK 提供的 `mentioned_bot`；支持 `mentions/all/disabled`、user/chat allowlist。SDK `PolicyConfig` 和 adapter 本地策略双层拦截。
- **必需环境变量**：`LARK_APP_ID`、`LARK_APP_SECRET`。
- **可选环境变量**：`LARK_DOMAIN`（Lark/飞书域选择）、`LARK_GROUP_MODE`、`LARK_ALLOWED_USERS`、`LARK_ALLOWED_CHATS`、`LARK_MESSAGE_LIMIT`、`LARK_DEDUP_SIZE`。
- **控制台/审核**：飞书/Lark 开放平台创建企业自建应用并启用 bot；申请接收消息和以 bot 身份发送权限，配置消息事件，发布版本并由企业管理员安装/批准。公网 webhook 不是本实现必需条件。
- **当前能力**：文本、SDK 安全化/mention 后正文、reply reference、allowlist、内存 message ID 去重、官方长连接启停。
- **残余风险**：当前不下载附件/图片、不支持 card action；app/tenant 多实例身份需进一步 namespace；一个进程持有连接时仍需健康检查与 token rotation 测试。
- **官方参考**：[Lark Channel SDK Python](https://github.com/larksuite/channel-sdk-python)、[飞书消息事件](https://open.feishu.cn/document/server-docs/im-v1/message/events/receive)。

### 4.6 企业微信智能机器人

- **事件和发送**：官方智能机器人 Python SDK，通过 `wss://openws.work.weixin.qq.com` 长连接接收；使用 `send_message()` 主动发 Markdown，使用官方 `download_file()` 下载并按 `aeskey` 解密媒体。
- **稳定身份**：当前 `wecom:<from.userid>`，userid 在企业内稳定。单 bot/单企业部署安全；若一个 runtime 托管多个企业机器人，必须增加 corp/bot scope，不能假设不同企业的 userid 不碰撞。
- **群聊策略**：单聊允许；群聊默认 `mentions`。较旧 payload 不总是给独立 mention flag，当前依据“智能机器人平台只把指向机器人的群消息路由到 bot”的官方交付语义接受；若平台以后扩展为全量群消息，必须重新校验。
- **必需环境变量**：`WECOM_BOT_ID`、`WECOM_SECRET`。
- **可选环境变量**：`WECOM_WS_URL`、`WECOM_GROUP_MODE`、`WECOM_ALLOWED_USERS`、`WECOM_ALLOWED_CHATS`、`WECOM_MESSAGE_LIMIT`、`WECOM_MAX_IMAGE_BYTES`、`WECOM_DEDUP_SIZE`。
- **控制台/审核**：企业微信管理后台创建“智能机器人”并取得 bot ID/secret，配置可见范围和可用会话；通常由企业管理员管理。不要把本实现和“群机器人自定义 webhook”混淆，后者主要是发送入口而不是完整收发 bot。
- **当前能力**：text、mixed text+image、官方 AES 下载/解密、Markdown 主动发送、allowlist、内存 msgid 去重、异步媒体任务和关闭。
- **残余风险**：未处理 voice/file/template card；同一 bot 通常只应保留一个活跃长连接；SDK/平台能力较新，需做企业版本、并发连接、主动发送频控和断网 soak。
- **官方参考**：[WeComTeam 官方智能机器人 Python SDK](https://github.com/WecomTeam/wecom-aibot-python-sdk)。

### 4.7 钉钉

- **事件和发送**：官方 `dingtalk-stream` 通过 Stream Mode WebSocket；`ChatbotHandler.process()` 只解析、入队并立即返回 `200/OK` ACK。LLM 完成后使用事件携带的 `sessionWebhook` 通过异步 `aiohttp` 回复。
- **稳定身份**：优先 `dingtalk:<senderCorpId>:<senderStaffId>`，缺字段时退化到官方 opaque `senderId`。
- **群聊策略**：conversation type `1` 视为单聊；群默认要求官方 `isInAtList`；支持 `mentions/all/disabled`、user/conversation allowlist。
- **必需环境变量**：`DINGTALK_CLIENT_ID`、`DINGTALK_CLIENT_SECRET`。
- **可选环境变量**：`DINGTALK_GROUP_MODE`、`DINGTALK_ALLOWED_USERS`、`DINGTALK_ALLOWED_CONVERSATIONS`、`DINGTALK_MESSAGE_LIMIT`、`DINGTALK_DEDUP_SIZE`。
- **控制台/审核**：钉钉开放平台创建应用/机器人、启用 Stream Mode、授予消息权限并发布到组织；企业管理员可见范围和应用发布规则需要单独处理。
- **当前能力**：文本、群 @、快速 ACK、异步 session webhook 发送、@原发送人、allowlist、内存 msgId 去重。
- **残余风险**：`sessionWebhook` 有过期时间，超时会明确失败；当前不走需要额外 access token 的主动消息兜底。官方 `dingtalk-stream==0.24.3` 没有 `stop()`，且其 `start()` 会吞第一次 `CancelledError`，adapter 已关闭底层 WebSocket并双取消任务，但仍应使用进程级 supervisor 验证退出。SDK 获取连接信息的部分路径也需要关注同步阻塞。
- **官方参考**：[DingTalk Stream Python SDK](https://github.com/open-dingtalk/dingtalk-stream-sdk-python)。

### 4.8 QQ 官方机器人

- **事件和发送**：官方 `qqbot-agent-sdk` 的 Gateway WebSocket、`EventParser` 和 REST `QQApiClient`；WS 在 SDK 线程中运行，事件投递到主 asyncio loop。
- **稳定身份**：QQ 的 `user_openid`、`member_openid` 等是场景作用域，当前键为 `qq:<chat_scope>:<openid>`；不会假设 c2c、group、guild 的 opaque ID 可直接合并。
- **群聊策略**：C2C 允许；group 只接受 `GROUP_AT_MESSAGE_CREATE`；guild 的 mentions 模式只接受 `_AT_MESSAGE_CREATE`。支持 `mentions/all/disabled`、user/chat/scope allowlist。
- **必需环境变量**：`QQ_APP_ID`、`QQ_CLIENT_SECRET`。
- **可选环境变量**：`QQ_GROUP_MODE`、`QQ_ALLOWED_USERS`、`QQ_ALLOWED_CHATS`、`QQ_ALLOWED_SCOPES`（默认 `c2c,group,guild`）、`QQ_MESSAGE_LIMIT`、`QQ_DEDUP_SIZE`。
- **控制台/审核**：QQ 开放平台创建机器人，先在 sandbox/测试群验证；生产能力取决于机器人审核、事件 intents、消息权限、IP 白名单和平台配额。
- **当前能力**：C2C/group/guild 文本、群 @、C2C typing、reply_to、allowlist、内存 message ID 去重、完整实际 `WSCallbacks` 状态契约。
- **残余风险**：官方 SDK 1.2.2 的 `send_text()` 只实现 c2c/group/guild，当前默认不开放 `dm` scope；即使手工加入也可能无法发回。没有附件、富媒体和 interaction。该 wheel 的 README 最小示例遗漏实际必需的部分 `WSCallbacks` 参数和 HTTP client `setup()`，实现以安装包 inspect 后的真实接口为准，升级必须回归。
- **官方参考**：[Tencent Connect QQ Bot Agent SDK](https://github.com/tencent-connect/qqbot-agent-sdk)。

### 4.9 LINE

- **事件和发送**：LINE Messaging API HTTPS webhook；官方 SDK parser 和异步 Messaging API。对原始 body 做 `base64(HMAC-SHA256(channel_secret, body))` 校验后才解析，立即返回 2xx，LLM 在后台执行。
- **稳定身份**：当前 `line:<userId>`。LINE userId 在 provider scope 下稳定；若同一 runtime 服务多个 provider，需加入 provider scope。群/room 以 `groupId`/`roomId` 作为会话目标。
- **群聊策略**：一对一允许；group/room 默认只接受 `mention.mentionees[].isSelf=true` 的结构化 bot mention；支持 `mentions/all/disabled` 与 user/chat allowlist。
- **必需环境变量**：`LINE_CHANNEL_SECRET`、`LINE_CHANNEL_ACCESS_TOKEN`。
- **可选环境变量**：`LINE_WEBHOOK_HOST`（默认仅 `127.0.0.1`）、`LINE_WEBHOOK_PORT`、`LINE_WEBHOOK_PATH`、`LINE_GROUP_MODE`、`LINE_ALLOWED_USERS`、`LINE_ALLOWED_CHATS`、`LINE_MESSAGE_LIMIT`、`LINE_MAX_WEBHOOK_BYTES`、`LINE_MAX_IMAGE_BYTES`、`LINE_DEDUP_SIZE`。
- **控制台/审核**：LINE Developers Console 创建 Messaging API channel/Official Account，签发 channel access token、启用并验证 webhook；若要群聊需打开 “Allow bot to join group chats”。生产 URL 必须为受信任 HTTPS。
- **当前能力**：text、入站 image 下载、原始 body 签名、快速 ACK、webhookEventId 内存去重、结构化群 mention、reply token 首次回复；token 过期或第二次发送时 fallback 到 push。
- **残余风险**：push 会消耗消息额度且要求目标仍可接收；reply token 短时且单次使用。图片当前按 LINE image content 作为 JPEG 交给视觉层，没有从响应头做更细 MIME 鉴定。持久 dedup 必须使用 `webhookEventId`，因为 LINE 可重投且顺序可能改变。
- **官方参考**：[接收 LINE webhook](https://developers.line.biz/en/docs/messaging-api/receiving-messages/)、[验证签名](https://developers.line.biz/en/docs/messaging-api/verify-webhook-signature/)、[群聊](https://developers.line.biz/en/docs/messaging-api/group-chats)。

### 4.10 WhatsApp Cloud API

- **事件和发送**：Meta HTTPS webhook + Graph API。GET 处理 subscription challenge；POST 在 JSON 解析前校验原始 body 的 `X-Hub-Signature-256`，事件转后台后快速 ACK。
- **稳定身份**：当前 `whatsapp:<receiving phone_number_id>:<wa_id>`，避免同一号码身份跨业务接收端被错误合并。
- **群聊策略**：当前硬性只支持 direct；检测到 group ID 即拒绝，`WHATSAPP_CHAT_MODE` 只允许 `direct`。
- **必需环境变量**：`WHATSAPP_APP_SECRET`（可回退 `META_APP_SECRET`）、`WHATSAPP_VERIFY_TOKEN`、`WHATSAPP_ACCESS_TOKEN`、`WHATSAPP_GRAPH_API_VERSION`（可回退 `META_GRAPH_API_VERSION`）、`WHATSAPP_PHONE_NUMBER_ID`。
- **可选环境变量**：`WHATSAPP_ALLOWED_USERS`、`WHATSAPP_ALLOWED_PHONE_NUMBER_IDS`、`WHATSAPP_WEBHOOK_HOST/PORT/PATH`、`WHATSAPP_WEBHOOK_MAX_BYTES`、`WHATSAPP_MESSAGE_LIMIT`、`WHATSAPP_CHAT_MODE=direct`。
- **控制台/审核**：Meta App、Business Portfolio、WhatsApp Business Account 和电话号码；配置 Webhooks/messages 订阅、system user 或可轮换 token、App Review/Business Verification 和生产权限。客户服务窗口之外的业务主动消息通常必须使用已批准 template，并受国家/类别/质量/费用规则影响。
- **当前能力**：一对一 text 收发、订阅验证、签名、phone/user allowlist、快速 ACK、内存 event dedup、Graph API 版本校验和消息拆分。
- **残余风险**：未实现 media、template、status callback、质量/额度状态或 24 小时窗口策略；当前普通 text 发送若越过平台允许窗口会被 Graph API 拒绝。访问 token、App Secret 和电话号码 scope 必须轮换并脱敏。
- **官方参考**：[WhatsApp Cloud API Webhooks](https://developers.facebook.com/docs/whatsapp/cloud-api/webhooks)、[Cloud API 发送消息](https://developers.facebook.com/docs/whatsapp/cloud-api/guides/send-messages)。

### 4.11 Messenger

- **事件和发送**：Facebook Page Messenger HTTPS webhook + Graph Send API；GET subscription challenge、POST 原始 body `X-Hub-Signature-256`，后台派发后快速 ACK。
- **稳定身份**：PSID 只在 Page scope 中稳定，当前为 `messenger:<page_id>:<PSID>`。
- **群聊策略**：当前只支持 Page 与用户 direct，`MESSENGER_CHAT_MODE` 只允许 `direct`。
- **必需环境变量**：`MESSENGER_APP_SECRET`（可回退 `META_APP_SECRET`）、`MESSENGER_VERIFY_TOKEN`、`MESSENGER_PAGE_ACCESS_TOKEN`、`MESSENGER_GRAPH_API_VERSION`（可回退 `META_GRAPH_API_VERSION`）、`MESSENGER_PAGE_ID`。
- **可选环境变量**：`MESSENGER_ALLOWED_USERS`、`MESSENGER_WEBHOOK_HOST/PORT/PATH`、`MESSENGER_WEBHOOK_MAX_BYTES`、`MESSENGER_MESSAGE_LIMIT`、`MESSENGER_CHAT_MODE=direct`。
- **控制台/审核**：Meta App 绑定 Facebook Page，配置 Webhooks Page `messages` 订阅和 Page access token；内部开发角色可测试，公开生产需要 `pages_messaging` 等权限、App Review、Live mode 及 Page/Business 管理批准。
- **当前能力**：direct text、typing_on/off、订阅验证、签名、PSID allowlist、快速 ACK、内存 dedup、拆分。
- **残余风险**：没有附件、postback、persistent menu、handover；受 Messenger 消息政策和窗口约束。PSID 不可跨 Page 合并；Page token 泄漏影响面很大。
- **官方参考**：[Messenger Platform Webhooks](https://developers.facebook.com/docs/messenger-platform/webhooks/)、[Messenger Send API](https://developers.facebook.com/docs/messenger-platform/send-messages/)。

### 4.12 Google Chat

- **事件和发送**：Google Chat HTTPS interaction endpoint；先验证 `Authorization: Bearer` OIDC token 的 audience、issuer/email，再解析 JSON。立即回空 JSON，使用官方 Chat async client 和 service account 异步 `spaces.messages.create`。
- **稳定身份**：`google_chat:<app_scope>:<domain_id or consumer>:<users/{id}>`。displayName/email 不作为主键。
- **群聊策略**：DM 允许；multi-person space 只在平台已调用 app（通常 @mention 或 command）时产生 interaction，当前 `mentions` 模式依赖这一交付语义；支持 app/user/space/domain allowlist，thread 保留。
- **必需环境变量**：`GOOGLE_CHAT_OIDC_AUDIENCE`、`GOOGLE_CHAT_SERVICE_ACCOUNT_FILE`（可回退 `GOOGLE_APPLICATION_CREDENTIALS`）。
- **可选环境变量**：`GOOGLE_CHAT_APP_SCOPE`、`GOOGLE_CHAT_ALLOWED_USERS`、`GOOGLE_CHAT_ALLOWED_SPACES`、`GOOGLE_CHAT_ALLOWED_DOMAINS`、`GOOGLE_CHAT_SPACE_MODE=direct|mentions`、`GOOGLE_CHAT_WEBHOOK_HOST/PORT/PATH`、`GOOGLE_CHAT_WEBHOOK_MAX_BYTES`、`GOOGLE_CHAT_MESSAGE_LIMIT`。
- **控制台/审核**：Google Cloud project、启用 Google Chat API、Chat API Configuration 中设置 HTTPS endpoint、audience、互动功能和可见范围；service account 需要 `chat.bot` app auth。组织内可由管理员控制安装；公开 Marketplace 分发和管理员预安装另有审核/同意流程。
- **当前能力**：text、DM/@invocation、OIDC verification、快速 ACK、异步 Chat API 回复、thread reply、user/space/domain allowlist、内存 dedup。
- **残余风险**：没有 cards/dialogs/files；service-account JSON 是高敏感长效凭据，应优先 workload identity。Google 可能重试 interaction，持久 event ID 去重不可省略；异步发送权限和同步 30 秒回复模型不同，当前选择异步是为了容纳 LLM 延迟。
- **官方参考**：[接收 interaction events](https://developers.google.com/workspace/chat/receive-respond-interactions)、[验证 Google Chat 请求](https://developers.google.com/workspace/chat/verify-requests-from-chat)、[Chat app 认证](https://developers.google.com/workspace/chat/authenticate-authorize)。

### 4.13 XMPP

- **事件和发送**：Slixmpp 通过 XMPP TLS 长连接收发 `chat`/`normal` stanza，启用 service discovery 与 ping。
- **稳定身份**：`xmpp:<bare JID lowercased>`；resource 只代表设备。防抖 reply target 仍保留完整 JID/resource，避免回复到错误客户端。
- **群聊策略**：当前不处理 `groupchat`，即没有 XEP-0045 MUC；只支持一对一，支持 bare JID allowlist。
- **必需环境变量**：`XMPP_JID`、`XMPP_PASSWORD`。
- **可选环境变量**：`XMPP_HOST` 与 `XMPP_PORT` 必须成对配置；`XMPP_ALLOWED_JIDS`、`XMPP_MESSAGE_LIMIT`。
- **控制台/审核**：没有中央商店审核；需要在目标 XMPP server 注册/分配账号，正确 DNS/SRV 或 host、证书、TLS、服务器策略和 roster。自建 server 的运维成本由部署方承担。
- **当前能力**：单聊 text、bare identity、allowlist、拆分、防抖、认证失败/断线边界和关闭。
- **残余风险**：没有 stanza ID 持久 dedup、MUC、媒体、引用、OMEMO；不同服务器的扩展和历史重放语义不同。匿名 MUC 昵称不是稳定私人身份，未来不能直接拿 nick 写私人记忆。
- **官方参考**：[XMPP RFC/XEP 索引](https://xmpp.org/extensions/)、[XEP-0045 MUC](https://xmpp.org/extensions/xep-0045.html)。更完整评审见同目录 [XMPP/Matrix 集成报告](./XMPP_MATRIX_ADAPTER_INTEGRATION_REPORT_2026-08-31.md)。

### 4.14 Matrix

- **事件和发送**：matrix-nio Client-Server `/sync` long polling；REST room send、typing 和 media download。
- **稳定身份**：`matrix:<fully qualified MXID>`，例如 `@user:example.org`。
- **群聊策略**：通过成员数近似 DM；支持 `direct/mentions/all`，结构化 `m.mentions.user_ids` 优先，另有 room/user allowlist、可选 invite auto-join。默认过滤 adapter 启动前的旧事件。
- **必需环境变量**：`MATRIX_HOMESERVER`、`MATRIX_USER_ID`，以及推荐的 `MATRIX_ACCESS_TOKEN`；无 token 时使用 `MATRIX_PASSWORD` 登录。
- **可选环境变量**：`MATRIX_DEVICE_ID`、`MATRIX_DEVICE_NAME`、`MATRIX_ROOM_MODE`、`MATRIX_ALLOWED_ROOMS`、`MATRIX_ALLOWED_USERS`、`MATRIX_AUTO_JOIN_INVITES`、`MATRIX_REPLAY_OLD_MESSAGES`、`MATRIX_STARTUP_EVENT_SKEW_MS`、`MATRIX_MESSAGE_LIMIT`、`MATRIX_MAX_IMAGE_BYTES`、`MATRIX_SYNC_TIMEOUT_MS`。
- **控制台/审核**：没有中央商店审核；需要 homeserver 账号/token、bot 加房权限和 room policy。托管 homeserver 可能有注册、速率、媒体和保留限制。
- **当前能力**：未加密 room text、plain image，以及已交付为 `RoomEncryptedImage` 且具备文件 key/hash 的 encrypted attachment 解密；typing、allowlist、group modes、旧 timeline 过滤、图片大小/MIME 检查。
- **残余风险**：这不等于完整 E2EE room 支持。当前没有 crypto store、固定设备验证、cross-signing、key backup/recovery 或已验证设备策略；没有持久 sync token/event dedup。成员数近似 DM 也不如 `m.direct` 映射准确。
- **官方参考**：[Matrix Client-Server API](https://spec.matrix.org/latest/client-server-api/)、[matrix-nio](https://github.com/matrix-nio/matrix-nio)。更完整评审见 [XMPP/Matrix 集成报告](./XMPP_MATRIX_ADAPTER_INTEGRATION_REPORT_2026-08-31.md)。

## 5. 本轮明确排除的平台与理由

“排除”分为两类：没有适合的官方通用 bot API，以及技术可接但本轮优先级较低。不能把二者混为一谈。

| 平台/方式 | 本轮判断 | 理由与未来条件 |
|---|---|---|
| 个人微信 | 不开发 | 没有面向第三方服务器的官方个人账号通用收发 bot API。基于桌面注入、UI 自动化、逆向协议或个人号挂机的方案不稳定，可能触发封号和合规风险，不作为正式 adapter。 |
| 微信公众号 | 不作为普通聊天 adapter | 有官方 server API，但主要是关注者—公众号 XML webhook/客服消息模型，存在快速响应、客服窗口、菜单、素材和审核约束；没有普通群聊语义。若产品明确需要公众号，应独立做“公众号客服入口”，不能伪装成个人微信 adapter。 |
| 企业微信群自定义 webhook | 不开发 | 主要是向群发送，缺少本 chatbot 所需的完整双向用户消息/稳定身份边界。本轮已选择官方“企业微信智能机器人”长连接。 |
| Mattermost | P2 候选，非技术阻塞 | 官方 bot account、WebSocket 和 REST 都较容易；但主要是组织自建/私有部署，每个实例的 URL、管理员 token、版本和插件策略不同，广泛用户覆盖低于本轮 11 平台。完成统一 runtime 后可低成本加入。参考：[Mattermost Bot Accounts](https://developers.mattermost.com/integrate/reference/bot-accounts/)。 |
| Rocket.Chat | P2 候选，非技术阻塞 | 官方 Apps-Engine/Realtime/REST 可实现，但同样是实例级管理、版本碎片和自建运维；本轮不继续扩大企业自建平台矩阵。参考：[Rocket.Chat bot 开发](https://developer.rocket.chat/docs/bots)。 |
| Zulip | P2 候选，技术较容易 | 官方 Python API 和 bot 机制成熟，stream/topic 会话语义也适合本项目；但市场较集中，优先级低于 Slack/Teams/Google Chat。参考：[Zulip API](https://zulip.com/api/)。 |
| Signal | 不开发正式 adapter | Signal 没有面向第三方的官方通用 bot/server API；常见 `signal-cli` 是社区工具，不是官方稳定平台契约。除非 Signal 发布正式 bot API，或用户明确接受非官方账号自动化风险，否则不进入生产范围。 |
| iMessage | 不开发普通 bot | Apple 没有开放任意 iMessage 账号的通用 server bot API。Messages for Business 是经批准的企业客服/服务提供商模式，iMessage app extension 也不是后台收发 bot。若业务拥有 Messages for Business 资质，应另立项目。参考：[Apple iMessage / Messages for Business](https://developer.apple.com/imessage/)。 |

还可评估 KakaoTalk、Viber 等地区平台，但它们通常需要当地 business channel、商务条款或额外审核；应由实际用户地域和商业资质驱动，而不是仅因存在 API 就继续无上限扩张 adapter 数量。

## 6. 正式接入主体程序所需修改

本节只说明未来需要修改什么；本轮报告和 adapter 交付**没有执行这些改动**。

### 6.1 `bot.py`：一个入口、一个 runtime、结构化监督

1. 增加 `ENABLE_DISCORD`、`ENABLE_SLACK`、`ENABLE_TEAMS`、`ENABLE_LARK`、`ENABLE_WECOM`、`ENABLE_DINGTALK`、`ENABLE_QQ`、`ENABLE_LINE`、`ENABLE_WHATSAPP`、`ENABLE_MESSENGER`、`ENABLE_GOOGLE_CHAT`、`ENABLE_XMPP`、`ENABLE_MATRIX`，均默认 `false`。保留 `ENABLE_TELEGRAM`。
2. 启动任何 worker 前，先对所有已启用平台做配置预检；日志只能报缺失字段名，不能打印 token、secret、session webhook 或完整 service-account JSON 路径内容。
3. 只执行一次 `AdapterRuntime.create()`，把同一实例注入全部 adapter。顶层只关闭一次 runtime。
4. 将 `start_telegram_adapter()` 改为接受 `runtime: AdapterRuntime | None`；统一入口必须传共享实例，独立开发入口才允许自行创建。
5. 使用 `asyncio.TaskGroup` 或等价的显式 task registry。任务命名为 `adapter:<platform>`，区分：配置/认证永久失败（fail-fast）、网络短故障（transport 自重连）、平台限流（退避）、正常取消（有界 drain）。
6. 任一不可恢复失败时取消并等待其他 adapter，先停止入站、再关闭 dispatcher/background tasks、最后 flush ingestion/growth/runtime。设置总 shutdown deadline，超时记录可恢复状态。
7. 保留进程单实例锁。不要同时运行统一入口和多个 `python -m src.bot.<platform>_adapter` 指向同一 data 目录。

### 6.2 共享 runtime 解耦

当前 `adapter_support.py` 为复用 Telegram 的 prompt、紧急回复和 model setting 函数而反向导入 `telegram_adapter.py`。应把 transport-neutral 能力迁到例如：

- `src/bot/runtime.py`：`AdapterRuntime` 构造、所有权和关闭；
- `src/bot/commands.py`：`/start`、`/identity`、`/clear`、`/memory_status`、`/model`；
- `src/bot/persona.py`：system prompt 与 private-memory emergency reply；
- `src/bot/dispatch.py`：统一消息 envelope、防抖、限流、typing、附件策略。

Telegram 与所有新增 adapter 只依赖这些中立模块。全局 `BOT_MAX_CONCURRENCY` semaphore、memory、journal、ingestion worker、growth worker、model settings store 和 coordinator 都必须只存在一份。

### 6.3 身份键与会话键必须分离

建议正式定义两个不可混用的类型：

```text
IdentityKey     = platform : tenant/provider/app-scope : stable-user-id
ConversationKey = platform : tenant/provider : space/chat/room : thread : stable-user-id
```

- **IdentityKey** 用于长期私人记忆、模型偏好和可撤销账号绑定。
- **ConversationKey** 用于防抖、工作记忆、并发锁、`/clear` 和回复路由。
- 群聊短期上下文必须包含 room/channel/thread；同一用户在两个群里不能共享最近对话。
- 长期身份默认跨会话但不跨平台。跨平台账号绑定必须由用户完成验证、可审计、可撤销，不能用昵称、手机号文本或邮箱猜测。
- 对 app-scoped ID（Lark、LINE、WeCom 多企业、多 Meta Page/phone、Google Chat）必须保留 scope。当前单 app 安全但多 app 会碰撞的平台，应在迁移前给 key 增加 scope 并提供数据迁移。

### 6.4 持久事件去重与 at-least-once 处理

建立共享 durable dedup store，例如 SQLite/数据库表：

```text
(platform, account_scope, event_id) PRIMARY KEY
received_at, status(reserved|completed|failed), conversation_key, expires_at
```

平台事件键建议：Telegram `update_id`；Discord message/event ID；Slack `event_id`；Teams Activity ID；Lark message ID；WeCom `msgid`；DingTalk `msgId`；QQ message ID；LINE `webhookEventId`；Meta message/mid；Google Chat message resource/event fingerprint；Matrix event ID；XMPP stanza origin-id/stanza-id（无 ID 时谨慎降级指纹）。

处理顺序应为：认证/签名 → schema/allowlist → 原子 reserve → 快速 ACK/入队 → 完成发送和 journal 后标记 completed。需要定义崩溃时 reserved 事件的租约与重试，而不是简单永久吞掉。TTL 至少覆盖平台最大重投窗口和部署回滚窗口；高价值长期记忆写入还应带幂等 batch ID。

### 6.5 `requirements` 与构建矩阵

建议不要把所有平台无条件塞进单一基础镜像，而是提供 extras 或分组：

```text
requirements-core.txt
requirements-adapters-western.txt
requirements-adapters-asia.txt
requirements-adapters-webhooks.txt
constraints-python311.txt
```

CI 至少覆盖 core/Telegram、每个平台最小 extra、all-adapters import，以及 Python 3.11/3.12 的 lock 安装。每次升级要回归官方 SDK 的构造器、异步/同步方法、callback contract、关闭语义和底层 `aiohttp/httpx/websockets` 冲突。尤其需要锁住 QQ 1.2.2 的实际 callback contract、DingTalk 0.24.3 的无 `stop()` 行为、LINE generated async API 和 WeCom 新 SDK。

### 6.6 `.env.example` 与 secret 管理

- 为每个平台登记 `ENABLE_*`、必需变量和所有当前代码读取的可选变量；值只能是假值或说明。
- allowlist 为空目前多表示“允许全部”。生产应改为 fail-closed，或要求额外 `*_ALLOW_ALL=true` 才能空 allowlist 启动。
- token、secret、password、service-account key、Meta Page token、LINE access token 不应提交到 Git；使用 secret manager、workload identity 或受限文件。
- 给 token rotation、双 token 灰度、撤销、审计和过期告警建立运维手册。
- webhook host/port/path 在多平台同进程时应统一由一个 HTTP server/router 承载，避免端口冲突和重复 TLS/health server。

## 7. TLS、网络与 webhook 生产要求

1. LINE、WhatsApp、Messenger、Google Chat 和 Teams 的公网入口只接受现代 HTTPS；本地 adapter 监听端口不得直接暴露互联网。由 Nginx/Envoy/云 LB/Cloud Run 等终止 TLS，并保留原始 body，不得在签名校验前改写 JSON。
2. 分平台设置独立 path、body size、request timeout、并发上限和速率限制。禁止把 token 放在 query string；代理日志必须过滤 Authorization、signature 和 webhook secret。
3. LINE/Meta 的 HMAC 必须覆盖精确原始 bytes；Google Chat 必须校验 OIDC audience/issuer；Teams 必须保持 SDK JWT 验证，生产严禁 `skip_auth`。
4. webhook 应先鉴权、再解析、再 durable reserve，并尽快 2xx。LLM、媒体下载和发送不应占用平台同步响应窗口。
5. Discord/Slack/Lark/WeCom/DingTalk/QQ 的 WSS，以及 Telegram/Matrix/Graph/Chat REST 都必须验证服务端证书和主机名；不提供“忽略证书”的生产开关。
6. XMPP 强制符合服务器策略的 TLS/STARTTLS 和证书校验；自定义 `XMPP_HOST` 不得隐式关闭验证。Matrix homeserver 必须使用受信任 HTTPS。
7. 媒体下载要限制 HTTPS host、重定向、Content-Length、实际流式字节、MIME magic、像素/帧数和解压炸弹；阻止 loopback、私网、链路本地、云 metadata 和 DNS rebinding。

## 8. 当前测试证据与仍需补充的验证

2026-08-31 最终完整回归为 **133/133 项通过**。其中本轮 11 个新增 adapter 有 **44 项专项测试**（西方平台 18 项、亚洲平台 14 项、webhook 平台 12 项）；连同 XMPP/Matrix 与共享 adapter 策略测试，adapter 相关测试共 **64 项**。覆盖 identity、群触发、allowlist、签名/OIDC、快速 ACK、发送、typing、拆分、去重、图片边界、真实 SDK 请求模型和 fake 生命周期。`python -m compileall -q src tests` 与 `pip check` 同样通过。

这些是协议边界单元测试，不等于平台生产认证。上线前仍需：

- 每个平台使用隔离真实账号/tenant/page/phone/channel 的端到端 smoke；
- token 过期、撤销、权限减少、限流、网络分区、重连和 webhook 重投；
- 24 小时以上 soak，观察线程/任务泄漏、重复回复、内存 dedup 增长和数据 writer 冲突；
- 同一用户跨两个 room/thread 的上下文隔离；
- 一个 adapter 永久失败时其他 adapter 的监督和有界关闭；
- secret 日志扫描、恶意大 body、伪 MIME、SSRF、消息费用型 DoS；
- Meta 消息窗口/template、QQ sandbox→production、Teams 多 tenant、Lark/WeCom/DingTalk 企业管理员安装、LINE reply→push fallback 的真实额度行为；
- Matrix 完整 E2EE 若要启用，必须单独完成设备验证、crypto store、key backup/recovery 测试。

## 9. 建议实施优先级

| 优先级 | 工作 | 放行标准 |
|---|---|---|
| P0 | runtime/Telegram 解耦、统一 `bot.py` TaskGroup | 所有启用平台只创建/关闭一次 runtime；故障和取消测试通过 |
| P0 | 身份键/会话键分离 | 同用户跨 room/thread 的工作记忆完全隔离 |
| P0 | 精确依赖锁与 `.env.example` | 干净 Python 3.11/3.12 环境可复现 all-adapters 安装 |
| P0 | durable dedup/outbox | 平台重投和进程崩溃不产生重复回复或长期记忆写入 |
| P0 | 统一 HTTPS router、TLS、secret/allowlist 安全默认 | webhook 外网验证、限流和日志脱敏通过 |
| P0 | 首批 allowlist canary | Telegram、Discord、Slack、Lark、WeCom、DingTalk 真实账号稳定运行 |
| P1 | LINE/Google Chat/Teams | 平台控制台、tenant/endpoint、额度和管理员流程验收 |
| P1 | WhatsApp/Messenger/QQ | Business/App/机器人审核通过；消息窗口/权限/sandbox 验收 |
| P1 | Matrix 明文房间/XMPP 单聊 | homeserver/server soak、事件幂等和 TLS 验收 |
| P2 | Mattermost/Rocket.Chat/Zulip | 有明确用户需求后复用统一 runtime/webhook/WS 基础设施 |
| 独立专项 | Matrix E2EE、XMPP MUC/OMEMO、Meta template/media | 各自安全和恢复模型完成后再承诺支持 |

## 10. 最终判断

从代码接入成本看，Discord、Slack、Lark、企业微信智能机器人和钉钉最适合快速纳入现有 Python asyncio chatbot；LINE 和 Google Chat 的代码边界也清晰，但需要公网 HTTPS 与平台配置；Teams、WhatsApp、Messenger、QQ 的主要时间成本在企业身份、权限、审核和生产政策。Telegram 仍是可用基线，XMPP/Matrix 提供开放协议覆盖但不能替代各自的群聊/E2EE专项。

真正的下一步不是继续堆更多 adapter 文件，而是完成统一 runtime、会话隔离、持久幂等、依赖/配置契约和 TLS 入口。完成这些 P0 后，14 个 adapter 才能从“可独立测试的 transport 边界”升级为“可被主体程序可靠监督的多平台聊天机器人”。
