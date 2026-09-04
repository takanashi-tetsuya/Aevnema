# 多平台 Adapter 正式集成完成报告

日期：2026-08-31  
范围：Telegram、Discord、Slack、Microsoft Teams、Lark/飞书、企业微信、钉钉、QQ、LINE、WhatsApp、Messenger、Google Chat、XMPP、Matrix，共 14 个平台边界。

## 1. 结论

两份前置评审报告中列出的正式集成 P0 已实现：

- `python bot.py` 可按 `ENABLE_<PLATFORM>` 同时启动任意已配置的 adapter；
- 整个进程只创建一个 `AdapterRuntime`，所有平台共享 MemorySystem、ConversationCoordinator、增长/导入 worker、模型设置、视觉模型、持久化事件表和全局并发预算；
- Telegram 已接受注入的共享 runtime，消息生成、视觉处理和平台无关命令走共享实现；
- 私人长期记忆继续使用稳定的 `IdentityKey`，短期工作记忆改用包含 chat/room/thread 的 `ConversationKey`；
- 入站事件使用 SQLite 持久化 claim/completed 协议，不再依赖重启即丢失的进程内 set/TTL cache；
- 启动前统一验证可选 SDK、凭据、访问策略和监听端口冲突；
- 每个平台默认 fail-closed：必须提供白名单，或者显式设置 `<PLATFORM>_ALLOW_ALL=true`；
- 平台依赖已按 western、asia、federated、webhooks 分组，并提供全量安装文件；
- `.env.example` 已形成 14 平台部署契约；README 已改为多平台说明。

这表示 adapter 已完成主体程序集成和离线协议验证，不表示所有平台都已经用真实租户、真实回调地址和真实凭据完成在线认证。在线 canary 仍必须在各平台后台逐一配置。

## 2. 核心名词

### 2.1 Adapter

Adapter 是平台协议与共享聊天核心之间的边界。它只负责：

1. 验证平台事件及访问策略；
2. 从平台对象提取稳定用户 ID、会话地址、文本、媒体和事件 ID；
3. 把消息提交给共享 runtime；
4. 将回答按平台限制拆分并发送；
5. 正确启动、停止和释放 SDK/HTTP 资源。

Adapter 不应拥有独立的长期记忆系统、模型配置或增长 worker。

### 2.2 AdapterRuntime

`AdapterRuntime` 是进程级唯一服务容器，定义于 `src/bot/adapter_support.py`。它持有：

- 三域 `MemorySystem`；
- `ConversationCoordinator`；
- 对话 journal 与导入 worker；
- Association 后台增长 worker；
- 用户模型设置仓库；
- 视觉模型；
- 全局 `asyncio.Semaphore`；
- SQLite 事件去重器。

由 `bot.py` 创建一次并注入所有 adapter。独立运行某个 adapter 文件时，该 adapter 可以临时拥有自己的 runtime，但仍受同一访问策略约束。

### 2.3 IdentityKey

`IdentityKey = platform + platform_user_id`，例如：

```text
telegram:123456789
matrix:@teacher:example.org
slack:T123:U456
```

它用于私人长期记忆、用户模型设置和对话 journal。显示名不参与主键，因为显示名会变化、可重复，也可能被冒用。

### 2.4 ConversationKey

ConversationKey 用于进程内短期上下文和同一会话的串行化，通常包含：

```text
platform + tenant/account + room/chat/channel + thread/topic + user
```

因此同一用户在私聊、群聊和不同 thread 中可以共享长期记忆，但不会共享最近若干轮原始对话。`/clear` 只清除当前 ConversationKey。

### 2.5 EventClaim

EventClaim 是平台事件的持久化处理权，唯一键为：

```text
(platform, account_scope, event_id)
```

状态协议与回答提交顺序：

```text
首次到达
  → claimed（带 lease）
  → 生成回答但暂不写会话副作用
  → 平台回答发送成功
  → 增长队列、journal 与短期会话提交
  → completed（保留 TTL）

处理失败/取消
  → release
  → 平台重试时可以重新 claim
```

进程崩溃留下的 `claimed` 会在 lease 到期后允许重试。completed 默认保留七天。旧的 adapter 内存去重层已经移除，避免它在共享处理失败后错误阻断合法重试。

## 3. 启动与关闭设计

`bot.py` 的顺序为：

```text
加载 .env
  → 解析所有 ENABLE_<PLATFORM>
  → 汇总验证依赖、凭据、访问策略、端口
  → 创建唯一 AdapterRuntime
  → asyncio.TaskGroup 并行启动所有 adapter
  → 任一 adapter 异常退出，取消其他 adapter
  → AdapterRuntime 统一 flush/close
```

这个顺序避免了配置错误发生在大型 embedding 索引加载之后，也避免某个平台掉线后其他平台仍使用一个处于不确定状态的共享进程。

运行时初始化本身具有回滚：若模型、worker 或事件数据库在创建过程中失败，已经启动的 worker 会被关闭。正常关闭时，即使一个 worker 关闭失败，也会继续尝试关闭其他 worker 和 SQLite 连接，并用 `ExceptionGroup` 汇总错误。

## 4. 事件、并发与副作用边界

### 4.1 全局并发

所有平台共享 `BOT_MAX_CONCURRENCY`，所以同时启用 14 个平台不会把允许的 LLM/检索并发放大 14 倍。不同 ConversationKey 可以并发；同一 ConversationKey 保持顺序。

### 4.2 防抖

连续消息先按 ConversationKey 合并，再占用全局并发预算。事件 claim 随合并批次保存；只有整个批次成功完成才将其中所有事件标为 completed。

### 4.3 Webhook 快速 ACK

WhatsApp、Messenger 和 Google Chat 在返回成功 ACK 前同步完成签名/OIDC、JSON/大小检查、allowlist 检查和 SQLite claim，并把消息加入共享 dispatcher。模型调用仍在 ACK 之后异步执行，不占用平台 webhook 时限。

注意：当前 SQLite 表是事件处理账本，不是保存完整 webhook payload 的 durable inbox。如果进程在成功 ACK 后、dispatcher 完成前永久损坏且平台不再重投，账本只能显示未完成 claim，不能凭空恢复原 payload。要求严格无损的部署应在下一阶段增加包含加密原始 payload 的 durable inbox/outbox；这不影响当前的重复副作用防护，但属于更高一级的交付保证。

## 5. 安全和部署约束

- 所有 enabled adapter 均要求白名单或显式 `ALLOW_ALL`；
- 群聊平台默认使用 mention/direct 等保守模式；Telegram 新增 `TELEGRAM_GROUP_MODE=mentions`；
- webhook 在解析 JSON 前验证 Meta HMAC、LINE signature 或 Google OIDC；
- 请求体与图片均有大小上限；Telegram 新增 `TELEGRAM_MAX_IMAGE_BYTES`；
- 多 webhook 监听同一端口时，preflight 会拒绝启动；
- 公网 webhook 必须置于 TLS 反向代理之后；
- Matrix 当前不宣称完整 E2EE；XMPP 当前不宣称完整 MUC/媒体支持。

## 6. 依赖与配置资产

新增依赖文件：

- `requirements-adapters-western.txt`
- `requirements-adapters-asia.txt`
- `requirements-adapters-federated.txt`
- `requirements-adapters-webhooks.txt`
- `requirements-all-adapters.txt`

核心与 Telegram 继续使用 `requirements.txt`。所有开关、凭据、白名单、监听地址和主要模式均已写入 `.env.example`，路径保持相对形式，不绑定本机目录。

## 7. 测试结果

完成集成后的全量 chatbot 回归：

```text
155 tests passed
13 tests skipped
0 failures
```

跳过项只对应当前 WSL 环境未安装的可选 SDK（Matrix、Teams、LINE、DingTalk、Google Chat 的部分真实 SDK 对象测试）。相关 adapter 的纯策略、identity、事件解析、fake transport、关闭路径和共享 dispatcher 测试仍然执行。

另外执行了 `pip install --dry-run --ignore-installed -r requirements-all-adapters.txt`。全部声明的发行包和版本范围成功解析，包括 Python 3.14 对应的可用 wheel；该命令未安装任何依赖。

本轮新增或强化的测试包括：

- completed 事件跨进程重启仍去重；
- release 后允许平台重试；
- 不同 bot/account scope 的同号事件不冲突；
- dispatcher 仅在完整成功后 complete；
- 平台发送成功后才提交 journal/增长，发送失败不提交本地记忆；
- 同一身份跨房间的短期上下文隔离；
- fail-closed 访问策略；
- 错误布尔配置；
- webhook 监听端口冲突；
- 一个 adapter 失败时取消 peer，并且共享 runtime 只关闭一次。

## 8. 当前部署迁移状态

现有本机 `.env` 只启用了 Telegram，但尚未设置 `TELEGRAM_ALLOWED_USERS`、`TELEGRAM_ALLOWED_CHATS` 或 `TELEGRAM_ALLOW_ALL=true`。因此新 preflight 会按设计拒绝启动，且不会先加载记忆索引。

部署者必须作出明确选择：

```dotenv
TELEGRAM_ALLOWED_USERS="稳定的 Telegram 数字用户 ID"
```

或明确接受公开访问：

```dotenv
TELEGRAM_ALLOW_ALL=true
```

推荐使用稳定数字用户 ID 白名单。

## 9. 尚需真实平台完成的验收

1. 为要上线的平台安装对应 requirements 分组；
2. 在平台控制台配置 bot/app、权限、事件订阅和回调 URL；
3. 先只启用一个平台完成私聊 canary；
4. 验证 group mention、thread reply、媒体大小限制和 allowlist；
5. 人为重投同一 event ID，确认没有重复回复和记忆增长；
6. 在生成过程中重启进程，验证 lease/retry 行为；
7. 再逐个平台加入同一进程，观察共享并发和 shutdown；
8. 若需要 Matrix 加密房间或 XMPP MUC，再单独进入对应协议阶段。
