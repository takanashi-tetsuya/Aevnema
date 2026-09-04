# XMPP / Matrix Adapter 集成评审报告

> 后续状态（2026-08-31）：本报告指出的统一启动、共享 runtime、会话隔离、持久化去重、依赖与 `.env.example` P0 已完成。实现结果与仍保留的 Matrix E2EE / XMPP MUC 边界见 `ADAPTER_INTEGRATION_COMPLETION_REPORT_2026-08-31.md`。下文保留为改造前审计快照。

日期：2026-08-31  
范围：`src/bot/xmpp_adapter.py`、`src/bot/matrix_adapter.py`、`src/bot/adapter_support.py`、现有 `src/bot/telegram_adapter.py` 与未来统一启动路径  
结论：本轮只新增 adapter、adapter 支撑代码、对应测试和本报告；**没有修改 `bot.py`、`requirements.txt`、`.env.example`、Telegram adapter 或其他主体程序**。因此 XMPP 与 Matrix 当前是“可独立调用的 adapter 实现”，尚未接入 `python bot.py` 的正式三平台启动流程，也不应直接视为生产就绪。

## 1. 当前交付快照

### 1.1 已具备

| 能力 | XMPP | Matrix |
|---|---|---|
| 稳定用户身份 | bare JID，归一化为小写，忽略 resource | 完整 Matrix user ID |
| 文本收发 | `chat` / `normal` stanza 单聊 | `m.room.message` 文本 |
| 连续消息防抖 | 有 | 有 |
| 长回复拆分 | 有，默认 8,000 字符 | 有，默认 12,000 字符 |
| 命令 | `/start`、`/identity`、`/clear`、`/memory_status`、`/model` | 同左 |
| 访问控制 | `XMPP_ALLOWED_JIDS` | `MATRIX_ALLOWED_ROOMS`、`MATRIX_ALLOWED_USERS`、房间响应模式 |
| 输入图片 | 尚无 | 明文房间中的 `m.image` 下载、大小和 MIME 初筛、视觉模型描述 |
| 历史消息策略 | 服务端实时 stanza；未实现 stanza 去重 | 默认按启动时间过滤旧事件，可显式允许回放 |
| 群聊 | 尚无 MUC | direct / mentions / all 三种模式 |
| 加密 | 依赖连接层 TLS；未实现 OMEMO | 当前未启用 E2EE |

`AdapterRuntime` 复用了现有对话协调器、三域记忆、后台增长、模型设置、图片描述和并发上限。`DebouncedDispatcher` 按会话合并突发消息，并按平台身份串行处理。测试目前覆盖身份规范化、Matrix 房间策略、拆分、防抖和处理期间到达的新消息不丢失等纯逻辑。

### 1.2 当前不能承诺

- `bot.py` 仍只读取 `ENABLE_TELEGRAM` 并启动 Telegram；设置 XMPP/Matrix 环境变量不会让主体程序启动它们。
- 新依赖尚未写入 `requirements.txt`，标准安装环境不能保证存在 `slixmpp` 或 `matrix-nio`。
- 示例配置尚未写入 `.env.example`，部署者没有正式配置契约。
- XMPP adapter 只处理一对一纯文本；MUC、文件/图片、引用、表情反应和 OMEMO 都不在当前范围。
- Matrix adapter 只支持未加密文本与图片；没有加密状态存储、设备验证、密钥恢复或加密附件处理。
- 当前没有持久事件去重。断线重连、服务端重放或进程崩溃窗口内，存在重复回复/重复写入记忆的可能。
- 当前没有针对真实 XMPP server / Matrix homeserver 的集成测试、长时重连测试和生产凭据 smoke test。

## 2. 真正接入 `bot.py` 所需修改（P0）

建议将三平台置于**同一进程、同一个共享运行时**中，而不是让三个 adapter 各自初始化一套记忆服务。

1. 在 `bot.py` 导入 `start_xmpp_adapter`、`start_matrix_adapter` 和共享运行时工厂。
2. 增加 `ENABLE_XMPP`、`ENABLE_MATRIX` 分支；三个开关均默认 `false`，至少启用一个才运行。
3. 启动前先完成所有已启用平台的配置预检。缺少凭据、端口非法、房间模式非法时应在连接前失败，并只输出字段名，绝不输出秘密值。
4. 只调用一次 `AdapterRuntime.create()`，将同一个实例注入所有已启用 adapter；顶层在所有平台停止后只调用一次 `runtime.close()`。
5. 将任务命名为 `adapter:telegram`、`adapter:xmpp`、`adapter:matrix`，记录每个平台的启动/退出状态。明确故障策略：建议认证/配置错误使整个进程 fail-fast；短时网络错误由各 transport 指数退避重连，不应拖垮其他平台。
6. 使用结构化并发（`asyncio.TaskGroup` 或等价的显式取消/回收），确保任一永久失败时取消并等待其余任务，再关闭共享 worker，避免遗留防抖任务和未提交对话批次。
7. 保留现有 `BotProcessLock` 作为统一入口的单实例保护。独立执行 `python -m src.bot.xmpp_adapter` / `matrix_adapter` 当前绕过该锁，生产部署应禁用这种旁路，或给独立入口补同等级锁与独立数据目录。

不建议直接在当前 `active_tasks` 中追加三个无参启动函数：Telegram 和每个无参新 adapter 会分别创建记忆、journal、增长 worker 和模型设置存储，造成同一路径多 writer、重复初始化、重复缓存及关闭次序不确定。

## 3. Telegram adapter 与共享运行时所需修改（P0）

当前 Telegram 自己构造完整运行时，而 `adapter_support.py` 又从 Telegram adapter 导入 persona prompt、紧急回复和 `/model` 设置函数。真正共享前应消除这种反向依赖。

建议改造如下：

- 将 persona prompt、私有记忆紧急回复、模型命令解析迁到 transport-neutral 模块，例如 `src/bot/runtime.py` / `src/bot/commands.py`；Telegram 和两个新 adapter 只依赖公共层，公共层不再依赖 Telegram。
- 让 `start_telegram_adapter(runtime: AdapterRuntime | None = None)` 支持注入；统一入口传共享实例，独立开发启动时仍可自行创建并负责关闭。
- 将 Telegram 的 `application.bot_data` 映射到共享 runtime 内的 coordinator、memory、workers、vision engine、model settings 和全局 semaphore；不要再构造第二套实例。
- 将 Telegram 现有的文本/图片处理、防抖、命令和异常映射逐步改用公共 dispatcher/runtime，或至少建立契约测试，防止三平台行为漂移。
- 并发上限必须是进程级共享 semaphore；否则每个平台各允许 `BOT_MAX_CONCURRENCY`，实际总并发会放大三倍。
- `/clear`、工作记忆和处理锁需同时接受“长期身份键”与“会话键”。长期身份继续用 `platform:user_id`；短期会话应至少用 `platform:chat_or_room:user_id`。当前 Telegram 和 Matrix 的防抖键含 chat/room，但 `ConversationCoordinator` 的 session 仍只按 `identity.key`，同一用户跨群/房间可能串接短期上下文。这是开启任何群聊模式前必须解决的 P0 隔离问题。
- 明确跨平台账号是否合并。默认应保持 Telegram/XMPP/Matrix 三个私人记忆域分离；未来若要绑定账号，必须经过用户验证和可撤销映射，不能靠显示名、昵称或邮箱猜测。

## 4. `requirements.txt` 所需修改（P0）

按当前 adapter 中的运行时检查，至少加入：

```text
slixmpp>=1.17,<2
matrix-nio>=0.26,<0.27
```

若生产目标包含 Matrix E2EE，应选择并锁定项目验证过的 E2EE extra（通常为 `matrix-nio[e2e]`）及其底层加密依赖，而不是同时保留普通包和 extra 的重复条目。需要在目标 Python 版本和目标操作系统上验证 wheel/本地库可安装性，再固化 lock/constraints 与依赖哈希。

建议将平台依赖拆成 extras 或 requirements 分组，使只运行 Telegram 的部署无需安装两套 transport；无论采用何种布局，CI 必须至少有一个“all adapters”环境验证三者可以同时 import。版本升级尤其要回归 Slixmpp 的 `connect`/事件回调语义和 matrix-nio 的登录、sync、download、加密存储 API。

## 5. `.env.example` 配置契约（P0）

示例文件只放假值，生产秘密应由 secret manager、受限环境文件或容器 secret 注入。平台开关建议默认关闭；allowlist 为空目前表示“不限制”，生产上应改为 fail-closed，或者必须再显式设置 `*_ALLOW_ALL=true` 才允许空 allowlist。

### 5.1 当前代码已经读取、集成时必须登记的字段

| 字段 | 必需条件 / 默认 | 敏感 | 说明 |
|---|---|---:|---|
| `ENABLE_TELEGRAM` | 默认 `false` | 否 | 统一入口启用 Telegram |
| `ENABLE_XMPP` | 新增，默认 `false` | 否 | 统一入口启用 XMPP |
| `ENABLE_MATRIX` | 新增，默认 `false` | 否 | 统一入口启用 Matrix |
| `XMPP_JID` | 启用 XMPP 时必需 | 否 | bot 完整 JID |
| `XMPP_PASSWORD` | 启用 XMPP 时必需 | **是** | 账号密码；不得记日志 |
| `XMPP_HOST` | 与 `XMPP_PORT` 同时设置 | 否 | 覆盖 DNS/SRV 发现 |
| `XMPP_PORT` | 与 `XMPP_HOST` 同时设置 | 否 | 1–65535，启动时校验整数和范围 |
| `XMPP_ALLOWED_JIDS` | 当前空值=允许所有 | 否 | 逗号分隔 bare JID；生产建议必填或显式 allow-all |
| `XMPP_MESSAGE_LIMIT` | 默认 `8000` | 否 | 正整数；服务端更严时下调 |
| `MATRIX_HOMESERVER` | 启用 Matrix 时必需 | 否 | 仅允许受信任的 HTTPS homeserver URL |
| `MATRIX_USER_ID` | 启用 Matrix 时必需 | 否 | 完整 MXID，如 `@bot:example.org` |
| `MATRIX_ACCESS_TOKEN` | 与 password 二选一，推荐 | **是** | 应可轮换、撤销，绝不记日志 |
| `MATRIX_PASSWORD` | 无 token 时必需 | **是** | 仅用于登录换 token；不建议长期明文保存 |
| `MATRIX_DEVICE_ID` | token/E2EE 强烈建议固定 | 否 | 必须与 token 和加密 store 配套 |
| `MATRIX_DEVICE_NAME` | 默认 `chat_bot matrix adapter` | 否 | 密码登录创建的设备名 |
| `MATRIX_ROOM_MODE` | 默认 `direct` | 否 | `direct` / `mentions` / `all`；生产先用 `direct` |
| `MATRIX_ALLOWED_ROOMS` | 当前空值=允许所有 | 否 | 逗号分隔 room ID |
| `MATRIX_ALLOWED_USERS` | 当前空值=允许所有 | 否 | 逗号分隔完整 MXID |
| `MATRIX_AUTO_JOIN_INVITES` | 默认 `false` | 否 | 仅允许 allowlist 内邀请；不要对互联网开放自动加入 |
| `MATRIX_REPLAY_OLD_MESSAGES` | 默认 `false` | 否 | 开启会导致旧事件可能被回复/写入，需先有持久去重 |
| `MATRIX_STARTUP_EVENT_SKEW_MS` | 默认 `5000` | 否 | 启动历史过滤容差，校验非负整数 |
| `MATRIX_MESSAGE_LIMIT` | 默认 `12000` | 否 | 正整数 |
| `MATRIX_MAX_IMAGE_BYTES` | 默认 `10485760` | 否 | 下载前后都校验；生产可按视觉 API 限制下调 |
| `MATRIX_SYNC_TIMEOUT_MS` | 默认 `30000` | 否 | 正整数；不是业务请求总超时 |

### 5.2 后续功能需要新增、但当前代码尚未读取的字段

| 字段组 | 建议字段 | 用途 |
|---|---|---|
| Matrix E2EE | `MATRIX_E2EE_ENABLED`、`MATRIX_STORE_PATH`、`MATRIX_DEVICE_TRUST_POLICY`、`MATRIX_KEY_BACKUP_*` | 加密开关、持久 crypto store、设备信任和密钥恢复 |
| Matrix 去重 | `MATRIX_SYNC_STORE_PATH`、`MATRIX_EVENT_DEDUP_TTL_DAYS` | 保存 sync token 和已处理 event ID |
| XMPP MUC | `XMPP_MUC_ROOMS`、`XMPP_MUC_NICK`、`XMPP_MUC_PASSWORDS_FILE`、`XMPP_MUC_MODE`、`XMPP_ALLOWED_MUC_SENDERS` | 加入房间、昵称、房间凭据、mention/all 策略和成员控制 |
| XMPP 媒体 | `XMPP_MEDIA_ENABLED`、`XMPP_MAX_MEDIA_BYTES`、`XMPP_MEDIA_ALLOWED_HOSTS` | 显式启用附件、限制体积和下载源 |
| 重连/健康 | `ADAPTER_RECONNECT_MIN_SECONDS`、`ADAPTER_RECONNECT_MAX_SECONDS`、`ADAPTER_HEALTH_STALE_SECONDS` | 指数退避和存活判定 |

`*_PASSWORDS_FILE`、恢复密钥等不要作为可提交的明文示例值；示例应说明使用 secret 文件/secret manager，文件权限仅允许运行账号读取。

## 6. Matrix E2EE 专项（P0：若要进入加密房间）

当前 `AsyncClient(homeserver, user_id)` 没有加密 client config、持久 store 或设备信任流程，因此不能宣称支持加密房间。进入 E2EE 房间前至少完成：

1. 安装并验证 matrix-nio 的 E2EE 依赖；以启用 encryption 的 client config 创建客户端。
2. 为 crypto store 使用持久、受限权限、可备份但不可公开的目录；token、固定 device ID 与 store 必须成套保留。每次创建新 device/store 会让对端重新验证，并可能无法解密历史。
3. 登录后上传设备密钥、持续 sync，并只把成功解密且通过策略检查的事件交给业务层。解密失败应报告可观测指标，不能把密文或错误对象交给 LLM。
4. 明确设备信任策略。建议生产默认只接收已验证设备的私聊/附件，或者为未验证设备提供明确但不泄露内容的提示；不能静默自动信任所有新设备。
5. 处理交叉签名、密钥请求、设备验证与 key backup/recovery；编写 token 轮换、设备吊销、store 损坏和灾难恢复手册。
6. 加密媒体必须按 Matrix 加密文件元数据下载并验证哈希后解密，再执行解密后大小、MIME、像素/解压炸弹检查。当前直接 `download(event.url)` 的路径只适用于未加密媒体。
7. 持久保存 sync token 和已处理 event ID；重启后的 at-least-once 事件不能重复调用 LLM、重复回复或重复进入长期记忆。

如果首期不做 E2EE，应在 README/部署配置中明确“仅支持未加密房间”，并让 adapter 对加密房间 fail-closed，而不是看似在线却静默漏消息。

## 7. XMPP MUC 与媒体专项（P1）

### 7.1 MUC

当前 adapter 明确只接受 `chat` / `normal`，没有 `groupchat`。实现 MUC 需要注册并使用 XEP-0045，在 session ready 后加入显式 allowlist 房间，并处理加入失败、踢出、昵称冲突和重连后重入。

响应策略建议与 Matrix 对齐：`direct`、`mentions`、`all`，默认不在 MUC 响应；忽略自身 stanza、历史回放、延迟投递和其他 bot；只在被 mention 时去掉规范化 mention 后交给 LLM。

MUC 身份是关键风险：`room@conference/nick` 不是稳定账号。在非匿名 MUC 可使用服务器暴露的真实 bare JID；匿名/半匿名 MUC 无法取得真实 JID 时，应使用带 room scope 的伪身份，且不得声称它能跨昵称、跨房间绑定私人记忆。短期会话键必须包含 room JID；长期私人记忆是否允许匿名 MUC 写入，应默认关闭并由产品明确决定。

### 7.2 媒体

建议分阶段支持 XEP-0066（Out of Band Data）和实际服务器采用的 HTTP Upload/附件描述；只有明确需要发送附件时再实现 XEP-0363。不要把任意聊天正文 URL 当附件自动抓取。

下载器必须：仅允许 HTTPS；阻止 loopback、链路本地、私网、云 metadata 和 DNS rebinding；限制重定向次数、允许 host、Content-Length、流式实际字节数、超时和并发；下载后以 magic bytes 验证真实类型，并限制图片像素/帧数。临时文件需随机命名、受限权限并在处理后删除。若未来加入 OMEMO，它是独立安全项目，不能用 TLS 连接等同于端到端加密。

## 8. 部署与安全要求（P0）

- 推荐一个进程、一个共享运行时、一个数据目录和一个进程锁。若因故拆成三个进程，必须先证明记忆数据库、journal、增长队列和 JSON 模型设置支持跨进程并发，否则应分离写路径并增加专门的单 writer 服务。
- 为 bot 使用专用低权限系统账号；限制 `.env`、Matrix store、token、XMPP 密码和用户记忆目录权限。日志、异常、健康接口不得输出 token、密码、完整消息正文或媒体 URL 中的秘密参数。
- XMPP 强制 TLS 并验证证书与服务器名；自定义 host 不得默认跳过证书验证。Matrix homeserver 强制 HTTPS，代理配置和 CA 变更需审计。
- 生产 allowlist fail-closed；Matrix 自动邀请关闭；群房间默认 direct/mentions；高成本 `/model`、`/memory_status` 等命令可增加管理员/用户级授权和频率限制。
- 在平台速率限制之外增加每用户、每房间、全局 token/请求预算；对消息长度、图片数、图片体积和待处理防抖缓冲设置硬上限，避免内存与 LLM 费用型 DoS。
- transport 发送错误不要把内部异常原样回给用户；用户看到稳定错误码/友好提示，详细堆栈只进脱敏日志。为认证失败、连续 sync 失败、队列堆积、消息处理延迟、去重命中和 E2EE 解密失败建立指标告警。
- 采用带抖动的指数退避和上限；认证/配置错误不应无限重试。发布、回滚和进程退出都应等待 dispatcher/ingestion/growth 的有界 drain，超时后留下可恢复队列。

## 9. 测试计划

### 9.1 合并前自动化（P0）

1. 三种依赖组合：core/Telegram、XMPP、Matrix、all-adapters import 与启动配置校验。
2. `bot.py` 开关矩阵：无平台、单平台、任意双平台、三平台；断言共享 runtime 只创建/关闭一次，失败时所有任务被取消并回收。
3. transport mock：XMPP 登录/失败认证/重连/自身消息/allowlist/拆分；Matrix login/token、room mode、invite、typing、send/download 错误、大小和 MIME 限制、旧事件过滤。
4. 事件幂等：同一个 stanza ID / Matrix event ID 重放不会重复回复、重复 journal 或增长任务。
5. 隔离：同一用户跨两个房间的短期上下文不串；不同平台同名用户的私人记忆不串；显式账号绑定另行测试。
6. 关闭与竞态：处理期间新消息、取消时缓冲 drain、worker 初始化半失败、一个 adapter 掉线时其他 adapter 继续、共享 runtime 只由 owner 关闭。
7. 安全：空 allowlist 策略、恶意超长消息、伪 MIME、重定向/SSRF、压缩炸弹、秘密日志扫描、Matrix 未解密事件不进入 LLM。

### 9.2 预发布集成（P0/P1）

- 使用隔离的 XMPP 测试域与 Matrix staging homeserver，分别从两个账号、两个设备、两个房间测试文本、命令、图片、断网、服务重启、token 撤销与限流。
- 至少进行一次 24 小时 soak：周期性断网/恢复，观察重连风暴、重复回复、内存增长、文件句柄、队列 backlog 与长期记忆写入。
- Matrix E2EE 需覆盖已验证/未验证设备、密钥缺失、加密图片、store 迁移与恢复；XMPP MUC 需覆盖匿名/非匿名房、历史 stanza、改昵称、踢出重入。
- 运行现有完整回归测试，确认 Telegram 文本、图片、命令、模型覆盖、私人记忆防泄漏和后台增长没有行为退化。

## 10. 建议上线顺序与优先级

| 优先级 | 工作 | 放行标准 |
|---|---|---|
| P0 | requirements 与 `.env.example` 契约 | 干净环境可复现安装；秘密/开关/allowlist 默认安全 |
| P0 | 公共 runtime 解耦、Telegram 注入、`bot.py` 三平台结构化启动 | 三平台共享一次初始化/关闭；故障与取消测试通过 |
| P0 | 会话键与身份键分离 | 跨房间短期上下文隔离测试通过 |
| P0 | 事件持久去重、同步状态和重连策略 | 重放不产生重复回复/记忆写入 |
| P0 | 单聊 staging 验收与部署加固 | Telegram + XMPP 单聊 + Matrix 明文单聊稳定运行 |
| P0（若启用） | Matrix E2EE | 验证、解密媒体、store/backup/恢复测试通过 |
| P1 | Matrix 群聊 mentions 模式 | allowlist、mention 语义、房间隔离和权限测试通过 |
| P1 | XMPP MUC mentions 模式 | 稳定/降级身份策略明确，历史/自身消息无回环 |
| P1 | XMPP 安全媒体输入 | SSRF、体积、类型、超时和清理测试通过 |
| P2 | 富文本、引用、反应、发送附件、跨平台账号绑定 | 有独立产品定义、隐私审查和迁移/撤销方案 |

推荐首个生产 canary 只开放显式 allowlist 的 XMPP 单聊与 Matrix 未加密单聊，保留 Telegram 为基线；新平台分别用独立开关逐个放量。Matrix 群聊、XMPP MUC 和任何 E2EE 宣称都应在对应专项完成后单独上线。每一步保留关闭单个平台而不迁移/删除已有记忆的回滚能力。

## 11. 最终判断

当前 adapter 代码为后续接入提供了合理骨架，尤其是稳定平台身份、公共消息处理、防抖、基础访问控制和 Matrix 图片上限。但正式三平台集成的核心工作并不是在 `bot.py` 简单追加两个 coroutine，而是先完成共享运行时所有权、Telegram 解耦、房间级短期上下文隔离、依赖/配置契约和事件幂等。完成这些 P0 后可上线受限单聊；Matrix E2EE、XMPP MUC 和 XMPP 媒体应作为边界清晰的后续里程碑。
