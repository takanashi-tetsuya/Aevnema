# Developer Manual

本手册说明 chatbot 的长期维护边界。安装、启动和管理命令见 README。

## 1. 依赖方向

只允许以下方向：

```text
platform adapter
    ↓
bot facade / AdapterRuntime / ConversationCoordinator
    ↓
MemorySystem public API + LLM Engine
    ↓
associative-memory public API
```

禁止反向依赖：

- `memory_demo` 不导入 chatbot。
- `src/memory` 不导入具体平台 adapter。
- adapter 不直接操作 SQLite、repository、prompt 或增长队列。
- benchmarks/experiments 不提供生产运行时实现。

## 2. 组合根

### `bot.py`

只负责：

1. 读取已启用 adapter；
2. 执行启动前校验；
3. 获取进程锁；
4. 创建一个 `AdapterRuntime`；
5. 在 TaskGroup 中启动 adapter；
6. 关闭共享运行时。

不得在 `bot.py` 加入平台特例、记忆检索或 prompt。

### `src/bot/runtime.py`

`AdapterRuntime` 是唯一的进程级服务容器，持有：

- MemorySystem；
- ConversationCoordinator；
- 会话 journal 和导入 worker；
- Association growth worker；
- 用户模型设置；
- vision engine；
- 全局并发 semaphore；
- 持久化事件去重器。

所有 adapter 必须复用同一个 runtime。只有独立运行单个 adapter 的开发入口可以临时创建 runtime，并负责关闭。

### 记忆引擎 `MemoryApplication`

这是数据库、repositories、RAM indexes、ModelClient、ImportPipeline 和 QueryEngine 的组合根。chatbot 只能从 `memory_demo` 公共包导入它。

## 3. 单条请求合同

平台层向共享运行时提供：

```python
PlatformIdentity(
    platform="telegram",
    platform_user_id="稳定且平台内唯一的 ID",
    display_name="仅展示",
)
```

并额外提供：

- `conversation_key`：平台 + 会话/房间稳定 ID；
- `event_id`：用于持久化去重；
- `text`；
- `images`；
- `send_text`；
- 可选 `set_typing`。

`display_name` 永远不能成为数据库隔离键。

`AdapterRuntime.process_message()` 返回 bool：

- `True`：回复已发送，可以 complete 事件 claim；
- `False`：未完成，必须 release claim，允许重试。

## 4. Adapter 开发

新增平台需要：

1. 新建 `src/bot/<platform>_adapter.py`。
2. 在 `adapter_registry.py` 注册 `AdapterSpec`。
3. 在 `.env.example` 增加 `ENABLE_<PLATFORM>`、凭据和白名单变量。
4. 在合适的 requirements 分组加入可选依赖。
5. 把平台事件转换为稳定身份、会话键和事件 ID。
6. 使用 `DebouncedDispatcher.submit()`，不自行实现会话锁。
7. 使用 `split_message()` 处理平台长度限制。
8. 使用 `require_access_policy()` 做启动前白名单校验。
9. 增加纯 fake 平台测试；正式测试不能访问真实网络。

Adapter 中可以包含：

- SDK 生命周期；
- webhook 验签；
- 平台事件解析；
- 群聊触发方式；
- 平台发送和 typing API。

Adapter 中不能包含：

- 记忆域选择；
- 角色 prompt；
- LLM 模型选择；
- 数据库路径；
- Association 写入；
- 用户显示名到身份的自定义映射。

## 5. Chat 编排

`ConversationCoordinator.handle()` 是一轮对话的应用事务：

```text
memory_intent(current message + bounded context)
→ compile DomainRecallRequest per selected domain
→ recall memory domains
→ build dynamic roleplay prompt
→ select fast/main model
→ generate reply
→ split hidden memory candidates
→ private-memory guard
→ return ChatReply with deferred finalizer
```

任何持久化副作用放进 `ChatReply.finalize()`。传输层必须先成功发送回复，再 finalize。

### 近期上下文

- `memory_intent` 可接收当前消息和近期上下文。
- 外发上下文同时受 `MEMORY_INTENT_CONTEXT_MESSAGES` 与 `MEMORY_INTENT_CONTEXT_CHARS` 限制。
- 当前消息独立放在 planning payload 中，不占近期上下文预算。
- 上下文只用于指代消解；模型必须按证据原始来源选择 private/public/knowledge。
- 若模型调用失败，规则层只提供保守全域 fallback，不进行语义分类。

### 生成模型

- light 使用 `chat_fast`。
- standard/deep 使用 `chat`。
- 用户设置只覆盖聊天任务。
- 内部 extract/audit/rewrite/embedding 配置不接受普通用户覆盖。

## 6. 记忆层

### 公共 facade

业务代码从 `src.memory` 或 `src.memory.service` 导入公共类型。模块内部依赖应直接指向职责模块：

```text
config.py     配置
contracts.py  计划、质量和结果
intent_planner.py  模型语义规划
routing.py    无语义假装的故障兜底
domain.py     单个数据库
system.py     三域编排
```

测试内部函数时，从其真实模块导入，不再把私有函数塞回 `service.py` facade。

### 域隔离

```text
knowledge -> 一个共享知识数据库
public    -> 一个共享公共记忆数据库
private   -> data/users/<platform>/<storage_key>/memory.db
```

普通对话不能直接写 public。剧情推论候选只能写 knowledge；创造性新事件只能进入当前 private。

### 请求级检索

`RetrievalPlan` 必须是请求局部值。禁止为了 light/standard/deep 修改共享 `app_config` 后再恢复。单域服务用配置 deepcopy 创建 query-local engine。

`DomainRecallRequest` 保存单域 query、结构化 intent 和可选 follow-up。它可缓存，并在 standard 证据不足升级 deep 时原样复用。直接传 `intent_override` 只保留给诊断测试，不属于在线主路径。

禁止在 `routing.py`、adapter 或 Coordinator 中加入具体作品、人物、地点词表来模拟语义理解。新的表达覆盖问题应通过 `memory_intent` prompt、模型评测集和结构化合同解决。

`RetrievalQuality` 决定 standard 是否升级 deep。reranker 的高分不能替代事实槽覆盖。

### 写操作

以下操作使用单域 operation lock：

- import；
- candidate audit；
- growth；
- index refresh；
- association cue rebuild。

普通前台查询在 growth 关闭且配置请求局部时不使用大锁。

## 7. Prompt 与模型

所有 prompt 必须位于 `config/prompt_config`。业务模块只能导入常量或 render 函数，不允许内联长 prompt。

模型配置只放在 `config/model_config.toml`：

- provider 和 base URL；
- 模型顺序；
- retry；
- token；
- thinking；
- sampling；
- response format。

Embedding 是同一向量空间的硬合同，不能 fallback。其他任务允许按任务配置 fallback。

## 8. 会话记忆和增长

`ConversationJournal` 先将已完成对话写入可恢复文件。达到批次阈值后，`ConversationIngestionWorker` 调用 private import。用户身份 metadata 与批次同目录保存。

`BackgroundGrowthWorker` 接收回答后的候选，而不是重新猜测可见回答。知识候选必须绑定本轮可见 Episode IDs；私人候选必须属于当前身份。

队列任务需要：

- 稳定 job ID；
- identity；
- route；
- question/query；
- evidence 摘要；
- consolidation decision；
- 状态和错误。

## 9. 管理入口

所有管理命令从 `manage.py` 进入：

```text
import
query
stats
clear-dynamic
clear-knowledge
```

子命令实现在 `src/cli`。新增管理功能时增加子模块和 `COMMANDS` 映射，不在根目录增加新的脚本。

清理命令必须：

- 默认 dry-run；
- 验证目标绝对路径；
- 拒绝项目根、用户目录和文件系统根；
- 检查 bot 进程锁；
- 对 SQLite 采用新空库原子替换；
- 清理 WAL/SHM；
- 重新验证业务表为空。

## 10. 测试

```bash
python -m pytest -q
```

正式基线只收集 `tests/`。网络、真实平台和付费模型调用不得出现在默认测试中。

测试层次：

1. 纯函数与合同；
2. 单域 service fake QueryEngine；
3. 三域 MemorySystem fake services；
4. ConversationCoordinator fake memory/model；
5. Adapter fake SDK/runtime；
6. 独立 experiments 中的真实模型验收。

## 11. 变更检查清单

- 是否改变平台稳定身份或会话键？
- 是否可能把 private 写入 public/knowledge？
- 是否把助手虚构内容当成既有证据？
- 是否修改共享检索配置？
- 是否绕过 RetrievalQuality 直接结束 standard？
- 是否在代码中加入具体剧情结论？
- 是否把 prompt 写回业务模块？
- 是否给 embedding 增加 fallback？
- 是否把副作用放在发送成功之前？
- 是否补充了测试和 README/报告？

## 12. 回退

2026-09-01 重构前的关键源码保存在：

```text
_archive/refactor-20260901/pre-refactor
```

该目录只用于人工比较和回退，不会被 pytest 或运行时自动加载。
