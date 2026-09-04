# 记忆系统与 Chatbot 架构整改报告

日期：2026-09-01

范围：associative-memory 引擎与多平台角色扮演 chatbot。

## 1. 整改结论

本次整改将原来“实验代码、记忆实现、聊天编排、平台传输和运维脚本相互穿插”的结构，整理为两个边界明确的项目：

```text
associative-memory
    负责证据数据、导入、检索、图增长和审计 trace

chatbot
    负责平台、身份、三域隔离、请求编排、角色回答和副作用提交
```

关键结果：

1. 记忆引擎的实验 Stage 代码退出生产包，运行时 API 不再暴露阶段编号。
2. chatbot 的 2,481 行 `memory/service.py` 拆为配置、合同、路由、单域服务和三域系统五个模块。
3. 527 行 adapter 公共文件拆为访问控制、消息工具、共享运行时和防抖调度四个模块。
4. 五个根目录管理脚本合并为一个 `manage.py` 入口和 `src/cli` 子命令包。
5. pytest 只收集正式 `tests/`，不再执行 `_archive` 中的历史测试。
6. 两项目 README 已按当前结构重写，不包含本机绝对路径。
7. `memory_intent` 已进入生产请求：允许读取当前消息和有界近期上下文，生成不可变的跨域检索合同。
8. 正常路由不再包含具体剧情词表；故障兜底只区分空输入、明确闲聊、显式 deep 和保守全域召回。
9. 重构后记忆引擎 334 项测试通过；chatbot 149 项测试通过、13 项按可选依赖跳过。

## 2. 整改前的主要问题

### 2.1 一个文件承担多个变化原因

chatbot 原 `src/memory/service.py` 同时包含：

- 环境变量解析和路径兼容；
- 单库配置和三域配置；
- RetrievalPlan、RetrievalQuality 和结果格式；
- 剧情/私人/公共请求路由；
- 单个数据库的初始化、召回、缓存、导入和增长；
- private/public/knowledge 三域编排。

任何一项改变都要求修改同一个 2,481 行文件，测试也只能从一个巨型模块导入内部函数。

### 2.2 平台公共层成为第二个组合根

原 `adapter_support.py` 同时管理白名单、消息分段、模型初始化、记忆初始化、命令、视觉、消息处理、防抖和事件确认。Adapter 依赖的是一个“万能模块”，而不是明确接口。

### 2.3 实验阶段污染运行时包

`memory_demo.stage5/stage8/stage9/stage10/stage11` 实际是评测、重放和统计工具，却被打包为生产模块。阶段名字表达的是研发历史，不是长期职责。

### 2.4 运维入口分散

导入、查询、统计、清除动态记忆和清除知识库各有一个根目录脚本。用户必须记住多个文件名，README 也重复说明启动方式。

### 2.5 测试范围没有隔离历史资产

chatbot 缺少 pyproject 测试发现配置。直接运行 pytest 会收集 `_archive` 中的旧测试，导致一个缺少异步插件的历史文件使正式基线失败。

## 3. 核心术语

### Source

原始证据分片。模型生成的 Episode、Concept 和 Association 必须能通过 `source_id` 回到 Source。Source 本身不被摘要替换。

### Episode

可独立检索和理解的事件、状态或事实。它不是固定长度段落，而是语义单元。Episode 保存 float32 embedding、参与者、时间语义、证据状态和 generation。

### Paragraph

可选的局部召回层。Paragraph 对原文片段生成 embedding，用于提高 Episode 候选召回率，但不承担最终事实语义，也不替代 Source。

### Concept

自然语言中可被指称的实体或抽象对象，包括人物、组织、物品、情绪、创伤、关系和主题。Concept 可以拥有多语言别名。

### Association

Episode 或 Concept 之间的有向联系。关系文本保存“为什么有关联”；weight 表示联想方向的强弱，不等同于严格因果概率。

### generation

推论到直接经验的距离。`generation=0` 表示直接材料或不依赖推论的结构；若新推论把已有推论作为前提，generation 增大。generation 与可信度是不同维度。

### RetrievalPlan

一次请求不可变的检索策略，包括查询规划方式、图跳数、候选预算、reranker、证据槽、后续检索条件和时间预算。它替代修改共享全局配置的选档方式。

### RetrievalQuality

检索是否足够回答问题的结构化判断。它综合实体覆盖、事实槽覆盖、最高相关分、分差、来源多样性、证据数量和时间线冲突。相关性高不等于证据完整。

### Coverage Selector

reranker 后的覆盖选择器。它不是简单取 Top-N，而是尽量保证不同答案槽、实体和事件都获得证据。

### 记忆域

- `private`：按平台稳定用户 ID 隔离的对话、经历、偏好、印象和共同创造事件。
- `public`：所有用户共享、但不属于外部文档知识库的长期记忆。
- `knowledge`：导入的剧情、百科、文档和经审计推论。

## 4. 整改后的文件职责

### 4.1 记忆引擎

```text
memory_demo.__init__       对上层公开 AppConfig/Database/MemoryApplication
memory_demo.contracts     请求和回答合同
memory_demo.app           引擎组合根
memory_demo.adapters      输入文件兼容
memory_demo.ingestion     分片、提取、审计和事务导入
memory_demo.ingestion.ordering  通用目录时间线分组和自然排序
memory_demo.retrieval     候选召回、重排、覆盖和图检索
memory_demo.retrieval.context  Source 证据摘录
memory_demo.retrieval.query_planning  结构化证据槽转换
memory_demo.associations  建边、遍历和增长
memory_demo.repositories  SQLite 表级访问
memory_demo.embeddings    float32 编码和 RAM 索引
benchmarks.support        评测共用实现
```

chatbot 现在通过 `from memory_demo import AppConfig, MemoryApplication` 使用公共边界，不再从 `memory_demo.app` 和 `memory_demo.config` 两个内部路径分别取对象。

### 4.2 Chatbot 记忆层

```text
src/memory/service.py    公共 facade，不含业务实现
src/memory/config.py     环境、路径、MemoryServiceConfig、MemorySystemConfig
src/memory/contracts.py  RetrievedMemory、RetrievalPlan、RetrievalQuality
src/memory/intent_planner.py  memory_intent 输出解析与单域合同编译
src/memory/routing.py    无领域词表的故障兜底
src/memory/domain.py     一个物理 SQLite 域的生命周期和召回
src/memory/system.py     private/public/knowledge 三域编排
```

`AssociativeMemoryService` 只知道一个数据库。`MemorySystem` 决定查询哪些域，并为用户创建 `platform/platform_user_id` 隔离目录。

### 4.3 Chatbot 平台层

```text
src/bot/access.py           白名单开关和稳定 ID 列表
src/bot/messages.py         平台消息分段
src/bot/runtime.py          进程级共享依赖和单条传输处理
src/bot/dispatch.py         防抖、同会话锁、事件 claim/complete/release
src/bot/request_planning.py 有界上下文、模型规划和 fallback
src/bot/chat_service.py     回答生成、审计与延迟提交
src/bot/adapter_support.py  adapter 使用的公共 facade
src/bot/*_adapter.py        平台协议转换
```

Adapter 只负责把平台事件转换为 `PlatformIdentity + conversation_key + text/images`，不包含记忆检索策略。

### 4.4 运维层

```text
manage.py
└── src/cli/
    ├── import_data.py
    ├── query.py
    ├── stats.py
    ├── clear_dynamic.py
    └── clear_knowledge.py
```

根目录只保留 `bot.py` 和 `manage.py` 两个入口。

## 5. 当前用户提问处理流程

### 第 1 步：平台接收

Adapter 从 Telegram、Discord、Matrix 等平台事件中取得：

- 平台名；
- 平台稳定用户 ID；
- 显示名；
- 会话/房间稳定 ID；
- 事件 ID；
- 文本与图片。

显示名不参与私人记忆隔离。`PlatformIdentity.key` 由平台名和稳定用户 ID 形成。

### 第 2 步：访问控制和事件去重

若该平台白名单开关开启，系统按稳定 ID 检查。`PersistentEventDeduplicator` 对入站事件执行 claim；重复 webhook 不会再次生成回复或写入记忆。

### 第 3 步：防抖与会话串行化

`DebouncedDispatcher` 在短时间内合并同一会话的连续消息。不同会话可以并行；同一会话使用独立 asyncio Lock，保证回答和记忆提交顺序一致。

### 第 4 步：命令和附件预处理

`AdapterRuntime` 优先处理 `/start`、`/identity`、`/clear`、`/memory_status` 和 `/model`。图片由 vision engine 生成文本描述，再与用户文本组合。

### 第 5 步：模型请求规划

`ChatRequestPlanner` 把当前用户消息和有限近期上下文交给 `memory_intent`。近期上下文同时受消息条数与字符数限制，当前消息单独保留为主输入。模型输出：

- 是否需要长期记忆；
- private/public/knowledge 域选择；
- 每个域不同的自包含 query；
- light/standard/deep；
- 是否为创造性互动；
- target entities；
- relation、temporal、causal 约束；
- answer slots、uncertainty 和 knowledge 写入策略。

模型规划被编译成每域不可变的 `DomainRecallRequest`。正常路径不再通过关键词猜域，也不会识别某部作品的角色、地点或章节名。若模型禁用、超时或返回无效 JSON，fallback 对非闲聊请求保守查询三个域，并依靠证据选择排除无关结果。

### 第 6 步：三域并行召回

`MemorySystem.recall` 为选中的每个域建立调用：

- private 调用当前稳定身份对应的数据库；
- public 调用共享公共库；
- knowledge 调用外部文档知识库。

不同域的召回通过 `asyncio.gather` 并发。各域输出保持独立，最终 prompt 中带域标题，避免私人记忆和剧情证据被混淆。

### 第 7 步：单域检索

`AssociativeMemoryService` 根据请求级 `RetrievalPlan + DomainRecallRequest` 创建 query-local QueryEngine：

1. embedding、稀疏、Concept、Paragraph 召回；
2. Source cohort 和 Association 图扩展；
3. 可选 BGE cross-encoder 或 LLM rerank；
4. Coverage Selector 选择多事实证据；
5. RetrievalQuality 检查实体、槽位、来源和时间线；
6. standard 不足时复用原语义合同自动升级 deep；
7. 返回证据、关系路径、质量、rerank trace 和时延。

请求配置是 deepcopy 后的局部配置，不修改共享 app config；只读前台查询可以并行。导入和增长仍使用写锁。

### 第 8 步：构造角色回答

Coordinator 将按域标注的证据写入动态 system prompt。light 请求使用 `chat_fast`，standard/deep 使用主聊天模型。用户级 `/model` 配置只覆盖聊天生成。

创造性请求可以使用知识库中的人物和世界设定作为约束，但新生成的“当前任务、消息、行程”不被当成既有剧情事实。

### 第 9 步：回答审计

模型可以在隐藏的 `<assistant_memory>` 区域返回候选命题：

- private candidate：本用户的事件、偏好或共同创造内容；
- knowledge candidate：由本轮证据支持、可能值得进入知识图的新推论。

`PrivateMemoryResponseGuard` 检查回答是否伪造“我记得的用户事实”。若审计失败则动态重写；被替换的草稿不能产生持久化候选。

### 第 10 步：先发送，再提交副作用

Adapter 先发送可见回复。发送成功后调用 `ChatReply.finalize()`：

1. 记录会话；
2. 达到阈值后将会话批次放入私人记忆导入队列；
3. 将证据绑定的增长候选放入后台队列；
4. 更新进程内近期会话。

如果发送失败，本轮不会被当成已完成对话提交。事件 claim 会 release，使平台可以安全重试。

## 6. 导入流程的责任边界

```text
manage.py import
→ src/cli/import_data.py 处理参数、断点和文件级状态
→ MemorySystem 选择 knowledge/public/private writer
→ AssociativeMemoryService 获取写锁
→ MemoryApplication.import_path
→ InputAdapter/Segmenter/Extractor/Resolver/AssociationBuilder
→ SQLite 事务
→ RAM index refresh
```

Embedding 模型不允许 fallback。推理模型可以重试或使用配置的 fallback。目录断点文件会记录 completed、partial、failed、interrupted 和 changed；含未解决记录的文件不会被盲目重复导入。

## 7. 并发模型

- 平台级：不同 adapter 在同一 TaskGroup 中运行。
- 全局请求级：`BOT_MAX_CONCURRENCY` semaphore 限制并发模型请求。
- 会话级：同一 conversation key 串行，不同会话并行。
- 记忆域级：private/public/knowledge 并行召回。
- 单域前台：query-local engine，无共享配置切换，可并行读。
- 单域写入：导入、候选审计和图刷新使用 operation lock。
- 后台：会话导入和 Association 增长使用持久化队列，崩溃后可恢复。

## 8. 删除与归档

- 原 `memory_demo.stage5/8/9/10/11` 已从生产包删除并迁至 `benchmarks/support`。
- 30 个按 Stage 编号命名、且不再被正式回归使用的历史 benchmark 驱动已迁入 `_archive/benchmark-stage-history-20260901`；三个仍被测试调用的评分器以职责名迁入 `benchmarks/support`。
- chatbot 根目录的 `clear_memory.py`、`clear_knowledge.py`、`db_stats.py`、`import_knowledge.py`、`memory_query.py` 已迁入 `src/cli`，原路径不再保留。
- 可再生 `__pycache__` 与 `.pytest_cache` 已清理；测试后可能重新生成。
- 重构前关键源码位于两项目各自的 `_archive/refactor-20260901/pre-refactor`。

缓存不可恢复但可自动再生；迁移的历史脚本和旧源码可从 `_archive` 恢复。

## 9. 未完成的技术债

### 9.1 记忆引擎内部仍有三个超大算法文件

- `retrieval/engine.py` 约 5,350 行；
- `ingestion/extractor.py` 约 2,900 行；
- `ingestion/pipeline.py` 约 2,100 行。

本轮已先迁出无状态、低耦合部分：Source 摘录、结构化查询槽、follow-up 判定、目录时间线分组和自然排序。剩余大文件共享候选 trace、配置快照和审计中间态；下一步应先定义 `CandidateSet`、`EvidenceSelection`、`ExtractionBatch`、`ImportOutcome`，再按 recaller/reranker/selector/growth 和 parser/auditor/persister 拆分。

### 9.2 memory_intent 需要真实流量校准

`memory_intent` 已按用户授权接收当前消息和有限上下文，且已有 JSON 解析、上下文边界、模型失败 fallback、跨域查询和请求缓存测试。尚缺真实角色扮演流量上的域选择准确率、规划耗时、错误升级率和 token 消耗分布。应记录匿名化 planning trace，建立以“正确证据域和答案槽”为目标的评测集，而不是重新添加表达枚举。

### 9.3 请求回答合同尚未成为在线主路径

引擎的 `RequestAnswerContract` 已通过离线实验，但 chatbot 在线回答仍使用动态 prompt + memory guard。后续应增加 claim ID、action ID 和结构化 persona 称呼字段，再决定是否接入在线路径。

### 9.4 RetrievalQuality 仍是第一版确定性判断

当前质量判断能发现空证据、实体缺失、槽位缺失、低约束相关性和时间线冲突，但对隐含槽位和关系方向的判断仍有限。需要用多领域固定集校准阈值和误升级率。

## 10. 后续建议顺序

1. 建立 `memory_intent` 固定集，测域选择、查询改写、答案槽、creative 和强度准确率。
2. 采集 planning/recall/rerank/chat 各阶段耗时，校准 light 的五秒目标和 planner 成本。
3. 为 QueryEngine 定义类型化阶段合同，再拆 recall/rerank/coverage/growth。
4. 为导入定义 ExtractionBatch 和 ImportOutcome，再拆提取、审计、持久化。
5. 将 RequestAnswerContract 以影子模式接入线上日志，先比较不改变回答的审计结果。
6. 建立跨私人、公共、知识域的固定验收集，持续测召回、覆盖、错误域写入和端到端时延。

## 11. 验证记录

```text
associative-memory: 334 passed
chatbot:            149 passed, 13 skipped
```

13 个 skip 对应未安装或未启用的可选平台依赖，不是重构失败。唯一持续 warning 来自第三方 `google.genai` 对 Python 3.14 内部类型的弃用提示。
