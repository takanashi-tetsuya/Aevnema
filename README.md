# Associative Roleplay Chatbot

多平台角色扮演机器人。聊天平台、用户身份、会话编排和最终回复由本项目负责；Source/Episode/Concept/Association 的导入、检索和增长由独立的 associative-memory 引擎负责。

## 系统边界

```text
平台事件
→ Adapter
→ AdapterRuntime
→ ConversationCoordinator
→ MemorySystem
   ├── private：平台 + 稳定用户 ID 隔离
   ├── public：所有用户共享
   └── knowledge：导入的剧情/文档知识
→ LLM 回复
→ 记忆审计与后台增长
→ Adapter 发送
```

聊天机器人不直接访问记忆引擎的 SQLite repository。两项目的唯一运行时边界是 `MemoryApplication` 和请求级检索合同。

## 目录

```text
bot.py                    机器人进程入口
manage.py                 导入、查询、统计和清理的统一管理入口
config/
├── model_config.toml     模型、fallback、thinking、采样和 token 配置
├── persona.toml          角色身份、关系和表达风格
└── prompt_config/        所有 prompt
src/
├── bot/
│   ├── adapter_registry.py  平台注册与启动开关
│   ├── adapter_support.py   adapter 公共 API facade
│   ├── access.py            白名单与稳定 ID 访问策略
│   ├── runtime.py           进程级共享服务
│   ├── dispatch.py          防抖、同会话串行化和事件确认
│   ├── messages.py          平台消息分段
│   ├── request_planning.py  上下文边界与模型记忆规划
│   ├── chat_service.py      单条用户请求的应用编排
│   ├── memory_guard.py      私人记忆回答审计
│   └── *_adapter.py         各平台传输实现
├── memory/
│   ├── service.py           记忆层公共 API facade
│   ├── config.py            环境和三域配置
│   ├── contracts.py         RetrievalPlan/Quality/Result
│   ├── intent_planner.py    memory_intent 结构化语义计划
│   ├── routing.py           无领域知识的故障兜底
│   ├── domain.py            单个物理数据库的异步服务
│   ├── system.py            private/public/knowledge 编排
│   ├── conversation.py      会话日志与批量私人记忆导入
│   ├── growth.py            后台 Association 增长队列
│   └── answer_consolidator.py 回答后候选命题分流
├── llm/                    统一模型网关和用户级聊天设置
├── cli/                    manage.py 的子命令实现
└── utils/                  日志等通用工具
tests/                      正式回归测试
experiments/                不进入运行时的实验
docs/reports/               架构与实验报告
data/                       数据库、队列、会话批次和进程锁
logs/                       运行日志
_archive/                   历史实现和重构回退资产
```

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell 激活命令为：

```powershell
.\.venv\Scripts\Activate.ps1
```

按需安装平台依赖：

```bash
python -m pip install -r requirements-all-adapters.txt
```

也可以只安装一个平台分组：

```text
requirements-adapters-asia.txt
requirements-adapters-federated.txt
requirements-adapters-webhooks.txt
requirements-adapters-western.txt
```

## 配置

```bash
cp .env.example .env
```

基本配置：

```dotenv
SILICONFLOW_API_KEY=
MEMORY_ENGINE_ROOT=../associative-memory
KNOWLEDGE_MEMORY_DB_PATH=data/knowledge/knowledge.db
PUBLIC_MEMORY_DB_PATH=data/public/memory.db
USER_MEMORY_DB_DIR=data/users
MEMORY_INTENT_ENABLED=true
MEMORY_INTENT_CONTEXT_MESSAGES=6
MEMORY_INTENT_CONTEXT_CHARS=4000

ENABLE_TELEGRAM=true
TELEGRAM_BOT_TOKEN=
TELEGRAM_WHITELIST_ENABLED=false
```

所有 adapter 都有独立的 `ENABLE_<PLATFORM>` 开关，默认关闭。白名单由 `<PLATFORM>_WHITELIST_ENABLED` 单独控制，默认关闭；开启后必须填写该平台支持的稳定用户、会话或空间 ID 列表。

模型设置位于 `config/model_config.toml`：

- `chat`：主回复模型。
- `chat_fast`：light 请求使用的低延迟模型。
- `extract`：结构化提取。
- `summary`：长期摘要。
- `memory_audit` / `memory_rewrite`：私人记忆安全审计和重写。
- `memory_consolidation`：回答后的命题候选分流。
- `memory_intent`：根据当前消息和有限近期上下文生成跨域检索计划。
- `vision`：图片理解。
- `embedding`：固定 embedding 模型；不允许 fallback。

每个任务可以独立设置 `enable_thinking`、`max_tokens`、`temperature` 和模型 fallback 顺序。普通用户的 `/model` 只修改自己的聊天生成参数，不会改动内部提取或审计任务。

## 仓库与运行数据

`.env`、SQLite 数据库、用户记忆、导入断点、队列和运行日志均属于本机运行数据，已由 `.gitignore` 排除，不应提交到仓库。仓库只保存源代码、配置模板、测试和文档；需要迁移数据库时请单独备份对应文件。`.env.example` 只包含配置项示例，不包含实际凭据。

## 启动

```bash
python bot.py
```

启动过程会：

1. 读取 `.env`。
2. 校验已启用 adapter 的凭据和白名单。
3. 获取进程锁。
4. 加载 knowledge/public 索引。
5. 启动会话导入、Association 增长和事件去重服务。
6. 启动所有已启用 adapter。

Telegram 命令：

```text
/start
/identity
/clear
/memory_status
/model
```

`/clear` 只清除当前进程中的近期会话上下文，不删除长期记忆。

## 管理命令

```bash
python manage.py --help
```

导入剧情或文档：

```bash
python manage.py import ./documents/story --domain knowledge
```

导入公共材料：

```bash
python manage.py import ./documents/public.txt --domain public
```

导入指定用户的私人材料：

```bash
python manage.py import ./documents/notes.txt --domain private \
  --platform telegram --user-id 123456789
```

带目录级断点导入：

```bash
python manage.py import ./documents/story --domain knowledge \
  --progress-file data/import-progress/story.json
```

查询证据 trace：

```bash
python manage.py query "问题" --domain knowledge --mode fast
```

统计：

```bash
python manage.py stats
python manage.py stats --platform telegram --user-id 123456789
```

清理命令默认为 dry-run：

```bash
python manage.py clear-dynamic
python manage.py clear-knowledge
```

确认目标无误、机器人已经停止后执行：

```bash
python manage.py clear-dynamic --yes
python manage.py clear-knowledge --yes
```

`clear-dynamic` 清除公共记忆、所有私人记忆、待导入对话和增长队列。`clear-knowledge` 重新建立空的知识库 schema。模型设置和诊断日志不属于记忆，不会被这两个命令删除。

## 用户问题处理流程

1. Adapter 从平台事件中提取稳定用户 ID、会话 ID、文本和附件。
2. 白名单按稳定 ID 检查；持久化事件去重阻止 webhook 重放。
3. `DebouncedDispatcher` 合并同一会话的连续消息，并保证同一会话串行处理。
4. `AdapterRuntime` 处理命令和图片，将普通文本交给 `ConversationCoordinator`。
5. `memory_intent` 接收当前消息和受字符数、消息数双重限制的近期上下文，输出域、强度、创造性、实体、关系、时间/因果约束、答案槽和各域独立查询。
6. `ChatRequestPlanner` 将模型结果编译成每域不可变的 `DomainRecallRequest`。模型不可用时，兜底不猜语义，除明确闲聊外保守查询三个域；兜底不包含任何剧情专名规则。
7. `MemorySystem` 并发查询被选中的物理数据库，私人库路径由 `platform + platform_user_id` 决定。
8. 每个域按请求级 `RetrievalPlan + DomainRecallRequest` 召回、重排并计算 `RetrievalQuality`；standard 证据不足时复用同一语义计划升级 deep。
9. 检索证据按域标注后注入角色 prompt；light 使用 `chat_fast`，其他请求使用主聊天模型。
10. 回答中的候选知识和私人事件被解析；私人记忆 guard 会审计未经证据支持的“我记得”。
11. Adapter 发送最终可见回复。只有发送成功后才 finalize 本轮副作用。
12. 用户与助手文本写入会话日志，达到批量阈值后导入该用户私人库。
13. 经证据绑定的知识候选进入后台增长队列；公共库只通过显式管理导入写入。

## 测试

```bash
python -m pytest -q
```

测试发现范围固定为 `tests/`。`_archive`、虚拟环境、data、logs 和 experiments 不会被 pytest 自动收集。

## 维护约束

- 平台适配器只做传输，不复制记忆或聊天业务逻辑。
- 用户隔离只使用平台稳定 ID，不使用显示名。
- knowledge、public、private 必须使用独立 SQLite 数据库。
- 角色 prompt、检索证据和运行回执不得混成一段不可审计文本。
- 新创造的角色扮演事件默认进入当前用户私人记忆，不能直接污染知识库。
- 对知识库的新推论必须保留前提 Episode、generation、证据状态和审计 trace。
- embedding 模型失败必须失败，不可静默切换不同 embedding 空间。
- 正常域选择和查询改写只能由 `memory_intent` 完成；`routing.py` 不得加入具体作品、人物或语料库词表。
