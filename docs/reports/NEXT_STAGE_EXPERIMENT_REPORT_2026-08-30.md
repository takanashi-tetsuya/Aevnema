# 下一阶段实验报告：模型参数控制、虚拟多账号与大型知识检索

日期：2026-08-30

## 1. 本阶段目标

本阶段同时完成三项工作：

1. 重构 `model_config.toml`，消除重复配置，并允许安全、可校验的底层参数覆盖。
2. 在只有一个真实 Telegram 账号的条件下，用虚拟平台身份验证多人隔离、公共空间共享和改名/重启连续性。
3. 延续上一阶段实验，对大型剧情知识库运行多次真实冷路径基准，确定当前主要延迟来源。

## 2. 模型配置重构

### 2.1 三层继承

配置现在按以下顺序合并：

```text
[defaults] 全局默认
  ↓
[task_defaults.<task>] 任务默认
  ↓
[[engines.<task>]] 单模型覆盖
```

全局层只需写一次 provider、base URL、API key 环境变量名和超时。每个任务设置自己的温度与输出预算，
单模型条目只保留模型名及真正不同的参数。fallback 顺序仍由条目书写顺序决定。

支持的参数包括：

- `timeout_seconds`、`max_retries`、`retry_delay`；
- `temperature`、`max_tokens`、`top_p`、`top_k`、`seed`；
- `presence_penalty`、`frequency_penalty`、`stop`；
- `response_format`；
- `enable_thinking`、`thinking_budget`；
- 任意服务商专用 `extra_body`。

所有逐请求覆盖都经过 Pydantic 范围校验。`LLMEngine.describe()` 可以输出最终有效配置，但不会输出 API key。

### 2.2 按用户选择角色模型参数

Telegram 新增 `/model` 命令：

```text
/model
/model thinking inherit|auto|on|off
/model thinking_budget 1024
/model temperature 0.4
/model top_p 0.9
/model max_tokens 1200
/model seed 42
/model reset
```

单项使用 `default` 可清除覆盖。四种 thinking 状态含义不同：

- `inherit`：使用 TOML 合并结果；
- `auto`：显式不发送 `enable_thinking`，由服务商决定；
- `on`：强制开启；
- `off`：强制关闭。

设置原子写入 `USER_MODEL_SETTINGS_PATH`，主键为 `platform:platform_user_id`。它只影响角色聊天请求；
提取、私人事实审计和违规重写不接受用户覆盖，因此普通用户不能通过关闭 thinking 等操作削弱事实边界。

## 3. 虚拟多账号方法

新增 `experiments/virtual_multi_account_acceptance.py`。它没有伪造 Telegram 服务端，而是直接调用生产环境中位于
Telegram 适配器下方的统一链路：

```text
PlatformIdentity
  → ConversationCoordinator
  → MemorySystem
  → chat model
  → private-memory guard
```

虚拟身份包括：

- `telegram:1001`，显示名“老师”；
- `telegram:1002`，显示名同样为“老师”；
- `discord:1001`，数字 ID 与 Telegram A 相同；
- `matrix:1003`，完全未知用户；
- 重启后的 `telegram:1001`，显示名改为“改名后的老师”。

私人 A 的暗号为“琥珀月亮”，私人 B 的暗号为“银色风铃”；公共空间事实为“青空灯塔”。A 强制关闭
thinking，B 强制开启 thinking 并设置 1024 预算，其他身份继承配置。

## 4. 第一轮失败与修复

第一轮并没有隐藏失败，报告保存在：

`logs/virtual-multi-account/20260830T042657.542940Z/report.json`

已经通过的部分：私库 source_key 分离、同数字跨平台隔离、未知身份隔离、改名重启连续性、模型设置隔离。

失败原因有两个：

1. 测试材料写成第三人称“Telegram 用户 1001 告诉阿洛娜……”。提取器据此把 Episode 标为
   `reported`，角色不应把它直接等同于当前用户亲口陈述。测试材料与真实聊天语义不一致。
2. 私人审计器只收到私人 evidence。回答同时使用正确公共事实“青空灯塔”时，部分审计调用误认为它是
   私人契约外的新事实。

修复：

- 私人测试资料改用生产对话相同的 `conversation_record`，明确 `role=user`、平台和原生 ID；
- `PrivateMemoryContract` 增加 `non_private_evidence`，携带本轮公共/剧情证据；这些证据只允许普通事实，
  不能支持私人暗号、偏好或共同经历。

这不是放宽私人审计，而是把“私人事实边界”和“有证据的非私人事实”正确分开。

## 5. 第二轮虚拟多账号结果

最终报告：

`logs/virtual-multi-account/20260830T043147.422233Z/report.json`

9 项检查全部通过：

| 检查 | 结果 |
|---|---|
| 同平台不同 ID 私人隔离 | 通过 |
| 相同显示名隔离 | 通过 |
| 相同数字 ID、不同平台隔离 | 通过 |
| 未知身份无私人命中 | 通过 |
| 显示名变化与重启连续性 | 通过 |
| 公共记忆对全部身份共享 | 通过 |
| 底层 source_key/物理库隔离 | 通过 |
| 用户模型参数隔离 | 通过 |
| 无内部字段泄漏 | 通过 |

三份资料并行导入耗时 30.396 秒。小型数据库的本地检索约 0.67–0.72 秒；完整回合还包括角色生成和
强事实审计。

本轮 A（thinking off）总耗时 13.475 秒，B（thinking on）为 21.396 秒。两者都正确回答各自私人暗号
和公共通行语。因为它们的证据文本不同且每种模式只有一次，这只能证明开关真实可用，不能证明 thinking
稳定增加 7.9 秒。需要同一提示、随机交错、至少 20 次/模式后才能评价性能和质量差异。

## 6. 大型剧情知识库冷路径实验

使用 `experiments/manifests/foreground_benchmark_v1.json` 的三道多跳问题，每题运行 3 次，关闭完全相同问题的前台 LRU，
共 9 次真实模型路径。

报告：

`logs/foreground-benchmark/20260830T044236.550995Z/report.json`

正确性：

- 9/9 运行通过；
- 27/27 必需事实槽命中；
- Fact-slot recall = 100%。

总耗时：

| 指标 | 秒 |
|---|---:|
| 平均 | 43.258 |
| P50 | 47.515 |
| P90 | 66.622 |
| 最小 | 21.290 |
| 最大 | 66.622 |

按问题：

| 问题 | 平均 | P50 | 最大 |
|---|---:|---:|---:|
| 补习部表象与真相 | 22.022 | 21.644 | 23.133 |
| 乐园悖论与信任危机 | 51.483 | 52.451 | 54.483 |
| 古圣堂袭击因果链 | 56.269 | 56.689 | 66.622 |

关键阶段：

| 阶段 | 平均 | P50 | P90 | 占总管线 |
|---|---:|---:|---:|---:|
| evidence rerank | 24.743 | 25.047 | 40.388 | 57.2% |
| intent parse | 11.785 | 10.870 | 25.577 | 27.2% |
| follow-up planning | 2.998 | 0 | 10.032 | 6.9% |
| initial retrieval | 1.098 | 1.013 | 2.337 | 2.5% |
| initial embedding | 0.807 | 0.857 | 0.986 | 1.9% |
| initial graph expansion | 0.392 | 0.382 | 0.481 | 0.9% |

结论非常明确：float32 向量、本地 FTS 和图遍历不是当前主要延迟。约 84.4% 时间消耗在 intent parse 与
evidence rerank 两个模型阶段；古圣堂题还稳定触发约 8–10 秒 follow-up planning。三题每次都向 reranker
提交 64 个 Episode，而本地预处理通常不到 0.1 秒。

## 7. 下一轮优化方向

推荐按以下顺序实验，而不是直接更换向量索引：

1. **Rerank 输入消融**：分别使用 48、32、24 个 Episode 重跑固定事实槽基准。只有保持 27/27 后才接受
   更小输入；比较的是 recall 与 P50/P90，不用增长边数量评分。
2. **Intent 规划压缩**：保留结构化查询能力，但缩短提示词、限制输出槽并测试 Qwen3.5-9B thinking off
   作为规划器；先要求固定题零事实槽回退。
3. **Follow-up 条件收紧**：补习部和悖论题 P50 follow-up 为 0，古圣堂题才持续触发。应只优化该触发器，
   不应全局关闭多跳补查。
4. **同提示 thinking A/B**：相同模型、相同问题，on/off 随机交错至少各 20 次，同时记录延迟、空正文、
   答案正确率和角色自然度。
5. **真实 Telegram 冒烟**：当前虚拟测试验证了项目可控链路；仍需用唯一真实账号确认 `/model` 命令、
   Telegram `effective_user.id` 映射和重启持久化。

## 8. 回归状态

- Chatbot 接入层：40 项测试通过。
- 虚拟多账号真实模型验收：9/9 检查通过。
- 大型剧情冷路径：9/9 运行、27/27 事实槽通过。

本阶段结论：用户现在可以安全控制角色模型的 thinking 与常用底层生成参数，多账号隔离无需多个真实
Telegram 账号也能进行端到端验证；大型知识检索的下一优化重点应是 rerank 与 intent 模型链，而不是
float32 或 NumPy 搜索。

## 9. 后续执行结果

上述第 7 节的 rerank 输入消融已经执行。48、32、24 首轮均为 9/9 事实槽；24 补足三轮后累计
27/27 事实槽命中。24 相对 64 将相关提示字符约减半，串行平均总耗时仅下降约 3.2%，说明 token 收益
明确但延迟主要仍在模型服务端。前台默认值已调整为 24，完整数据见
`RERANK_ABLATION_REPORT_2026-08-30.md`。
