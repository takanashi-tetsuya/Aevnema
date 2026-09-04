# 前台冷查询阶段计时与证据槽预算报告

## 1. 实验目标

此前只能观察一次检索的总耗时，无法判断瓶颈来自 float32 向量扫描、SQLite、图遍历还是远端模型。
本阶段在不改变检索语义的前提下加入结构化计时，并验证一个架构问题：用于高召回的多个同义检索问题，
是否应该全部被当作独立事实槽交给证据精排模型。

本报告只测前台记忆召回，不包含机器人最终组织自然语言回复的时间。前台关闭 Association 增长、关闭
基准 LRU，数据库仍为正式剧情库的只读检索路径。

## 2. 专有名词

- **检索问题（retrieval query）**：用于 embedding、FTS 和多路召回的文字查询。多个同义改写可以提高
  召回率，它们不一定代表多个不同答案。
- **事实槽/原子证据槽（atomic evidence slot）**：最终答案必须独立核验的一项事实，例如“爆炸现场”、
  “直接执行者”和“此前支援者”。同一事实的十种问法仍应是一个槽。
- **Candidate@100**：进入 LLM 精排前的 100 个 Episode 候选。本阶段没有缩小它。
- **evidence rerank**：模型从 Candidate@100 建立 coverage 并选出回答证据的阶段。
- **compressor**：初次 coverage 明确有缺口或过薄时，追加的一次短名单压缩/修正调用。
- **固定查询计划（fixed query plan）**：预先固定 intent 和 follow-up 查询，使 A/B 两侧看到完全相同的
  检索问题与候选集合，用来隔离上游 LLM 随机性。
- **unattributed time**：尚未归入具体阶段的对象组装与日志开销。

## 3. 计时实现

QueryEngine 现在在每个结果的 `timings` 中返回 `query-stage-timing-v1`：

```text
intent_parse
initial_embedding
initial_retrieval
initial_graph_expansion
followup_planning
followup_embedding
followup_retrieval
followup_graph_expansion
source_cohort
evidence_preparation
evidence_rerank
association_growth
final_selection
growth_utility
answer_generation
```

其中 embedding 只计算远端 embedding API；retrieval 是本地 float32 矩阵、FTS、Paragraph/Concept 融合；
graph expansion 与 SQLite source cohort 分开计时。所有阶段之和与总耗时的差记录为 unattributed，便于
发现遗漏。`memory_query.py` 和固定题集基准都会保留这些字段；基准汇总每阶段平均值、范围和占比。

## 4. 三题基线结果

未限制精排事实槽时，三道题均通过，共 9/9 事实槽，结果位于
`logs/foreground-benchmark/20260829T150947.584610Z/report.json`。

| 问题 | 总耗时 | intent | follow-up 规划 | evidence rerank | 精排槽数 | 审查 |
|---|---:|---:|---:|---:|---:|---|
| 补习部表象与真相 | 26.471 s | 7.079 s | 3.528 s | 12.418 s | 5 | none |
| 乐园悖论与信任危机 | 65.208 s | 10.300 s | 7.014 s | 42.781 s | 20 | none |
| 古圣堂袭击因果链 | 170.708 s | 28.975 s | 15.667 s | 119.939 s | 21 | compress |

三题平均 87.462 秒。平均阶段占比：evidence rerank 66.75%，intent 17.67%，follow-up 规划 9.99%。
三类远端文本模型阶段合计约 94.4%。初始/后续 embedding、本地向量与稀疏召回、图遍历、source cohort
合计约 4%；因此目前没有理由用 float16/float8 或 ANN 来解决前台延迟。

第三题的初次精排响应尤其异常：21 个槽产生 20 组 coverage、2366 completion tokens，耗时约 98 秒；
随后 compressor 又消耗约 22 秒。大量槽其实是“谁支援阿里乌斯”“支援如何产生影响”等同义改写。

## 5. 架构调整

新增 `rerank_atomic_query_limit`。它只限制交给精排模型的事实槽，不删除任何召回查询，不改变
Candidate@100，也不改变 deterministic evidence floor。

前台默认 12，后台保持 40。预算选择规则：

1. 永远保留完整问题。
2. 永远保留 `__constraint_slot__` 和 `__answer_slot__`。
3. 剩余容量约三分之二给早期通用问题。
4. 约三分之一给第一跳之后生成、已经填入实体名的后续查询。

这样既不把“同义问法”全部强迫模型逐项输出 coverage，也不会只保留问题前半段而丢掉后续实体化跳跃。
trace 额外记录 `expanded_atomic_query_count` 和 `atomic_queries_truncated`。

## 6. 完整题集复验

使用 12 槽预算重新运行三题，各题一次，分别保存在：

- `logs/foreground-benchmark/20260829T152317.473223Z/report.json`：前两题；
- `logs/foreground-benchmark/20260829T152031.421933Z/report.json`：第三题。

三题仍为 9/9 事实槽。总耗时 198.509 秒、折合平均 66.170 秒；同期无限制基线为 262.387 秒、平均
87.462 秒，样本内降低 24.3%。但这两组不是固定计划配对，远端 API 和自动 query plan 都有随机性，
所以该百分比只能视为运行样本描述，不能单独证明稳定加速。

## 7. 固定计划受控 A/B

固定资产 `experiments/manifests/foreground_latency_fixed_plan_v1.json` 保存同一份 intent、13 条初始检索查询和 7 条 follow-up。
两侧最终收到顺序完全相同的 100 个候选 Episode；只改变精排事实槽预算。

| 指标 | 40 槽上限 | 12 槽上限 |
|---|---:|---:|
| 展开后的事实槽 | 20 | 20 |
| 实际送入精排 | 20 | 12 |
| Candidate | 同序 100 | 同序 100 |
| prompt tokens | 15,844 | 15,654 |
| completion tokens | 2,089 | 322 |
| evidence rerank | 139.517 s | 21.043 s |
| 总耗时 | 145.599 s | 27.060 s |
| 必要事实槽命中 | 2/3 | 3/3 |

原始报告：

- `logs/foreground-benchmark/20260829T152935.385594Z/report.json`：40 槽；
- `logs/foreground-benchmark/20260829T153014.880938Z/report.json`：12 槽。

输入 token 只减少 1.2%，completion 却减少 84.6%；精排时间减少 84.9%。这说明主要收益不是少传了
少量查询文本，而是避免模型为大量重复槽生成冗长 coverage。40 槽轮反而遗漏了直接执行证据，说明
同义槽过多还会造成注意力稀释。单次配对可能受供应商负载与模型随机性影响，因此不能把 6.6 倍速度视为
稳定 SLA；但“同样候选下更短输出且覆盖不下降”的方向已经得到直接证据。

## 8. 三组交替顺序配对

新增 `experiments/memory_foreground_atomic_ab.py`，奇数对按 40→12 运行，偶数对按 12→40 运行。每一侧关闭 LRU，
共享同一固定 intent、follow-up 和正式只读知识库。报告采用 nearest-rank P90；只有 3 个样本时 P90
等于观测最大值，不能当生产 SLA。

报告：`logs/foreground-atomic-ab/20260829T154647.544790Z/report.json`。

| 指标 | 40 槽 | 12 槽 |
|---|---:|---:|
| 运行 | 3 | 3 |
| 事实槽 | 8/9 | 9/9 |
| 完整通过 | 2/3 | 3/3 |
| 平均总耗时 | 80.933 s | 54.115 s |
| 中位总耗时 | 100.665 s | 58.737 s |
| P90（3 样本最大值） | 105.788 s | 78.038 s |
| 平均 evidence rerank | 74.813 s | 47.920 s |

3/3 配对的 Candidate@100 顺序逐项相同。12 槽每对分别节省 47.051、22.627、10.774 秒；配对中位
相对降幅 29.6%。40 槽仍有一次遗漏“阿里乌斯直接执行”，说明更多同义槽不是更多独立证据，反而可能
稀释注意力。

## 9. 配对日志暴露的问题与第二轮调整

12 槽虽然 9/9，但初版 3 次中有 2 次触发 compressor。日志表明首轮模型已经找到多个有效 coverage，
只是诚实报告“候选没有直接证明具体渗透机制”。compressor 看到的仍是同一候选池，无法创造不存在的
直接证据，因此这类调用只是重复花费。

调整如下：

1. `fast_adaptive` 仅在 coverage 结构过薄或独立槽异常多时压缩；单纯的 `missing_aspects` 保留为回答时
   不确定性。后台 `adaptive/strict` 的强审计策略不变。
2. coverage 提示词要求合并主谓宾和答案完全相同的同义问法，优先使用实体已解析的最短 query；
   `reason` 和每条缺口说明限制为 80 个汉字。
3. deterministic atomic floor 将约 1/3 查询名额预留给 follow-up，避免前六个泛化同义查询占满保护位。

修改后固定计划三连跑报告：
`logs/foreground-benchmark/20260829T155440.012214Z/report.json`。

- 3/3 通过，9/9 事实槽；
- compressor 触发 0/3；
- 平均 22.965 秒，范围 20.823—24.081 秒；
- evidence rerank 平均 16.765 秒；
- 相对修改前 12 槽三次样本，平均耗时降低 57.6%。

该降幅仍包含远端模型方差，但“零二次调用、三次事实完整、耗时范围收窄”说明修改命中了真实原因。

## 10. 动态计划反例与答案槽邻接窗

重新启用动态 intent/follow-up 后，三题报告
`logs/foreground-benchmark/20260829T155744.027075Z/report.json` 为 8/9：前两题通过，古圣堂题漏掉
“阿里乌斯直接执行”。诊断确认：正确 Episode #1421 与 #1545 已在 Candidate@100，失败不是基础召回；
显式 `__answer_slot__` 保护了 #1544（阿里乌斯兵力出现在古圣堂），但没有保护同文件相邻的 #1545
（角色明确归责阿里乌斯一手造成）。这是自动 Episode 边界把“势力出现”和“责任归属”拆开后的局部损失。

因此增加 `answer-slot neighbor floor`：只对显式答案槽命中，在相同 `source_key` 的当前 Candidate 集内
保护前后各一条 Episode，默认半径 1、额外总上限 4。它不查询整个文件、不启用 Paragraph、不建立新
Association，也不新增 API 调用。

修复后的动态古圣堂题三连跑报告：
`logs/foreground-benchmark/20260829T160457.196653Z/report.json`。

- 3/3 通过，9/9 事实槽；
- #1545 在 3/3 中保留，其中一次同时保留更直接的 #1595；
- compressor 触发 0/3；
- 平均 45.083 秒，范围 38.049—52.553 秒；
- intent + follow-up planning 平均约 22.35 秒，evidence rerank 平均 18.02 秒。

这表明当前动态冷查询的下一个主要优化对象已经是查询规划，而不是 float32、SQLite 或 Candidate@100。

## 11. 验证状态与当前边界

- 聊天项目确定性测试：21/21。
- 记忆引擎测试：164/164。
- 固定计划修改后：3/3、9/9、无 compressor。
- 动态失败题修复后：3/3、9/9、无 compressor。
- 正式知识库仍为 736 Source、4073 Episode、2970 Concept、8904 Association；前台实验没有写边。
- SQLite 与 RAM embedding 仍统一使用归一化 float32；Paragraph 仍关闭、Concept 仍为 conservative。

当前 12 是有受控证据支持的前台默认值，不是普遍最优常数。非常长、确实包含 12 个以上独立事实的
问题仍可能需要自适应提高上限。答案槽邻接窗也只解决同文件局部切分；如果正确证据根本不在
Candidate@100，它不会提供帮助。后台继续使用 40，避免前台延迟策略降低 Association 增长审计覆盖。

开发环境另有一个非算法问题：记忆引擎目录的 `.venv` 指向已不存在的 Python 3.14。此次使用聊天项目
可用的虚拟环境并显式加入引擎 `src` 完成 164 项测试；协作前应重建引擎虚拟环境。

## 12. 下一步

1. 把动态回归集扩至 10—20 题，覆盖短问句、跨文件因果、对比、否定和 8—12 个真实独立槽；至少做
   5 轮关键题重复，报告中位数、P90 和 compressor 触发率。
2. 优先优化 intent 与 follow-up planning：评估结构化 query-plan 缓存、同义查询本地去重，以及在不降低
   Candidate@100 Recall 的前提下减少规划模型输出。当前两阶段约占动态古圣堂题的一半。
3. 增加答案槽邻接窗的负例：相邻 Episode 属于转场或另一事件时不得被当作直接事实；邻接项只是候选
   保护，最终回答仍必须按 Episode 文本区分事实、归责和推论。
4. completion token 硬上限暂不启用。只有在更大样本证明 JSON 能稳定闭合并有修复/回退策略后再测试，
   避免为了速度制造截断和额外 repair 调用。
