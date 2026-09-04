# 证据驱动检索架构迁移报告

日期：2026-08-31

## 1. 目标

本次迁移解决五个耦合问题：

1. 普通的“为什么、关系、第一次”等词不再静态触发 deep。
2. standard 结束后正式判断证据是否足够，而不是把 reranker 排名当作完整性证明。
3. `light / standard / deep` 只作为 preset，内部策略由请求级 `RetrievalPlan` 分别控制。
4. 前台查询不再临时修改共享 `app_config`，同一知识域可以并行读取。
5. BGE Top-N 后保留 Coverage Selector，并生成回答阶段可执行的 `LoreFactContract`。

## 2. 专有名词

### RetrievalPlan

一次请求的不可变检索策略。字段包括：

- `query_planner`：启发式或 LLM 查询规划；
- `graph_hops`：Association 图最大扩展跳数；
- `candidate_limit`：进入重排/覆盖选择的候选预算；
- `reranker`：配置的 reranker、LLM 或禁用；
- `evidence_slots`：是否保护独立事实槽；
- `followup_policy`：不追问、证据不足再升级或始终规划 follow-up；
- `verification`：本地或 LLM 验证；
- `deadline_seconds`：计划的延迟预算；
- Episode、Concept、Association path 的最终输出预算。

### Coverage Selector

重排之后的覆盖选择器。它不只问“哪条最相关”，还检查多个独立问题槽是否分别保留了候选。对于含有多个疑问槽的请求，系统从 dense、sparse 和结构化查询的独立排名中建立 evidence floor；BGE 排序可以重排其他候选，但不能把所有后置事实槽全部挤掉。

### RetrievalQuality

standard 结束后的可审计质量记录：

- `entity_coverage`：目标实体在最终 Episode/Concept 中的覆盖比例；
- `claim_slot_coverage`：独立事实槽的覆盖比例；
- `top_score`：第一候选的 reranker 或融合分数；
- `score_margin`：第一名与第二名的分差；
- `source_diversity`：独立 `source_key` 数；
- `timeline_conflict`：时间线服务是否报告冲突；
- `evidence_count`：最终 Episode 数；
- `sufficient` 与 `reasons`：是否足以直接回答及升级原因。

### LoreFactContract

回答模型的剧情证据边界。合同记录本轮允许使用的 Episode、Source、Association、认知状态、最大 generation、时间线冲突和未解决原因。证据不完整时，回答 prompt 会明确要求说明不确定性。

## 3. preset

| preset | Planner | 图跳数 | 候选预算 | reranker | follow-up |
|---|---:|---:|---:|---|---|
| light | heuristic | 1 | 20 | configured/BGE | never |
| standard | heuristic | 2 | 30 | configured/BGE | on insufficient evidence |
| deep | LLM | 3 | 100 | LLM（配置 reranker 时） | always |

`为什么、关系、联系、第一次、背后、因果`不再决定初始 deep。只有明确的“综合分析、深入分析、完整推导、深度检索”等请求可以从 deep 开始。

## 4. 执行流程

```text
问题
  → 域路由（private / public / knowledge）
  → preset 展开为 RetrievalPlan
  → dense + sparse + Paragraph + Concept + Association 召回
  → BGE/LLM rerank
  → Coverage Selector
  → RetrievalQuality
  → sufficient: LoreFactContract → 回答
  → insufficient: 主目标域自动 deep → 新合同 → 回答
```

自动升级按主目标域执行：

- 剧情问题只升级 knowledge，不会因为辅助 public 域为空而额外调用 deep；
- 私人记忆问题升级 user；
- 仅公共记忆的问题才升级 public；
- light 不自动升级；
- 后台增长直接使用 deep，不走 standard 后递归升级。

## 5. 并发边界

前台每个请求创建独立 AppConfig 快照、ModelClient、QueryEngine、GraphTraverser 和查询期辅助状态。以下重对象仍共享：

- SQLite Repository 工厂；
- Episode / Concept / Paragraph / Association RAM index；
- 只读检索数据。

因此前台 light、standard 和 deep 可以并行，不再需要用一个大锁保护配置切换。前台取消请求也不需要等待后台线程完成配置恢复。

导入、候选审计和 Association 后台增长仍可能写 SQLite 或刷新索引，继续使用 `_operation_lock`。后台 `to_thread` 被取消时仍等待实际写任务结束后才释放写锁。

## 6. 可观测输出

每个域的 raw result 新增：

- `retrieval_plan`；
- `coverage_selector`；
- `retrieval_quality`；
- `lore_fact_contract`；
- `retrieval_escalation`（发生升级时）。

升级 trace 保留首轮质量、首轮 Episode ID、首轮 timings、升级原因和 deep 错误，便于离线统计“哪些 standard 实际需要 deep”。

## 7. 验证结果

- chatbot 记忆集成：54 项通过；
- chatbot 记忆集成 + 平台策略：65 项通过；
- associative-memory 全量：246 项通过；
- Python compileall：两个项目均通过；
- chatbot 全量发现 132 项，除可选 Matrix 测试依赖 `matrix-nio` 未安装外，其余执行项通过，5 项按环境跳过。

回归过程中还修复了一个独立问题：未安装 `slixmpp` 时，XMPP allowlist 过去不会验证 JID；现在无可选依赖时也会验证并规范化 bare JID。

## 8. 当前限制

1. `top_score` 与 `score_margin` 已记录但暂不作为硬升级阈值。BGE 分数需要在真实问题集上校准，不能直接假定 0.5 或其他通用阈值。
2. `source_diversity` 是诊断指标，不是强制条件。同一文件完全可能包含一个问题所需的全部事实。
3. entity coverage 目前以规范化字符串命中为主。别名、多语言名称和代号虽可通过 Concept 进入证据，但质量判定仍可能保守升级。
4. 普通多问句按疑问标记和结构化查询建立事实槽；隐含但没有疑问词的槽仍可能漏检。
5. `deadline_seconds` 目前是计划与诊断字段，尚未作为强制超时。直接强制取消同步 API 线程会留下继续运行的网络调用，需要先统一 ModelClient 的连接、读取和总请求超时。
6. hard case 会产生 standard + deep 两次检索成本。这是证据驱动升级的预期代价，后续应比较节省的普通请求数量是否足以覆盖该成本。

## 9. 下一轮实验

建议从真实会话和现有剧情题集中抽取至少 100 个问题，记录：

- standard 直接通过率；
- 自动升级率及升级原因分布；
- 升级前后正确率、事实槽覆盖率和平均/95 分位延迟；
- BGE `top_score`、`score_margin` 与人工正确性的对应关系；
- entity alias 导致的误升级率；
- Candidate@20、30、40 对多事实覆盖和延迟的影响；
- public/user/knowledge 各域实际 deep 次数，确认主目标域抑制是否有效。

只有完成这组校准后，才适合把分数阈值、source diversity 或硬 deadline 纳入生产升级规则。
