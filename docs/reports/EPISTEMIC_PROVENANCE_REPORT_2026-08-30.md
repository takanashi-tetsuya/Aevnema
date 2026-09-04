# 记忆证据来源与认知状态改造报告

日期：2026-08-30  
实现版本：`v3.41_asserted_by_default`  
数据库结构版本：`8`

## 1. 问题

过去系统主要用 `confidence` 和 Association 的 `generation` 描述可靠性，但它们无法回答以下问题：

- 这段内容来自原始文档、导入者补充，还是系统自己生成？
- 文本是在描述已经发生的事情、转述某人的说法，还是提出猜测？
- 一条内容离直接证据有多少层推理？

如果把三者混成一个分数，导入者写下的高置信推测可能在检索、建边和回答时被当成直接事实。

## 2. 三个正交维度

### `evidence_origin`

记录“内容来自哪一层作者”，不判断内容真假：

| 值 | 含义 |
|---|---|
| `source` | 原始文档或用户原话 |
| `importer` | 导入者后来添加的批注、分析或补充 |
| `system` | 模型或程序生成的内容 |
| `mixed` | 无法拆开的多种来源混合 |
| `unknown` | 旧数据或来源无法确定 |

### `epistemic_status`

记录 Episode 中心命题的认知状态：

| 值 | 含义 |
|---|---|
| `observed` | Source 直接呈现事件发生或系统直接观测到状态 |
| `asserted` | 文档作者以事实口吻陈述；未标注的普通文档默认使用此状态 |
| `reported` | 只能确定某人作出了这项陈述或报告 |
| `speculative` | 猜测、假说、怀疑、可能性解释 |
| `mixed` | 事实、转述和推测无法安全拆开 |
| `unknown` | 尚未分类，必须结合正文保守处理 |

### `generation`

非负整数，记录中心命题经历了多少层“以推论为前提的推论”。它不是可信度：

- `0`：直接来自当前证据层；
- `1`：至少经过一层推论；
- `2+`：继续使用已有推论作前提。

导入者或系统添加的 `speculative/mixed` 内容至少记为 generation 1。查询中产生的新 Association 使用“端点 Episode 与前提 Association 的最大 generation + 1”。

`confidence` 仍只表示在当前证据下的把握程度。高 confidence 不能把 speculative 变成 asserted/observed，也不能把高 generation 变成直接经验。

## 3. 默认导入规则

普通文档不要求额外标注。没有 `_memory` 或 `[[memory ...]]` 的内容默认按下面处理：

```text
evidence_origin = source
epistemic_status = asserted
generation = 0
```

这适用于百科全书、说明书、新闻资料、整理后的知识文档等。`asserted` 的含义是“来源文档把它作为事实陈述”，不是“系统已经通过多个来源独立核验”。

如果正文自身明确写出“可能、猜测、疑似、据某人说、传闻”等内容，Episode 提取器仍应根据语义覆盖默认值，使用 `speculative` 或 `reported`。显式批注只用于需要覆盖普通默认规则的段落。

## 4. 关键语义例子

角色在原文中说“我猜未花做了这件事”：

```text
evidence_origin = source
epistemic_status = speculative
generation = 0
```

直接证据是“角色提出了猜测”，不是“未花确实做了这件事”。

导入者在文档里加上同样的解释：

```text
evidence_origin = importer
epistemic_status = speculative
generation >= 1
```

系统回答模型必须明确写成“导入者推测……”，不能写成剧情事实。

## 5. 显式导入标注

### TXT / Markdown

批注写在一个段落的第一行；该指令不会进入正文 embedding，但会进入 Source 的证据元数据：

```text
[[memory {"origin":"importer","status":"speculative","generation":1,"note":"导入者根据前后文的解释"}]]
这可能说明未花更早就参与了计划。
```

### JSON

文档级默认值：

```json
{
  "_memory": {
    "origin": "source",
    "status": "asserted"
  },
  "content": []
}
```

单条记录可以覆盖文档默认值：

```json
{
  "TextCn": "导入者认为现场可能预埋了炸药。",
  "_memory": {
    "origin": "importer",
    "status": "speculative",
    "generation": 1,
    "note": "人工推测"
  }
}
```

## 6. 全链路行为

1. 适配器读取显式标注，并把它放入标准化 Block 元数据。
2. Source 分片将标注写成可审计标签；推理视图压缩多语言内容时不会删除这些标签。
3. Episode 提取器输出四个新字段，并区分原文叙事、角色说法、角色猜测和导入者批注。
4. 时间/粒度审计不得把 `reported/speculative` 升级为 `asserted/observed`。
5. Concept 结构保持不变，但 Concept 提取提示会看到 Episode 证据层，描述不得把推测改写为事实。
6. Episode→Concept 边继承 Episode generation；Episode→Episode 推理边取两个端点的最大 generation 再加一。
7. 查询增长允许连接推测，但关系文本必须保留“谁的说法、导入者推测、尚未确认”等限定；限定丢失会被确定性守卫拒绝。
8. 粗排、精排、最终证据和聊天机器人上下文均携带四个字段。
9. 回答器可把 `observed/asserted` 且正文直接支持的命题称为该来源所陈述的事实；`reported` 只支持“某人这样说”，`speculative` 只支持“存在这项推测”。即使标签是 asserted，正文中的明确猜测措辞仍必须保留。

## 7. 聊天机器人对话记忆

每次用户—助手交流现在拆成两个独立、带标注的文本段：

- 用户原话：`source + unknown + generation 0`，由提取器根据“我亲眼看到/我听说/我猜”等语义继续分类；
- 助手回答：`system + mixed + generation 1`，允许被记住和联想，但不会冒充用户亲历事实。

这也保留了平台和平台原生用户 ID，因此用户记忆隔离规则不变。

## 8. 旧资产迁移

实际 Blue Archive 知识库已经迁移到 schema v8，资产数量和全文索引数量没有变化：

```text
Source       736
Episode     4073
Concept     2970
Association 8904
```

未标注的旧 Episode 按新的普通文档默认规则回填为：

```text
evidence_origin = source
epistemic_status = asserted
generation = 0
```

旧 Episode 正文中已经存在的“据某人说、推测、可能”等限定仍会在回答时保留；以后重新提取时会进一步写入精确状态。迁移前备份为：

```text
data/knowledge/blue_archive.pre-v7-20260830.db
data/knowledge/blue_archive.pre-v8-20260830.db
```

## 9. 验证结果

```text
核心记忆系统：177 tests passed
聊天机器人：   22 tests passed
```

专项测试覆盖：TXT/JSON 标注、压缩推理视图保留、Episode 落库往返、导入者推测的最小 generation、增长边继承端点代际，以及聊天机器人最终上下文标签。

## 10. 当前无法由技术可靠解决的边界

如果导入者不作任何标注，并把自己的推测写成完全肯定的陈述，系统无法仅凭文本可靠判断“这句话原本属于文档，还是后来由导入者添加”。LLM 可以识别“可能、我猜、疑似”等语义不确定性，但不能凭空恢复作者层来源。

因此：

- 来源归属依靠显式 `_memory` / `[[memory ...]]` 标注；
- 命题状态同时使用显式标注和 LLM 语义判断；
- 未标注且没有不确定措辞的普通内容进入 `asserted`；只有系统确实无法判断来源层或内容混杂时才使用 `unknown/mixed`。

后续若需要处理大量第三方批注文档，建议在兼容层为每种文档格式提供“正文/脚注/评论/批注作者”的确定性映射，而不是让一个模型猜完整的编辑历史。
