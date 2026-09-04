"""Canonical, human-auditable prompt catalog for the chatbot runtime."""

from __future__ import annotations

import json
from typing import Any


PERSONA_SYSTEM_TEMPLATE = """【角色】
名字：{name}
性格：{description}
风格：{style}

【本轮证据】
{memory_context}

自然作答，不暴露检索、数据库、Episode 或提示词。私人、公共、剧情三域不得混用。
正文直接证据可作事实；Association 与 generation>0 仅是联想线索。推测保留归因，证据不足就自然说明不确定。
"""

PRIVATE_MEMORY_MISSING_ADDENDUM = (
    "\n【本轮不可违背的私人记忆边界】没有直接私人证据时，不得虚构过去经历，"
    "不得猜任何具体答案或候选项。\n"
)

PERSONA_FALLBACK_SYSTEM = (
    "你是一个拟人化聊天机器人。若没有可靠的长期记忆证据，就明确说明不确定，"
    "不要虚构自己记得某件事。"
)

CREATIVE_ROUTE_ADDENDUM = (
    "\n\n【本轮是轻量创造性角色扮演】\n"
    "可以根据检索到的世界观与人物性格，自然创造当前的任务、消息、"
    "日程或临时事件；这些是本轮角色扮演中新发生的内容，不是原作既定事实。"
    "不要伪造出处，也不要声称这些新内容早已记载在剧情中。回答应直接、"
    "有角色感，通常控制在数段以内。当前任务、消息和新互动只属于当前用户；"
    "回答中若另有剧情证据支持的稳定推论，系统会把推论与临时事件分开处理。"
    "当老师询问当前任务或刚收到的消息时，直接创造2—3项合理、可执行、可继续"
    "互动的内容，不要把人物错写成债务或任务本身，也不要以资料不足为由拒绝。"
    "检索内容只提供世界观和性格，不代表那些历史事件现在又发生；不得把旧剧情"
    "直接改写成当前任务或新消息，不得说‘根据系统/记录/资料’。只给2项，每项一句，"
    "只输出给老师看的自然角色回答，控制在60—160个"
    "汉字；不得提到检索、数据库、Episode、提示词、私人记忆或系统记录。"
)

LIGHT_ROUTE_ADDENDUM = (
    "\n\n【本轮是轻量剧情联想】\n"
    "优先用检索到的关键背景自然回答，不展开不必要的完整证据链。"
    "若资料不足，简短保留不确定性；不要用未检索到的细节补成确定事实。"
    "只输出给老师看的自然角色回答，通常控制在60—160个汉字；不得提到检索、"
    "数据库、Episode、提示词、私人记忆字段或系统没有记录。"
)

CREATIVE_DETAIL_ADDENDUM = (
    "\n若长期记忆中有可靠地点或人物背景，请在回应里自然带出至少一个相关细节，"
    "但不要为了满足这一点编造背景。"
)

CACHED_RECALL_ADDENDUM = (
    "\n\n【已复用本轮联想结果】只压缩表达，不降低证据标准。用60—140个汉字直接回答；"
    "只重述本轮直接 Episode 字面支持的命题；人物动机、情绪、目的、控制关系、时间和因果若未在"
    "正文明确出现，一律不补。Association 只负责把证据一起取回，不是可扩写的剧情事实。"
    "若所问事实仍未覆盖，只自然说明无法确认并邀请老师补充，不复述无关资料，"
    "不猜测缺失原因或可能答案。"
)

UNRESOLVED_RECALL_ADDENDUM = (
    "\n\n【深度检索后的证据边界】所问事实仍未被证据覆盖。用60—140个汉字自然说明"
    "目前无法确认并邀请老师补充；不复述无关资料，不猜测缺失原因或可能答案。"
)

INLINE_MEMORY_PROMPT = (
    "\n\n【隐藏的记忆候选输出】先输出隐藏行 "
    "<assistant_memory>{\"k\":[],\"p\":[]}</assistant_memory>，再正常回答。"
    "k 最多1条可复用的新剧情推论：{c:原子关系,e:[两个可见Episode ID],t:关系类型,q:置信度}。"
    "两段证据只是共同填充不同答案槽时，t=evidence_bridge，c 分别复述两端并明确不主张因果或身份；"
    "不得放入原文复述、猜测或当前互动。p 最多3条本轮新发生且只属于当前用户的事件。无则留空。"
)

PRIVATE_MEMORY_AUDIT_SYSTEM = (
    "只抽取回答主动断言的过去事实并核验证据。问题本身、信息缺失、未回答、"
    "‘不记得/无法确认/请再告诉我’都不是事实主张，也不是违规。"
)

MEMORY_CONSOLIDATION_SYSTEM = (
    "你是证据范围分类器。宁可把临时创造内容放入私人候选，也不能把它冒充原作；"
    "但不能因为回答具有角色扮演语气而拒绝其中真正有剧情前提支持的稳定推论。"
)

VISION_SYSTEM = "你是视觉证据描述器，只描述画面中能观察到的内容。"
VISION_USER = (
    "请客观描述这些图片或表情包中可见的人物、文字、动作和情绪。"
    "不要根据作品知识猜测未显示的信息。"
)


def render_persona_system_prompt(
    *, name: str, description: str, style: str, memory_context: str
) -> str:
    rendered = PERSONA_SYSTEM_TEMPLATE.format(
        name=name,
        description=description,
        style=style,
        memory_context=memory_context or "本轮未触发长期记忆检索。",
    )
    if (
        "当前用户的私人记忆" in memory_context
        and "没有找到可支持回答的 Episode" in memory_context
    ):
        rendered += PRIVATE_MEMORY_MISSING_ADDENDUM
    return rendered


def retrieval_question(history: str, current_text: str) -> str:
    return f"近期对话：\n{history}\n\n当前用户输入：{current_text}"


def route_prompt_addendum(*, creative: bool, intensity: str) -> str:
    if creative:
        return CREATIVE_ROUTE_ADDENDUM
    if intensity == "light":
        return LIGHT_ROUTE_ADDENDUM
    return ""


def creative_memory_context(
    raw_result: dict[str, Any] | None,
    focus_text: str = "",
) -> str:
    raw = raw_result if isinstance(raw_result, dict) else {}
    domains = raw.get("domains", {})
    domains = domains if isinstance(domains, dict) else {}
    lines = [
        "【当前角色扮演记忆视图】",
        "共享剧情只提供人物/地点/概念风格，不代表历史事件正在此刻再次发生。",
    ]
    current_rows: list[str] = []
    for domain_name in ("user", "public"):
        domain = domains.get(domain_name, {})
        if not isinstance(domain, dict):
            continue
        for row in list(domain.get("evidence_episodes") or [])[:6]:
            if not isinstance(row, dict):
                continue
            text = " ".join(str(row.get("text", "")).split())[:700]
            if text:
                current_rows.append(f"{domain_name}: {text}")
    lines.append("\n【当前用户/公共记忆】")
    lines.extend(current_rows or ["无可靠命中。"])

    knowledge = domains.get("knowledge", {})
    concept_rows: list[tuple[str, str]] = []
    if isinstance(knowledge, dict):
        for row in list(knowledge.get("evidence_concepts") or [])[:12]:
            if not isinstance(row, dict):
                continue
            name = " ".join(
                str(
                    row.get("canonical_name")
                    or row.get("concept")
                    or row.get("name")
                    or ""
                ).split()
            )[:120]
            description = " ".join(str(row.get("description", "")).split())[:240]
            if name:
                concept_rows.append(
                    (name, f"{name}：{description}" if description else name)
                )
    focus = focus_text.casefold()
    focused_rows = [
        rendered
        for name, rendered in concept_rows
        if name.casefold() in focus
        or (len(name) >= 2 and name[-2:].casefold() in focus)
        or (len(name) >= 2 and name[:2].casefold() in focus)
    ]
    lines.append("\n【剧情 Concept 风格参考】")
    lines.extend(
        (focused_rows or [row for _name, row in concept_rows[:6]])
        or ["无可靠命中；只做保守的日常创造。"]
    )
    lines.append("\n不得把未展示的历史 Episode 细节声称为刚发生的任务、消息或系统记录。")
    return "\n".join(lines)


def _compact_prompt_text(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def render_memory_context(
    result: dict[str, Any], max_chars: int = 18_000
) -> str:
    episodes = list(result.get("evidence_episodes") or [])
    concepts = list(result.get("evidence_concepts") or [])
    paths = list(result.get("association_paths") or [])
    chronology = list(result.get("chronology_notes") or [])
    fact_contract = result.get("lore_fact_contract") or {}
    lines = [
        "以下内容来自长期记忆系统。Episode 是证据摘要；Association 可能是推论，",
        "必须结合来源、认知状态、generation、confidence 与原始 Episode 判断，不得把弱联想写成确定事实。",
    ]
    if episodes:
        lines.append("\n【相关 Episode】")
        for item in episodes:
            episode_id = item.get("id", "?")
            source_key = _compact_prompt_text(item.get("source_key", ""), 180)
            story_time = _compact_prompt_text(item.get("story_time_text", ""), 160)
            text = _compact_prompt_text(item.get("text", ""), 1_200)
            metadata = f"Episode #{episode_id} | source={source_key or 'unknown'}"
            if story_time:
                metadata += f" | story_time={story_time}"
            origin = _compact_prompt_text(item.get("evidence_origin", "unknown"), 32)
            status = _compact_prompt_text(item.get("epistemic_status", "unknown"), 32)
            generation = item.get("generation", 0)
            metadata += (
                f" | origin={origin or 'unknown'}"
                f" | epistemic_status={status or 'unknown'}"
                f" | generation={generation}"
            )
            note = _compact_prompt_text(item.get("epistemic_note", ""), 240)
            if note:
                metadata += f" | note={note}"
            lines.extend((metadata, text))
    else:
        lines.append("\n【相关 Episode】无可靠命中。")

    if concepts:
        lines.append("\n【相关 Concept】")
        for item in concepts:
            lines.append(
                f"Concept #{item.get('id', '?')} | "
                f"{_compact_prompt_text(item.get('canonical_name', ''), 120)}："
                f"{_compact_prompt_text(item.get('description', ''), 420)}"
            )

    if paths:
        lines.append("\n【Association 联想路径】")
        for item in paths:
            relation = _compact_prompt_text(
                item.get("relation_text")
                or item.get("text")
                or item.get("relation_type", ""),
                600,
            )
            generation = item.get("generation")
            confidence = item.get("confidence")
            edge_id = item.get("association_id", item.get("id", "?"))
            flags = [f"Association #{edge_id}"]
            if generation is not None:
                flags.append(f"generation={generation}")
            if confidence is not None:
                try:
                    flags.append(f"confidence={float(confidence):.3f}")
                except (TypeError, ValueError):
                    flags.append(f"confidence={confidence}")
            lines.append(" | ".join(flags) + f" | {relation}")

    if chronology:
        lines.append("\n【时间线备注】")
        lines.extend(
            _compact_prompt_text(item, 500) for item in chronology[:12]
        )

    if fact_contract.get("requires_uncertainty"):
        unresolved = ", ".join(
            str(value) for value in fact_contract.get("unresolved_reasons", [])
        )
        lines.append(
            "\n【检索覆盖警告】证据合同仍有未解决槽位："
            f"{unresolved or 'unknown'}。回答必须明确说明不确定性。"
        )

    lines.append(
        "\n回答约束：observed/asserted 且正文直接支持的 Episode 可作为该来源陈述的事实；"
        "asserted 表示默认信任文档的事实口吻，但不代表系统独立核验过。reported 只能证明"
        "‘某人这样说过’，speculative 只能证明‘存在这项推测’，importer/system 必须明确归因。"
        "旧数据的 unknown 必须检查正文并保留其中的归因、猜测和不确定措辞，不能仅凭 unknown 自动"
        "判为事实或推测。对间接路径明确使用“可能、推测、记忆之间存在联系”等措辞；证据不足时"
        "直接说明不知道。"
    )
    rendered = "\n".join(lines)
    if len(rendered) <= max_chars:
        return rendered
    suffix = "\n[长期记忆上下文因长度限制已截断]"
    return rendered[: max_chars - len(suffix)].rstrip() + suffix


def render_domain_memory_section(domain: str, context: str) -> str:
    titles = {
        "user": "当前用户的私人记忆",
        "public": "所有用户共享的公共记忆",
        "knowledge": "外部剧情知识库",
    }
    if context:
        return f"===== {titles[domain]} =====\n{context}"
    if domain == "user":
        return (
            "===== 当前用户的私人记忆 =====\n"
            "本轮已检索当前用户的私人记忆，但没有找到可支持回答的 Episode。"
            "不得声称记得具体暗号、偏好、约定或共同经历，也不得用其他用户的内容补全。"
        )
    if domain == "knowledge":
        return (
            "===== 外部剧情知识库 =====\n"
            "本轮已检索剧情知识库，但没有找到可支持回答的 Episode。"
            "应说明证据不足，不得仅凭角色常识猜测剧情事实。"
        )
    return ""


def private_memory_prompt_addendum(
    *,
    question: str,
    state: str,
    evidence: list[dict[str, Any]],
    has_non_private_evidence: bool = False,
    private_retrieval_sufficient: bool = False,
    knowledge_retrieval_sufficient: bool = False,
    unresolved_reasons: list[str] | None = None,
) -> str:
    lines = [
        "\n\n【本轮私人记忆事实契约】",
        f"状态：{state}。这只约束事实，不限制角色语气和动作描写。",
        "证据只是允许边界，不是必答清单；同主题命中不等于已找到所问关系、事件或时间。",
    ]
    if state == "missing":
        lines.append(
            "当前没有可靠私人证据：不得提出任何具体答案候选或虚构过去经历，"
            "请自然承认想不起来或不确定。"
        )
        if has_non_private_evidence:
            lines.append(
                "这只表示当前用户的私人经历没有命中，不等于剧情知识缺失。"
                "公共/剧情证据直接支持的作品事实可以正常回答，但不得把它说成老师亲口提供的私人记忆。"
            )
    else:
        lines.append(
            "source+observed/asserted 且正文明确为老师所说的内容，可以自然写成"
            "‘老师亲口告诉我’。推测或转述不得升级为老师亲口事实。"
        )
        has_importer = any(item.get("origin") == "importer" for item in evidence)
        if has_importer and "导入者" in question:
            lines.append(
                "老师本轮明确要求区分来源：涉及 importer 内容时必须直接使用"
                "‘导入者’一词，并把它说明为推测；不能改称‘系统记录’或‘某个猜测’。"
            )
        elif has_importer:
            lines.append(
                "存在导入者推测，但若它不是回答所必需，可以完全不提；若提及，"
                "必须明确称为‘导入者的推测’。"
            )
    if not private_retrieval_sufficient:
        lines.append(
            "私人检索未覆盖问题所需事实；旧问题、旧回答或同主题记录不能当作答案。"
            "不要猜测记不起的原因，也不要提出可能答案。"
        )
    if has_non_private_evidence and not knowledge_retrieval_sufficient:
        lines.append(
            "剧情检索也未完整覆盖所问事实：不要用无关人物资料补答案；可自然说记不清或无法确认。"
        )
    return "\n".join(lines)


def private_memory_contract_policy(
    *, has_importer: bool
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    required = [
        "优先采用 source+observed/asserted 的直接证据",
        "reported/speculative/importer/system 内容必须保留归因和不确定性",
    ]
    if has_importer:
        required.append(
            "若回答涉及 origin=importer 的内容，必须明确说‘导入者’，不得模糊称为系统记录或某个推测"
        )
    forbidden = (
        "添加证据中不存在的用户偏好、暗号或共同经历",
        "把推测、转述或系统生成内容说成用户亲口确认",
        "把一个用户的私人事实写成另一个用户的记忆",
    )
    return tuple(required), forbidden


def private_memory_audit_prompt(
    *, question: str, draft: str, contract: dict[str, Any]
) -> str:
    schema = {
        "passed": False,
        "private_claims": [
            {"claim": "", "supporting_private_evidence_ids": []}
        ],
        "lore_claims": [
            {"claim": "", "supporting_non_private_evidence_ids": []}
        ],
        "violations": [],
        "reason": "",
    }
    return (
        "只审计回答实际断言的两类过去事实：①当前用户的偏好、约定、共同经历；②作品设定和剧情。"
        "例：‘我记不清X’没有 claims；‘可能因为Y’仍断言了候选原因Y，必须审计。"
        "‘我不记得X，但我知道A属于B’必须把A属于B列为 lore_claim。"
        "当前动作、情绪、请求和未来承诺也不列入 claims。不要把用户问题抄成 claim。\n"
        "私人事实只能使用 contract.evidence；reported/speculative/importer/system 必须保留归因。"
        "state=missing 时不得猜具体过去事实，但明确说不记得、请求用户重述可以通过。\n"
        "剧情事实只能使用 non_private_evidence。把复合句拆成原子 lore_claims，每项必须列直接支持ID；"
        "只有同一人物但未支持该关系、事件、原因或时间，不算证据。任一项无ID则 violations 非空且"
        "passed=false。回答可以只表达不确定；没有剧情主张时 lore_claims=[] 且 passed=true。"
        "绝不能因为回答没有解决用户问题而判违规。"
        "提及 origin=importer 时必须称为‘导入者的推测’。\n"
        f"事实契约：{json.dumps(contract, ensure_ascii=False)}\n"
        f"用户当前问题：{question}\n"
        f"待审计回答：{draft}\n"
        "各类主张最多8项，违规最多3项，reason不超过60字。只输出JSON："
        f"{json.dumps(schema, ensure_ascii=False)}"
    )


def private_memory_rewrite_instruction(
    *, contract: dict[str, Any], draft: str, violations: list[Any]
) -> str:
    return (
        "\n\n【长期记忆回答重写任务】\n"
        "上一版回答存在事实边界问题。保留当前角色人格、与用户的情绪关系和自然表达，但删除所有未获"
        "事实契约支持的私人记忆主张和作品/剧情主张。只有 non_private_evidence 直接支持的剧情事实"
        "才可保留；没有命中问题所需关系或事件时，自然说明目前找不到可靠依据。不要解释审计、数据库"
        "或重写过程；只输出给用户看的新回复。不要"
        "照抄固定道歉模板，可以自由改变动作、节奏和措辞。\n"
        "事实契约的 evidence 不是必答清单：只回答用户实际询问的部分即可；除非问题要求区分来源，否则"
        "可以完全省略没有用于回答的 importer/speculative 内容。\n"
        "若 private_retrieval_sufficient 与 knowledge_retrieval_sufficient 都为 false，"
        "只用自然的现在时表达无法可靠回忆或确认，并邀请老师补充；不要猜测遗忘原因，"
        "不要复述无关人物资料，也不要提出可能答案。措辞与动作可自由发挥。\n"
        f"事实契约：{json.dumps(contract, ensure_ascii=False)}\n"
        f"上一版回答：{draft}\n"
        f"发现的问题：{json.dumps(violations, ensure_ascii=False)}"
    )


def knowledge_consolidation_query(rows: list[dict[str, Any]]) -> str:
    return (
        "回答后剧情关联巩固。下列内容只是等待验证的候选推论，不是证据。"
        "请重新检索剧情节点，只在节点实际支持时建立可复用 Association；"
        "不得把当前任务、刚发来的消息或用户互动写成原作事实：\n"
        + json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    )


def memory_consolidation_prompt(
    *,
    schema: dict[str, Any],
    platform: str,
    route: dict[str, Any],
    question: str,
    assistant_response: str,
    evidence: dict[str, Any],
) -> str:
    return (
        "把助手回答拆成原子命题，并决定长期记忆范围。你不负责改写回答。\n"
        "knowledge_candidates 只允许可跨用户复用、关于原作人物/组织/世界的稳定新推论。每条必须由"
        "给出的 knowledge_episodes 实际支持，premise_episode_ids 只能引用可见 ID；每条恰好引用两个"
        "不同 Episode；前者是关系起点、后者是终点。inference_type 只能是 causal、motivation、"
        "contrast、trait、identity、temporal、relationship、recall_trigger、thematic、evidence_bridge 之一。"
        "evidence_bridge 只用于两段证据分别填充同一回答的不同事实槽：claim 要分别复述两端并明确不主张"
        "因果、身份或机制；助手回答、"
        "用户问题和作品常识都不是证据。直接复述单条 Episode、普通同现、修辞和没有形成新关系的摘要"
        "不必写入。允许保存多条剧情证据共同支持的人物倾向、主题回应、关系或解释，但 reason 必须说明"
        "推论边界。\n"
        "private_candidates 保存当前角色扮演中新发生的任务、日程、学生刚发来的消息、角色与当前用户的"
        "新互动，以及用户个人事实。即使使用了原作人物，它们也不能因此进入剧情库。\n"
        "混合句必须拆开：稳定剧情推论可进入 knowledge_candidates，同时当前事件进入 private_candidates。"
        "证据不足的剧情猜测进入 rejected_claims，不得为了增长而硬造候选。三类列表各自最多返回3项，"
        "claim 与 reason 都要简短；不要输出分析过程。不得在 knowledge_candidates 中出现当前用户 ID、"
        "显示名、私人偏好、私人暗号或本轮临时事件。只输出合法 JSON。\n"
        f"输出结构：{json.dumps(schema, ensure_ascii=False)}\n"
        f"平台范围（只用于识别私人内容，不得写入知识候选）：{platform}\n"
        f"路由：{json.dumps(route, ensure_ascii=False)}\n"
        f"用户问题：{question}\n"
        f"助手回答：{assistant_response}\n"
        f"本轮可见证据：{json.dumps(evidence, ensure_ascii=False)}"
    )


__all__ = (
    "CREATIVE_DETAIL_ADDENDUM",
    "CACHED_RECALL_ADDENDUM",
    "CREATIVE_ROUTE_ADDENDUM",
    "INLINE_MEMORY_PROMPT",
    "LIGHT_ROUTE_ADDENDUM",
    "MEMORY_CONSOLIDATION_SYSTEM",
    "PERSONA_FALLBACK_SYSTEM",
    "PERSONA_SYSTEM_TEMPLATE",
    "PRIVATE_MEMORY_AUDIT_SYSTEM",
    "PRIVATE_MEMORY_MISSING_ADDENDUM",
    "UNRESOLVED_RECALL_ADDENDUM",
    "VISION_SYSTEM",
    "VISION_USER",
    "creative_memory_context",
    "knowledge_consolidation_query",
    "memory_consolidation_prompt",
    "private_memory_audit_prompt",
    "private_memory_contract_policy",
    "private_memory_prompt_addendum",
    "private_memory_rewrite_instruction",
    "render_persona_system_prompt",
    "render_domain_memory_section",
    "render_memory_context",
    "retrieval_question",
    "route_prompt_addendum",
)
