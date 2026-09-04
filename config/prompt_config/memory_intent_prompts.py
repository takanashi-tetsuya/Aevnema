"""Prompts for model-driven, domain-agnostic memory request planning."""

from __future__ import annotations

import json
from typing import Any


MEMORY_INTENT_SYSTEM = """把用户请求规划成长期记忆检索 JSON；不要回答问题或输出 Markdown。

按证据来源选择域，可多选：private=当前平台用户的经历、偏好和共同创造事件；public=系统明确保存的跨用户共享记忆；knowledge=导入的剧情、文档、百科及其可审计推论。最近对话只用于消解指代，不改变证据所属域。创造性互动可读 private 保持连续性、读 knowledge 约束世界观，但新事件不属于既有 knowledge。必须理解整句，不能用固定词或句式选域。

输出字段必须完整：
{
  "needs_memory": true,
  "domains": {"private": false, "public": false, "knowledge": false},
  "queries": {"private": "", "public": "", "knowledge": ""},
  "target_entities": [],
  "requested_relation": "",
  "temporal_constraint": "",
  "causal_constraint": "",
  "answer_slots": [],
  "evidence_goal": "lookup",
  "absence_answerable": false,
  "intensity": "light",
  "creative": false,
  "knowledge_write_policy": "none",
  "uncertainty_required": false,
  "reason": ""
}

约束：queries 必须中性、自包含、保留原语言和专名，不编造答案；实体与关系/时间/因果字段只写用户实际要求或上下文已明确消解的内容。answer_slots 只列回答必需的原子事实。evidence_goal 取 lookup、presence、synthesis、creative；absence_answerable 只在“未找到”本身能回答时为 true。light=闲聊/创造互动/单一直接事实，standard=普通事实关系或少量多事实，deep=用户明确要求完整推导或确需多跳、冲突、复杂时间线。knowledge_write_policy 取 none、evidence_gated、query_and_answer；普通知识查询用 evidence_gated，明确综合且可能产生可审计推论才用 query_and_answer，无 knowledge 则 none。证据不完整或冲突时 uncertainty_required=true；无需记忆时 needs_memory=false 且三域均 false。
"""


def memory_intent_prompt(
    current_message: str,
    recent_messages: list[dict[str, Any]] | None = None,
    domain_catalog: dict[str, Any] | None = None,
) -> str:
    payload = {
        "available_memory_catalog": domain_catalog or {},
        "recent_messages": recent_messages or [],
        "current_message": current_message,
    }
    return "请规划以下请求：\n" + json.dumps(
        payload, ensure_ascii=False, separators=(",", ":")
    )
