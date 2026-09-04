from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import tempfile
from threading import Barrier, Event
import unittest
from unittest.mock import patch

import numpy as np

from src.bot.chat_service import ConversationCoordinator
from src.bot.memory_guard import PrivateMemoryResponseGuard
from src.bot.persona import get_dynamic_system_prompt
from src.bot.request_planning import PlannedChatRequest
from src.memory.conversation import (
    ConversationIngestionWorker,
    ConversationJournal,
    ConversationSessionBuffer,
)
from src.memory.identity import PlatformIdentity
from src.memory.growth import BackgroundGrowthWorker
from src.memory.answer_consolidator import (
    derive_evidence_bridge_candidate,
    split_inline_memory_response,
)
from src.memory.service import (
    AssociativeMemoryService,
    DomainRecallRequest,
    MemoryRoute,
    MemoryServiceConfig,
    MemorySystem,
    MemorySystemConfig,
    RetrievalQuality,
    RetrievedMemory,
    assess_retrieval_quality,
    format_memory_context,
    route_memory_query,
)
from src.memory.config import _apply_ingestion_env_overrides, _portable_path_text
from src.memory.contracts import _retrieval_answer_shape
from src.memory.routing import _knowledge_retrieval_question


class ImportOverrideTests(unittest.TestCase):
    def test_import_environment_reaches_embedded_engine_config(self):
        class Ingestion:
            task_max_attempts = 2
            prepare_workers = 4
            relation_workers = 4
            relation_batch_size = 24
            episode_factual_audit_batch_size = 3
            episode_audit_always = False
            episode_factual_audit_mode = "off"
            episode_factual_audit_model = ""
            build_inference_relations = True

        class Config:
            ingestion = Ingestion()

        config = Config()
        with patch.dict(
            os.environ,
            {
                "MEMORY_IMPORT_TASK_MAX_ATTEMPTS": "3",
                "MEMORY_IMPORT_PREPARE_WORKERS": "2",
                "MEMORY_IMPORT_RELATION_WORKERS": "3",
                "MEMORY_IMPORT_RELATION_BATCH_SIZE": "12",
                "MEMORY_IMPORT_EPISODE_PROFILE": "single_pass_evidence",
                "MEMORY_IMPORT_EPISODE_AUDIT_ALWAYS": "true",
                "MEMORY_IMPORT_EPISODE_FACTUAL_AUDIT_MODE": "always",
                "MEMORY_IMPORT_EPISODE_FACTUAL_AUDIT_MODEL": "fact-checker",
                "MEMORY_IMPORT_EPISODE_FACTUAL_AUDIT_BATCH_SIZE": "2",
            },
        ):
            _apply_ingestion_env_overrides(config)

        self.assertEqual(Config.ingestion.task_max_attempts, 3)
        self.assertEqual(Config.ingestion.prepare_workers, 2)
        self.assertEqual(Config.ingestion.relation_workers, 3)
        self.assertEqual(Config.ingestion.relation_batch_size, 12)
        self.assertEqual(Config.ingestion.episode_factual_audit_batch_size, 2)
        self.assertTrue(Config.ingestion.episode_audit_always)
        self.assertEqual(
            Config.ingestion.episode_extraction_profile,
            "single_pass_evidence",
        )
        self.assertEqual(config.prompt_version, "v4.1_single_pass_line_spans")
        self.assertEqual(Config.ingestion.episode_factual_audit_mode, "always")
        self.assertEqual(
            Config.ingestion.episode_factual_audit_model, "fact-checker"
        )


class IdentityTests(unittest.TestCase):
    def test_platform_and_native_id_form_the_stable_key(self):
        telegram = PlatformIdentity("Telegram", "42", "same-name")
        discord = PlatformIdentity("discord", "42", "same-name")
        renamed = PlatformIdentity("telegram", "42", "new-name")
        self.assertEqual("telegram:42", telegram.key)
        self.assertNotEqual(telegram.key, discord.key)
        self.assertEqual(telegram.storage_key, renamed.storage_key)

    def test_identity_metadata_preserves_platform_id_and_mutable_name(self):
        with tempfile.TemporaryDirectory() as directory:
            identity = PlatformIdentity("telegram", "abc/42", "Alice")
            identity.write_metadata(directory)
            restored = PlatformIdentity.from_metadata(directory)
        self.assertEqual(identity, restored)
        self.assertNotIn("/", identity.storage_key)


class RoutingAndFormattingTests(unittest.TestCase):
    def test_casual_greeting_does_not_spend_memory_calls(self):
        self.assertEqual(
            MemoryRoute(False, False, False, "casual"),
            route_memory_query("你好呀"),
        )

    def test_fallback_is_conservative_and_domain_agnostic(self):
        for question in (
            "你还记得我们之前讨论的结论吗？",
            "某人物为什么作出这个决定？",
            "What happened after the meeting?",
        ):
            route = route_memory_query(question)
            self.assertEqual(
                MemoryRoute(
                    True,
                    True,
                    True,
                    "model_planner_fallback",
                    "standard",
                    "evidence_gated",
                ),
                route,
            )

    def test_only_explicit_analysis_request_starts_deep_fallback(self):
        self.assertEqual(
            "deep",
            route_memory_query("请综合分析全部证据并完整推导").intensity,
        )
        self.assertEqual(
            "standard",
            route_memory_query("甲和乙是什么关系？").intensity,
        )

    def test_fallback_does_not_rewrite_user_semantics(self):
        question = "你还记得某个人与老师相遇的事情吗？"
        self.assertEqual(question, _knowledge_retrieval_question(question))

    def test_fallback_source_has_no_corpus_specific_vocabulary(self):
        source = Path("src/memory/routing.py").read_text(encoding="utf-8")
        for value in ("蔚蓝档案", "伊甸园条约", "阿拜多斯", "阿里乌斯"):
            self.assertNotIn(value, source)

    def test_fast_planner_answer_shape_respects_memory_domain(self):
        self.assertEqual(
            "private_memory_evidence",
            _retrieval_answer_shape("user:telegram:42", lightweight=False),
        )
        self.assertEqual(
            "lore_evidence",
            _retrieval_answer_shape("knowledge", lightweight=False),
        )
        self.assertEqual(
            "shared_memory_evidence",
            _retrieval_answer_shape("public", lightweight=False),
        )


class RetrievalQualityTests(unittest.TestCase):
    def test_cross_encoder_entity_match_is_not_relation_proof(self):
        quality = assess_retrieval_quality(
            "老师怎样与某人相遇？",
            {
                "intent": {
                    "target_entities": ["某人"],
                    "requested_relation": "所问实体之间的相遇事件及经过",
                },
                "evidence_episodes": [
                    {
                        "id": 1,
                        "score": 0.72,
                        "text": "某人后来与老师一起处理了其他事件。",
                        "source_key": "story/later.txt",
                    }
                ],
                "rerank_trace": {
                    "backend": "cross_encoder",
                    "cross_encoder_scores": [0.72],
                },
                "evidence_slot_trace": {
                    "deterministic": {
                        "constraint_slots": [
                            {"query": "相遇事件", "satisfied": True}
                        ]
                    }
                },
            },
        )

        self.assertFalse(quality.sufficient)
        self.assertEqual(0.0, quality.semantic_constraint_coverage)
        self.assertIn(
            "semantic_constraint_relevance_low", quality.reasons
        )

    def test_missing_entity_and_claim_slot_require_escalation(self):
        quality = assess_retrieval_quality(
            "谁协助阿里乌斯，和未花有什么关系？",
            {
                "intent": {"target_entities": ["阿里乌斯", "未花"]},
                "evidence_episodes": [
                    {
                        "id": 1,
                        "score": 0.8,
                        "text": "阿里乌斯执行了袭击。",
                        "source_key": "main/a.json",
                    }
                ],
                "evidence_slot_trace": {
                    "coverage_slots": [
                        {"query": "幕后协助者", "satisfied": False}
                    ]
                },
            },
        )

        self.assertIsInstance(quality, RetrievalQuality)
        self.assertFalse(quality.sufficient)
        self.assertIn("entity_coverage_incomplete", quality.reasons)
        self.assertIn("claim_slot_coverage_incomplete", quality.reasons)

    def test_complete_multi_source_evidence_is_sufficient(self):
        quality = assess_retrieval_quality(
            "阿里乌斯和未花有什么关系？",
            {
                "intent": {"target_entities": ["阿里乌斯", "未花"]},
                "evidence_episodes": [
                    {
                        "id": 1,
                        "score": 0.9,
                        "text": "阿里乌斯发动袭击。",
                        "source_key": "main/a.json",
                    },
                    {
                        "id": 2,
                        "score": 0.7,
                        "text": "未花曾暗中协助阿里乌斯。",
                        "source_key": "main/b.json",
                    },
                ],
                "evidence_slot_trace": {
                    "coverage_slots": [
                        {"query": "双方关系", "satisfied": True}
                    ]
                },
            },
        )

        self.assertTrue(quality.sufficient)
        self.assertEqual(2, quality.source_diversity)
        self.assertEqual(1.0, quality.claim_slot_coverage)


class EvidenceEscalationRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_lore_question_escalates_knowledge_not_auxiliary_public(self):
        class FakeService:
            def __init__(self, domain):
                self.domain = domain
                self.calls = []

            @staticmethod
            def retrieval_plan(preset):
                return AssociativeMemoryService.retrieval_plan(preset)

            async def recall(
                self,
                question,
                *,
                retrieval_intensity="standard",
                retrieval_plan=None,
                auto_escalate=True,
                request=None,
                **_kwargs,
            ):
                preset = (
                    retrieval_plan.preset
                    if retrieval_plan is not None
                    else retrieval_intensity
                )
                self.calls.append((preset, auto_escalate, request))
                evidence = (
                    [
                        {
                            "id": 9,
                            "score": 0.9,
                            "text": "deep 剧情证据",
                            "source_key": "main/deep.json",
                        }
                    ]
                    if preset == "deep"
                    else []
                )
                return RetrievedMemory(
                    context=f"{self.domain}:{preset}",
                    raw_result={
                        "retrieval_intensity": preset,
                        "evidence_episodes": evidence,
                        "retrieval_quality": {
                            "sufficient": bool(evidence),
                            "reasons": [] if evidence else ["no_evidence"],
                        },
                    },
                    domains=(self.domain,),
                )

        system = MemorySystem.__new__(MemorySystem)
        system.public = FakeService("public")
        system.knowledge = FakeService("knowledge")
        public_request = DomainRecallRequest(query="共享背景")
        knowledge_request = DomainRecallRequest(
            query="外部资料中的因果",
            intent_override={"causal_constraint": "原因与结果"},
        )

        result = await system.recall(
            PlatformIdentity("synthetic", "teacher"),
            "这件事为什么发生？",
            MemoryRoute(False, True, True, "model_intent_planner", "standard"),
            domain_requests={
                "public": public_request,
                "knowledge": knowledge_request,
            },
        )

        self.assertEqual(
            [("standard", False, public_request)], system.public.calls
        )
        self.assertEqual(
            [
                ("standard", False, knowledge_request),
                ("deep", False, knowledge_request),
            ],
            system.knowledge.calls,
        )
        knowledge = result.raw_result["domains"]["knowledge"]
        self.assertEqual("deep", knowledge["retrieval_intensity"])
        self.assertTrue(knowledge["retrieval_escalation"]["triggered"])

    def test_memory_guard_distinguishes_teaching_from_recall(self):
        route = MemoryRoute(True, True, False, "personal")
        self.assertFalse(
            PrivateMemoryResponseGuard.applies(
                "请记住：我们的暗号是蓝莓雨伞。", route
            )
        )
        self.assertTrue(
            PrivateMemoryResponseGuard.applies("我们的暗号是什么？", route)
        )

    def test_memory_guard_addendum_requires_importer_only_when_relevant(self):
        route = MemoryRoute(True, True, False, "personal")
        memory = RetrievedMemory(
            "private",
            raw_result={
                "domains": {
                    "user": {
                        "evidence_episodes": [
                            {
                                "id": 7,
                                "text": "导入者推测老师可能不喜欢茶。",
                                "evidence_origin": "importer",
                                "epistemic_status": "speculative",
                                "generation": 1,
                            }
                        ]
                    }
                }
            },
        )

        explicit = PrivateMemoryResponseGuard.prompt_addendum(
            "请区分老师亲口说的和导入者推测。", memory, route
        )
        ordinary = PrivateMemoryResponseGuard.prompt_addendum(
            "你记得老师喜欢什么吗？", memory, route
        )

        self.assertIn("明确要求区分来源", explicit)
        self.assertIn("若它不是回答所必需，可以完全不提", ordinary)

    def test_private_contract_carries_public_evidence_without_changing_state(self):
        memory = RetrievedMemory(
            "combined",
            raw_result={
                "domains": {
                    "user": {"evidence_episodes": []},
                    "public": {
                        "evidence_episodes": [
                            {
                                "id": 8,
                                "text": "公共通行语是青空灯塔。",
                                "evidence_origin": "source",
                                "epistemic_status": "asserted",
                                "generation": 0,
                            }
                        ]
                    },
                }
            },
        )
        contract = PrivateMemoryResponseGuard.build_contract(memory)
        self.assertEqual("missing", contract.state)
        self.assertEqual("public", contract.non_private_evidence[0]["domain"])
        self.assertIn("青空灯塔", contract.non_private_evidence[0]["text"])
        addendum = PrivateMemoryResponseGuard.prompt_addendum(
            "你记得青空灯塔吗？",
            memory,
            MemoryRoute(True, True, True, "personal_and_entity_lore"),
        )
        self.assertIn("不等于剧情知识缺失", addendum)

    def test_formatter_labels_inference_distance(self):
        rendered = format_memory_context(
            {
                "evidence_episodes": [
                    {
                        "id": 7,
                        "source_key": "main/1.json",
                        "story_time_text": "过去",
                        "text": "直接证据",
                        "evidence_origin": "importer",
                        "epistemic_status": "speculative",
                        "generation": 1,
                        "epistemic_note": "导入者的推测",
                    }
                ],
                "evidence_concepts": [
                    {"id": 3, "canonical_name": "概念", "description": "描述"}
                ],
                "association_paths": [
                    {
                        "association_id": 9,
                        "generation": 2,
                        "confidence": 0.7,
                        "relation_text": "推论关系",
                    }
                ],
            }
        )
        self.assertIn("Episode #7", rendered)
        self.assertIn("generation=2", rendered)
        self.assertIn("origin=importer", rendered)
        self.assertIn("epistemic_status=speculative", rendered)
        self.assertIn("导入者的推测", rendered)
        self.assertIn("speculative 只能证明", rendered)
        self.assertIn("不得把弱联想写成确定事实", rendered)

    def test_no_private_hit_adds_turn_specific_non_invention_guard(self):
        prompt = get_dynamic_system_prompt(
            "===== 当前用户的私人记忆 =====\n"
            "本轮已检索当前用户的私人记忆，但没有找到可支持回答的 Episode。"
        )
        self.assertIn("本轮不可违背的私人记忆边界", prompt)
        self.assertIn("不得猜任何具体答案或候选项", prompt)
        self.assertIn("不得虚构", prompt)


class ConfigTests(unittest.TestCase):
    def test_windows_and_wsl_absolute_paths_are_portable(self):
        self.assertEqual(
            "/mnt/c/Users/Admin/PycharmProjects/PythonProject",
            _portable_path_text(
                r"C:\Users\Admin\PycharmProjects\PythonProject",
                "posix",
            ),
        )
        self.assertEqual(
            r"C:\Users\Admin\Documents\chat_bot",
            _portable_path_text(
                "/mnt/c/Users/Admin/Documents/chat_bot",
                "nt",
            ),
        )

    def test_three_domains_have_distinct_database_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = root / "engine"
            (engine / "src" / "memory_demo").mkdir(parents=True)
            with patch.dict(
                os.environ,
                {
                    "MEMORY_ENGINE_ROOT": str(engine),
                    "KNOWLEDGE_MEMORY_DB_PATH": "data/knowledge.db",
                    "PUBLIC_MEMORY_DB_PATH": "data/public.db",
                    "USER_MEMORY_DB_DIR": "data/users",
                    "SILICONFLOW_API_KEY": "test",
                    "MEMORY_RERANKER_MODEL": (
                        "Pro/BAAI/bge-reranker-v2-m3"
                    ),
                },
                clear=True,
            ):
                config = MemorySystemConfig.from_env(root)
            self.assertNotEqual(config.knowledge_database_path, config.public_database_path)
            user = PlatformIdentity("telegram", "42")
            private = config.user_database_dir / user.platform / user.storage_key / "memory.db"
            self.assertNotIn(str(config.knowledge_database_path), str(private))
            self.assertNotIn(str(config.public_database_path), str(private))

            foreground = config.domain_config("knowledge", config.knowledge_database_path)
            background = config.domain_config(
                "knowledge", config.knowledge_database_path, background=True
            )
            self.assertEqual("balanced", foreground.optimization_profile)
            self.assertEqual(
                "Pro/BAAI/bge-reranker-v2-m3",
                foreground.reranker_model,
            )
            self.assertTrue(foreground.rerank_enabled)
            self.assertEqual("cross_encoder", foreground.rerank_backend)
            self.assertEqual("fast_adaptive", foreground.rerank_review_mode)
            self.assertEqual(
                "entity_resolved", foreground.followup_planning_mode
            )
            self.assertEqual(12, foreground.rerank_atomic_query_limit)
            self.assertEqual(24, foreground.rerank_precompression_limit)
            self.assertEqual(64, foreground.recall_cache_size)
            self.assertEqual(32, foreground.small_database_fast_path_max_episodes)
            self.assertFalse(foreground.growth_enabled)
            self.assertEqual("balanced", background.optimization_profile)
            self.assertTrue(background.rerank_enabled)
            self.assertEqual("llm", background.rerank_backend)
            self.assertEqual("", background.rerank_review_mode)
            self.assertEqual("always", background.followup_planning_mode)
            self.assertEqual(40, background.rerank_atomic_query_limit)
            self.assertEqual(100, background.rerank_precompression_limit)
            self.assertEqual(0, background.recall_cache_size)
            self.assertEqual(0, background.small_database_fast_path_max_episodes)
            self.assertTrue(background.growth_enabled)

            user_background = config.domain_config(
                "user:telegram:42", private, background=True
            )
            self.assertEqual("lean", user_background.optimization_profile)
            self.assertEqual(
                "Qwen/Qwen3.5-35B-A3B", user_background.reasoning_model
            )
            self.assertEqual(
                "deepseek-ai/DeepSeek-V3.2", user_background.fallback_model
            )
            self.assertEqual(2400, user_background.reasoning_max_tokens)
            self.assertFalse(user_background.reasoning_enable_thinking)
            self.assertEqual(3000, background.reasoning_max_tokens)
            self.assertEqual(
                "deepseek-ai/DeepSeek-V3.2", background.reasoning_model
            )
            self.assertEqual(
                "Qwen/Qwen3.5-35B-A3B", background.fallback_model
            )
            self.assertFalse(background.reasoning_enable_thinking)

    def test_empty_optional_reranker_model_disables_every_domain(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = root / "engine"
            (engine / "src" / "memory_demo").mkdir(parents=True)
            with patch.dict(
                os.environ,
                {
                    "MEMORY_ENGINE_ROOT": str(engine),
                    "SILICONFLOW_API_KEY": "test",
                    "MEMORY_RERANKER_MODEL": "   ",
                },
                clear=True,
            ):
                config = MemorySystemConfig.from_env(root)
                foreground = config.domain_config(
                    "knowledge", config.knowledge_database_path
                )
                background = config.domain_config(
                    "knowledge",
                    config.knowledge_database_path,
                    background=True,
                )

            self.assertEqual("", config.reranker_model)
            self.assertFalse(foreground.rerank_enabled)
            self.assertFalse(background.rerank_enabled)
            self.assertEqual("llm", foreground.rerank_backend)
            self.assertEqual("llm", background.rerank_backend)


class InlineMemoryTests(unittest.TestCase):
    def test_coverage_proof_derives_retrieval_only_bridge_without_model_footer(self):
        raw_result = {
            "domains": {
                "knowledge": {
                    "rerank_trace": {
                        "merged_coverage": {
                            "coverage": [
                                {"query": "谁执行袭击", "episode_ids": [7]},
                                {"query": "谁提供合作背景", "episode_ids": [8]},
                            ],
                            "missing_aspects": [],
                        }
                    },
                    "evidence_episodes": [
                        {
                            "id": 7,
                            "text": "甲执行袭击。",
                            "generation": 0,
                            "evidence_origin": "source",
                        },
                        {
                            "id": 8,
                            "text": "乙参与合作。",
                            "generation": 0,
                            "evidence_origin": "source",
                        },
                    ],
                }
            }
        }

        candidate = derive_evidence_bridge_candidate("综合说明两项事实", raw_result)

        self.assertIsNotNone(candidate)
        self.assertEqual("evidence_bridge", candidate["inference_type"])
        self.assertEqual([7, 8], candidate["premise_episode_ids"])
        self.assertIn("不", candidate["claim"])
        self.assertIn("因果", candidate["claim"])

    def test_coverage_bridge_abstains_when_any_aspect_is_missing(self):
        raw_result = {
            "domains": {
                "knowledge": {
                    "rerank_trace": {
                        "merged_coverage": {
                            "coverage": [
                                {"query": "已覆盖", "episode_ids": [7]},
                                {"query": "另一槽", "episode_ids": [8]},
                            ],
                            "missing_aspects": ["尚缺直接证据"],
                        }
                    },
                    "evidence_episodes": [],
                }
            }
        }

        self.assertIsNone(
            derive_evidence_bridge_candidate("综合说明", raw_result)
        )

    def test_coverage_bridge_keeps_two_proven_slots_when_optional_detail_is_missing(self):
        raw_result = {
            "domains": {
                "knowledge": {
                    "rerank_trace": {
                        "merged_coverage": {
                            "coverage": [
                                {"query": "谁执行袭击", "episode_ids": [7]},
                                {"query": "谁提供合作背景", "episode_ids": [8]},
                                {"query": "额外细节", "episode_ids": []},
                            ],
                            "missing_aspects": ["可选的额外细节"],
                        }
                    },
                    "evidence_episodes": [
                        {
                            "id": 7,
                            "text": "甲执行袭击。",
                            "generation": 0,
                            "evidence_origin": "source",
                        },
                        {
                            "id": 8,
                            "text": "乙参与合作。",
                            "generation": 0,
                            "evidence_origin": "source",
                        },
                    ],
                }
            }
        }

        candidate = derive_evidence_bridge_candidate("综合说明", raw_result)

        self.assertIsNotNone(candidate)
        self.assertEqual([7, 8], candidate["premise_episode_ids"])
        self.assertIn("explicit reranker coverage", candidate["reason"])

    def test_planned_slots_derive_bridge_when_cross_encoder_has_no_coverage_map(self):
        raw_result = {
            "intent_planning": {
                "raw_plan": {"answer_slots": ["袭击者", "合作名称"]}
            },
            "domains": {
                "knowledge": {
                    "rerank_trace": {"backend": "cross_encoder"},
                    "evidence_episodes": [
                        {
                            "id": 7,
                            "text": "某组织袭击了条约现场。",
                            "generation": 0,
                            "evidence_origin": "source",
                        },
                        {
                            "id": 8,
                            "text": "某成员曾与该组织合作。",
                            "generation": 0,
                            "evidence_origin": "source",
                        },
                    ],
                }
            },
        }

        candidate = derive_evidence_bridge_candidate("综合两个槽", raw_result)

        self.assertIsNotNone(candidate)
        self.assertEqual([7, 8], candidate["premise_episode_ids"])
        self.assertIn("local lexical proof", candidate["reason"])

    def test_footer_accepts_a_bounded_evidence_bridge(self):
        raw_result = {
            "domains": {
                "knowledge": {
                    "evidence_episodes": [
                        {"id": 7, "text": "甲记录合作"},
                        {"id": 8, "text": "乙记录袭击"},
                    ]
                }
            }
        }
        footer = {
            "k": [
                {
                    "c": (
                        "一端记录合作，另一端记录袭击；只用于共同检索，"
                        "不主张合作导致袭击。"
                    ),
                    "e": [7, 8],
                    "t": "evidence_bridge",
                    "q": 0.82,
                }
            ],
            "p": [],
        }
        response = (
            "自然回答<assistant_memory>"
            + json.dumps(footer, ensure_ascii=False)
            + "</assistant_memory>"
        )

        visible, decision = split_inline_memory_response(response, raw_result)

        self.assertEqual("自然回答", visible)
        self.assertEqual(1, len(decision["knowledge_candidates"]))
        self.assertEqual(
            "evidence_bridge",
            decision["knowledge_candidates"][0]["inference_type"],
        )

    def test_footer_is_hidden_and_only_visible_episode_ids_are_accepted(self):
        raw_result = {
            "domains": {
                "knowledge": {
                    "evidence_episodes": [
                        {"id": 7, "text": "前因证据"},
                        {"id": 8, "text": "后果证据"},
                    ]
                }
            }
        }
        footer = {
            "knowledge_candidates": [
                {
                    "claim": "有证据的新推论",
                    "premise_episode_ids": [7, 8],
                    "inference_type": "causal",
                    "confidence": 0.8,
                    "reason": "跨节点解释",
                },
                {
                    "claim": "伪造编号的推论",
                    "premise_episode_ids": [7, 999],
                    "inference_type": "causal",
                    "confidence": 1.0,
                    "reason": "无",
                },
            ],
            "private_candidates": [
                {"claim": "白子刚给老师发来消息", "reason": "本轮创造"}
            ],
        }
        response = (
            "给老师看的回答。\n<assistant_memory>"
            + json.dumps(footer, ensure_ascii=False)
            + "</assistant_memory>"
        )
        visible, decision = split_inline_memory_response(response, raw_result)
        self.assertEqual("给老师看的回答。", visible)
        self.assertNotIn("assistant_memory", visible)
        self.assertEqual(1, len(decision["knowledge_candidates"]))
        self.assertEqual(
            [7, 8],
            list(decision["knowledge_candidates"][0]["premise_episode_ids"]),
        )
        self.assertEqual(1, len(decision["private_candidates"]))
        self.assertIn("伪造编号", decision["rejected_claims"][0]["claim"])
        self.assertIn("有证据的新推论", decision["knowledge_query"])

    def test_local_gate_rejects_restated_and_compound_shiroko_candidates(self):
        raw_result = {
            "domains": {
                "knowledge": {
                    "evidence_episodes": [
                        {"id": 2249, "text": "白子谈论新技术与传统钢索。"},
                        {"id": 2324, "text": "老师说自己是白子的支持者。"},
                        {"id": 2325, "text": "白子珍视老师的支持。"},
                        {"id": 2326, "text": "白子因支持而心跳加速。"},
                    ]
                }
            }
        }
        footer = {
            "k": [
                {
                    "c": "白子因老师的支持而心跳加速",
                    "e": [2324, 2326],
                    "t": "direct_fact",
                    "q": 0.85,
                },
                {
                    "c": "白子对技术变革持开放但怀旧态度，且极度珍视支持",
                    "e": [2249, 2325],
                    "t": "trait",
                    "q": 0.9,
                },
            ],
            "p": [],
        }
        response = (
            "自然回答<assistant_memory>"
            + json.dumps(footer, ensure_ascii=False)
            + "</assistant_memory>"
        )
        visible, decision = split_inline_memory_response(response, raw_result)
        self.assertEqual("自然回答", visible)
        self.assertEqual([], decision["knowledge_candidates"])
        self.assertEqual("", decision["knowledge_query"])
        self.assertEqual(2, len(decision["rejected_claims"]))
        self.assertIn("non-inferential", decision["rejected_claims"][0]["reason"])
        self.assertIn("non-atomic", decision["rejected_claims"][1]["reason"])


class RecallCacheTests(unittest.IsolatedAsyncioTestCase):
    async def test_query_embedding_cache_survives_recall_cache_miss(self):
        class RetrievalConfig:
            rerank_enabled = False
            rerank_backend = "llm"
            answer_episode_limit = 30
            answer_concept_limit = 20
            answer_path_limit = 24

        class QueryConfig:
            retrieval = RetrievalConfig()

        class FakeQueryEngine:
            config = QueryConfig()

            def __init__(self):
                self.calls = []
                self.last_query_embeddings = {}

            def query(self, question, generate_answer=False, **kwargs):
                self.calls.append(kwargs.get("query_embeddings_override") or {})
                self.last_query_embeddings = {
                    question: np.array([1.0, 0.0], dtype=np.float32)
                }
                return {
                    "question": question,
                    "evidence_episodes": [
                        {
                            "id": 1,
                            "score": 0.9,
                            "text": "缓存证据",
                            "source_key": "source.json",
                        }
                    ],
                }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service = AssociativeMemoryService(
                MemoryServiceConfig(
                    engine_root=root,
                    database_path=root / "memory.db",
                    log_dir=root / "logs",
                    api_key="test",
                    recall_cache_size=0,
                    query_embedding_cache_size=8,
                    growth_enabled=False,
                )
            )
            engine = FakeQueryEngine()
            service._query_engine = engine
            service._application = object()
            service._has_memories = True

            await service.recall("相同检索计划")
            await service.recall("相同检索计划")

            self.assertEqual({}, engine.calls[0])
            self.assertIn("相同检索计划", engine.calls[1])
            np.testing.assert_array_equal(
                np.array([1.0, 0.0], dtype=np.float32),
                engine.calls[1]["相同检索计划"],
            )

    async def test_standard_automatically_escalates_when_evidence_is_empty(self):
        class RetrievalConfig:
            def __init__(self):
                self.graph_max_hops = 3
                self.rerank_precompression_limit = 24
                self.answer_episode_limit = 30
                self.answer_concept_limit = 20
                self.answer_path_limit = 24
                self.rerank_atomic_floor_enabled = True
                self.followup_planning_mode = "entity_resolved"
                self.rerank_enabled = True
                self.rerank_backend = "cross_encoder"
                self.rerank_review_mode = "fast_adaptive"
                self.rerank_atomic_query_limit = 12

        class QueryConfig:
            def __init__(self):
                self.retrieval = RetrievalConfig()

        class TemplateEngine:
            config = QueryConfig()

        class RequestEngine:
            def __init__(self, config):
                self.config = config

            def query(self, question, generate_answer=False, **kwargs):
                if self.config.retrieval.rerank_backend != "llm":
                    return {"question": question, "evidence_episodes": []}
                return {
                    "question": question,
                    "evidence_episodes": [
                        {
                            "id": 9,
                            "score": 0.9,
                            "text": "deep 补齐的证据",
                            "source_key": "main/deep.json",
                        }
                    ],
                }

        class FakeApplication:
            def query_engine(self, *, config):
                return RequestEngine(config)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service = AssociativeMemoryService(
                MemoryServiceConfig(
                    engine_root=root,
                    database_path=root / "memory.db",
                    log_dir=root / "logs",
                    api_key="test",
                    reranker_model="Pro/BAAI/bge-reranker-v2-m3",
                    rerank_enabled=True,
                    rerank_backend="cross_encoder",
                    growth_enabled=False,
                )
            )
            service._application = FakeApplication()
            service._query_engine = TemplateEngine()
            service._has_memories = True

            result = await service.recall("普通剧情事实问题")

            self.assertEqual("deep", result.raw_result["retrieval_intensity"])
            self.assertTrue(result.raw_result["retrieval_escalation"]["triggered"])
            self.assertEqual(
                ["no_evidence"],
                result.raw_result["retrieval_escalation"]["reasons"],
            )

    async def test_request_local_plans_allow_foreground_queries_to_overlap(self):
        barrier = Barrier(2)

        class RetrievalConfig:
            graph_max_hops = 3
            rerank_precompression_limit = 24
            answer_episode_limit = 30
            answer_concept_limit = 20
            answer_path_limit = 24
            rerank_atomic_floor_enabled = True
            followup_planning_mode = "entity_resolved"
            rerank_enabled = True
            rerank_backend = "cross_encoder"
            rerank_review_mode = "fast_adaptive"
            rerank_atomic_query_limit = 12

        class QueryConfig:
            def __init__(self):
                self.retrieval = RetrievalConfig()

        class TemplateEngine:
            config = QueryConfig()

        class RequestEngine:
            def __init__(self, config):
                self.config = config

            def query(self, question, generate_answer=False, **kwargs):
                barrier.wait(timeout=2.0)
                return {
                    "question": question,
                    "evidence_episodes": [
                        {
                            "id": 1,
                            "score": 0.8,
                            "text": "并发读取证据",
                            "source_key": "main/example.json",
                        }
                    ],
                }

        class FakeApplication:
            def __init__(self):
                self.configs = []

            def query_engine(self, *, config):
                self.configs.append(config)
                return RequestEngine(config)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service = AssociativeMemoryService(
                MemoryServiceConfig(
                    engine_root=root,
                    database_path=root / "memory.db",
                    log_dir=root / "logs",
                    api_key="test",
                    reranker_model="Pro/BAAI/bge-reranker-v2-m3",
                    rerank_enabled=True,
                    rerank_backend="cross_encoder",
                    growth_enabled=False,
                )
            )
            application = FakeApplication()
            service._application = application
            service._query_engine = TemplateEngine()
            service._has_memories = True

            await asyncio.gather(
                service.recall(
                    "轻量问题",
                    retrieval_plan=service.retrieval_plan("light"),
                ),
                service.recall(
                    "请完整推导复杂问题",
                    retrieval_plan=service.retrieval_plan("deep"),
                ),
            )

            self.assertEqual(2, len(application.configs))
            self.assertIsNot(application.configs[0], application.configs[1])
            self.assertEqual(
                {1, 3},
                {
                    config.retrieval.graph_max_hops
                    for config in application.configs
                },
            )
            self.assertEqual(
                "cross_encoder",
                service._query_engine.config.retrieval.rerank_backend,
            )

    async def test_light_recall_uses_cross_encoder_without_llm_planning(self):
        class RetrievalConfig:
            rerank_enabled = True
            rerank_backend = "cross_encoder"
            answer_episode_limit = 30
            answer_concept_limit = 20
            answer_path_limit = 24

        class QueryConfig:
            retrieval = RetrievalConfig()

        class FakeQueryEngine:
            config = QueryConfig()

            def query(self, question, generate_answer=False, **kwargs):
                self.rerank_during_query = self.config.retrieval.rerank_enabled
                self.backend_during_query = self.config.retrieval.rerank_backend
                self.options = kwargs
                self.budgets_during_query = (
                    self.config.retrieval.answer_episode_limit,
                    self.config.retrieval.answer_concept_limit,
                    self.config.retrieval.answer_path_limit,
                )
                return {
                    "question": question,
                    "evidence_episodes": [
                        {"id": index, "text": f"episode {index}"}
                        for index in range(20)
                    ],
                    "evidence_concepts": [],
                    "association_paths": [],
                    "rerank_trace": {
                        "backend": "cross_encoder",
                        "model": "Pro/BAAI/bge-reranker-v2-m3",
                        "review_level": "cross_encoder",
                    },
                }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service = AssociativeMemoryService(
                MemoryServiceConfig(
                    engine_root=root,
                    database_path=root / "memory.db",
                    log_dir=root / "logs",
                    api_key="test",
                )
            )
            engine = FakeQueryEngine()
            service._query_engine = engine
            service._application = object()
            service._has_memories = True
            service._small_database_fast_path = False

            result = await service.recall(
                "你觉得白子是一个什么样的人？",
                retrieval_intensity="light",
            )

            self.assertTrue(engine.rerank_during_query)
            self.assertEqual("cross_encoder", engine.backend_during_query)
            self.assertTrue(engine.config.retrieval.rerank_enabled)
            self.assertEqual(
                "cross_encoder_light", result.raw_result["rerank_route"]
            )
            self.assertEqual(
                "completed",
                result.raw_result["rerank_execution"]["status"],
            )
            self.assertEqual(
                "cross_encoder",
                result.raw_result["rerank_execution"]["backend"],
            )
            self.assertEqual([], engine.options["followup_queries_override"])
            self.assertEqual((12, 10, 10), engine.budgets_during_query)
            self.assertEqual(30, engine.config.retrieval.answer_episode_limit)
            self.assertEqual(20, engine.config.retrieval.answer_concept_limit)
            self.assertEqual(24, engine.config.retrieval.answer_path_limit)
            self.assertEqual(
                [], engine.options["intent_override"]["target_entities"]
            )
            self.assertEqual(12, len(result.raw_result["evidence_episodes"]))
            self.assertTrue(result.raw_result["lightweight_fast_path"])

    async def test_deep_recall_uses_llm_default_and_restores_cross_encoder(self):
        class RetrievalConfig:
            rerank_enabled = True
            rerank_backend = "cross_encoder"
            rerank_review_mode = "fast_adaptive"
            followup_planning_mode = "entity_resolved"
            rerank_atomic_query_limit = 12
            rerank_precompression_limit = 24

        class QueryConfig:
            retrieval = RetrievalConfig()

        class FakeQueryEngine:
            config = QueryConfig()

            def __init__(self):
                self.calls = []

            def query(self, question, generate_answer=False, **kwargs):
                self.calls.append(
                    (
                        self.config.retrieval.rerank_enabled,
                        self.config.retrieval.rerank_backend,
                        self.config.retrieval.rerank_review_mode,
                        self.config.retrieval.followup_planning_mode,
                        self.config.retrieval.rerank_atomic_query_limit,
                        self.config.retrieval.rerank_precompression_limit,
                    )
                )
                if "触发失败" in question:
                    raise RuntimeError("synthetic deep failure")
                return {
                    "question": question,
                    "evidence_episodes": [
                        {
                            "id": 1,
                            "score": 0.8,
                            "text": "星野的经历证据",
                            "source_key": "main/example.json",
                        }
                    ],
                }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service = AssociativeMemoryService(
                MemoryServiceConfig(
                    engine_root=root,
                    database_path=root / "memory.db",
                    log_dir=root / "logs",
                    api_key="test",
                    reranker_model="Pro/BAAI/bge-reranker-v2-m3",
                    rerank_enabled=True,
                    rerank_backend="cross_encoder",
                )
            )
            engine = FakeQueryEngine()
            service._query_engine = engine
            service._application = object()
            service._has_memories = True
            service._small_database_fast_path = False

            standard = await service.recall(
                "星野的经历", retrieval_intensity="standard"
            )
            deep = await service.recall(
                "请综合分析星野经历背后的因果关系",
                retrieval_intensity="deep",
            )

            self.assertEqual(
                [
                    (
                        True,
                        "cross_encoder",
                        "fast_adaptive",
                        "entity_resolved",
                        12,
                        24,
                    ),
                    (True, "llm", "lean", "always", 12, 12),
                ],
                engine.calls,
            )
            self.assertEqual(
                "cross_encoder_standard",
                standard.raw_result["rerank_route"],
            )
            self.assertEqual(
                "llm_deep_default", deep.raw_result["rerank_route"]
            )
            self.assertTrue(engine.config.retrieval.rerank_enabled)
            self.assertEqual(
                "cross_encoder", engine.config.retrieval.rerank_backend
            )
            self.assertEqual(
                "fast_adaptive",
                engine.config.retrieval.rerank_review_mode,
            )
            self.assertEqual(
                "entity_resolved",
                engine.config.retrieval.followup_planning_mode,
            )
            self.assertEqual(12, engine.config.retrieval.rerank_atomic_query_limit)
            self.assertEqual(
                24, engine.config.retrieval.rerank_precompression_limit
            )

            engine.config.retrieval.rerank_enabled = False
            disabled = await service.recall(
                "请综合分析另一条因果链",
                retrieval_intensity="deep",
            )
            self.assertEqual(
                (
                    False,
                    "cross_encoder",
                    "fast_adaptive",
                    "entity_resolved",
                    12,
                    24,
                ),
                engine.calls[-1],
            )
            self.assertEqual(
                "disabled_not_configured",
                disabled.raw_result["rerank_route"],
            )

            engine.config.retrieval.rerank_enabled = True
            failed = await service.recall(
                "请综合分析并触发失败",
                retrieval_intensity=" deep ",
            )
            self.assertIn("synthetic deep failure", failed.error)
            self.assertTrue(engine.config.retrieval.rerank_enabled)
            self.assertEqual(
                "cross_encoder", engine.config.retrieval.rerank_backend
            )
            self.assertEqual(
                "fast_adaptive",
                engine.config.retrieval.rerank_review_mode,
            )
            self.assertEqual(
                "entity_resolved",
                engine.config.retrieval.followup_planning_mode,
            )
            self.assertEqual(12, engine.config.retrieval.rerank_atomic_query_limit)
            self.assertEqual(
                24, engine.config.retrieval.rerank_precompression_limit
            )

    async def test_cancelled_deep_recall_waits_for_thread_before_restore(self):
        started = Event()
        release = Event()

        class RetrievalConfig:
            rerank_enabled = True
            rerank_backend = "cross_encoder"
            rerank_review_mode = "fast_adaptive"
            followup_planning_mode = "entity_resolved"
            rerank_atomic_query_limit = 12
            rerank_precompression_limit = 24

        class QueryConfig:
            retrieval = RetrievalConfig()

        class FakeQueryEngine:
            config = QueryConfig()

            def query(self, question, generate_answer=False, **kwargs):
                self.during_query = (
                    self.config.retrieval.rerank_backend,
                    self.config.retrieval.rerank_review_mode,
                    self.config.retrieval.followup_planning_mode,
                    self.config.retrieval.rerank_atomic_query_limit,
                    self.config.retrieval.rerank_precompression_limit,
                )
                started.set()
                release.wait(timeout=2.0)
                return {"question": question, "evidence_episodes": []}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service = AssociativeMemoryService(
                MemoryServiceConfig(
                    engine_root=root,
                    database_path=root / "memory.db",
                    log_dir=root / "logs",
                    api_key="test",
                )
            )
            engine = FakeQueryEngine()
            service._query_engine = engine
            service._application = object()
            service._has_memories = True
            service._small_database_fast_path = False

            task = asyncio.create_task(
                service.recall("综合分析因果", retrieval_intensity="deep")
            )
            self.assertTrue(await asyncio.to_thread(started.wait, 1.0))
            task.cancel()
            await asyncio.sleep(0)
            self.assertEqual("llm", engine.config.retrieval.rerank_backend)
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task

            self.assertEqual(
                ("llm", "lean", "always", 12, 12),
                engine.during_query,
            )
            self.assertEqual(
                "cross_encoder", engine.config.retrieval.rerank_backend
            )
            self.assertEqual(
                "fast_adaptive",
                engine.config.retrieval.rerank_review_mode,
            )

    async def test_small_database_fast_path_uses_deterministic_plan_and_cache(self):
        class FakeQueryEngine:
            def __init__(self):
                self.calls = 0

            def query(self, question, generate_answer=False, **kwargs):
                self.calls += 1
                self.last_options = kwargs
                return {"question": question, "evidence_episodes": []}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service = AssociativeMemoryService(
                MemoryServiceConfig(
                    engine_root=root,
                    database_path=root / "memory.db",
                    log_dir=root / "logs",
                    api_key="test",
                    recall_cache_size=2,
                    small_database_fast_path_max_episodes=32,
                )
            )
            query_engine = FakeQueryEngine()
            service._query_engine = query_engine
            service._application = object()
            service._has_memories = True
            service._small_database_fast_path = True

            first = await service.recall("老师最喜欢什么饮料？")
            second = await service.recall("老师最喜欢什么饮料？")

            self.assertEqual(1, query_engine.calls)
            self.assertEqual(
                ["老师最喜欢什么饮料？"],
                query_engine.last_options["intent_override"]["search_queries"],
            )
            self.assertEqual(
                [], query_engine.last_options["followup_queries_override"]
            )
            self.assertTrue(first.raw_result["small_database_fast_path"])
            self.assertTrue(second.raw_result["service_cache_hit"])

    async def test_successful_recall_is_cached_and_graph_refresh_invalidates_it(self):
        class FakeAssociations:
            _concept_reach_cache = {}

        class FakeApplication:
            associations = FakeAssociations()

        class FakeQueryEngine:
            def __init__(self):
                self.calls = 0

            def query(self, question, generate_answer=False, **kwargs):
                self.calls += 1
                self.last_options = kwargs
                return {
                    "question": question,
                    "evidence_episodes": [
                        {
                            "id": self.calls,
                            "score": 0.8,
                            "text": "固定证据",
                            "source_key": "memory/example.txt",
                        }
                    ],
                    "episode_ids": [self.calls],
                }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = MemoryServiceConfig(
                engine_root=root,
                database_path=root / "memory.db",
                log_dir=root / "logs",
                api_key="test",
                recall_cache_size=2,
            )
            service = AssociativeMemoryService(config)
            query_engine = FakeQueryEngine()
            service._application = FakeApplication()
            service._query_engine = query_engine
            service._has_memories = True

            first = await service.recall(" 同一个   问题 ")
            second = await service.recall("同一个 问题")
            self.assertEqual(1, query_engine.calls)
            self.assertFalse(first.raw_result["service_cache_hit"])
            self.assertTrue(second.raw_result["service_cache_hit"])

            await service.refresh_graph_state()
            third = await service.recall("同一个 问题")
            self.assertEqual(2, query_engine.calls)
            self.assertFalse(third.raw_result["service_cache_hit"])

            override = {"search_queries": ["固定查询"]}
            await service.recall(
                "同一个 问题",
                intent_override=override,
                followup_queries_override=["固定后续查询"],
            )
            await service.recall(
                "同一个 问题",
                intent_override=override,
                followup_queries_override=["固定后续查询"],
            )
            self.assertEqual(4, query_engine.calls)
            self.assertEqual(override, query_engine.last_options["intent_override"])
            self.assertEqual(
                ["固定后续查询"],
                query_engine.last_options["followup_queries_override"],
            )

    async def test_deep_failure_is_briefly_cached_and_graph_refresh_retries(self):
        class FakeAssociations:
            _concept_reach_cache = {}

        class FakeApplication:
            associations = FakeAssociations()

        class FailingQueryEngine:
            def __init__(self):
                self.calls = 0

            def query(self, question, generate_answer=False, **kwargs):
                self.calls += 1
                raise TimeoutError("deep evidence review timed out")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service = AssociativeMemoryService(
                MemoryServiceConfig(
                    engine_root=root,
                    database_path=root / "memory.db",
                    log_dir=root / "logs",
                    api_key="test",
                    recall_cache_size=2,
                    recall_failure_ttl_seconds=60,
                    growth_enabled=False,
                )
            )
            query_engine = FailingQueryEngine()
            service._application = FakeApplication()
            service._query_engine = query_engine
            service._has_memories = True

            first = await service.recall(
                "同一个复杂问题",
                retrieval_intensity="deep",
                auto_escalate=False,
            )
            second = await service.recall(
                "同一个复杂问题",
                retrieval_intensity="deep",
                auto_escalate=False,
            )

            self.assertEqual(1, query_engine.calls)
            self.assertTrue(first.error)
            self.assertTrue(second.raw_result["service_cache_hit"])
            self.assertTrue(second.raw_result["failure_cache_hit"])

            await service.refresh_graph_state()
            await service.recall(
                "同一个复杂问题",
                retrieval_intensity="deep",
                auto_escalate=False,
            )
            self.assertEqual(2, query_engine.calls)

    async def test_graph_refresh_replaces_association_route_matcher(self):
        class FakeAssociations:
            _concept_reach_cache = {}

        class FakeApplication:
            associations = FakeAssociations()

            def __init__(self):
                self.version = 0

            def rebuild_association_cue_index(self):
                self.version += 1

            def new_logger(self, _operation):
                return None

            def query_engine(self, _logger):
                return {"cue_index_version": self.version}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service = AssociativeMemoryService(
                MemoryServiceConfig(
                    engine_root=root,
                    database_path=root / "memory.db",
                    log_dir=root / "logs",
                    api_key="test",
                    association_cue_enabled=True,
                )
            )
            application = FakeApplication()
            service._application = application
            service._query_engine = {"cue_index_version": 0}
            service._has_memories = True

            await service.refresh_graph_state()

            self.assertEqual(1, application.version)
            self.assertEqual(
                {"cue_index_version": 1}, service._query_engine
            )


class SessionTests(unittest.TestCase):
    def test_context_is_isolated_by_platform_identity(self):
        sessions = ConversationSessionBuffer(max_messages=2)
        sessions.add("telegram:42", "user", "private A")
        sessions.add("discord:42", "user", "private B")
        self.assertEqual("private A", sessions.get("telegram:42")[0].content)
        self.assertEqual("private B", sessions.get("discord:42")[0].content)


class JournalTests(unittest.IsolatedAsyncioTestCase):
    async def test_journal_records_stable_identity_and_rotates_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = ConversationJournal(directory, batch_exchanges=1)
            identity = PlatformIdentity("telegram", "42", "Alice")
            ready = await journal.record_exchange(identity, "hello", "hi")
            self.assertIsNotNone(ready)
            text = ready.read_text(encoding="utf-8")
            metadata = json.loads((ready.parent / "identity.json").read_text("utf-8"))
            self.assertIn('"platform":"telegram"', text)
            self.assertIn('"platform_user_id":"42"', text)
            self.assertIn('"origin":"source"', text)
            self.assertIn('"origin":"system"', text)
            self.assertIn('"status":"mixed"', text)
            self.assertIn('"role":"assistant"', text)
            self.assertIn('"content":"hello"', text)
            self.assertEqual("Alice", metadata["display_name"])

    async def test_worker_uses_same_pipeline_and_keeps_source_receipt(self):
        imported: list[tuple[Path, Path]] = []

        async def fake_import(path: Path, root: Path) -> dict:
            imported.append((path, root))
            return {"failed_tasks": 0, "failed_paragraph_sources": 0, "episodes": 1}

        with tempfile.TemporaryDirectory() as directory:
            journal = ConversationJournal(directory, batch_exchanges=1)
            identity = PlatformIdentity("telegram", "42", "Alice")
            ready = await journal.record_exchange(identity, "hello", "hi")
            worker = ConversationIngestionWorker(journal, fake_import)
            await worker.start()
            await worker._queue.join()
            await worker.close(flush_pending=False)
            self.assertEqual(1, worker.imported_batches)
            self.assertEqual(1, len(imported))
            self.assertTrue(any(ready.parent.glob("*.source.txt")))
            self.assertTrue(any(ready.parent.glob("*.imported.json")))


class GrowthWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_creative_roleplay_waits_for_private_journal_import(self):
        calls = []

        async def fake_grow(identity, question, route):
            calls.append(route)
            return {"domains": {}}

        with tempfile.TemporaryDirectory() as directory:
            worker = BackgroundGrowthWorker(directory, fake_grow, min_chars=8)
            await worker.start()
            question = (
                "请结合今天的学园安排，告诉我现在有哪些出差任务，以及有哪些学生"
                "给为师发消息了，并说明各自最需要处理的事情。"
            )
            route = MemoryRoute(
                True,
                True,
                True,
                "model_intent_planner",
                "light",
                "evidence_gated",
                True,
            )
            path = await worker.enqueue(
                PlatformIdentity("synthetic", "teacher"),
                question,
                route,
            )
            self.assertIsNone(path)
            await worker.close()

            self.assertEqual([], calls)
            self.assertFalse(any(Path(directory).glob("*.completed.json")))

    async def test_precomputed_local_rejection_skips_growth_call(self):
        calls = []

        async def fake_grow(identity, question, route):
            calls.append(route)
            return {"domains": {}}

        with tempfile.TemporaryDirectory() as directory:
            worker = BackgroundGrowthWorker(directory, fake_grow, min_chars=8)
            await worker.start()
            path = await worker.enqueue(
                PlatformIdentity("synthetic", "teacher"),
                "你觉得白子是一个什么样的人？",
                MemoryRoute(
                    False,
                    False,
                    True,
                    "indirect_lore",
                    knowledge_write_policy="evidence_gated",
                ),
                assistant_response="白子很重视老师的支持。",
                consolidation_decision={
                    "knowledge_candidates": [],
                    "private_candidates": [],
                    "rejected_claims": [
                        {
                            "claim": "白子因支持而心跳加速",
                            "reason": "direct fact",
                        }
                    ],
                    "knowledge_query": "",
                },
            )
            await worker.close()

            self.assertIsNone(path)
            self.assertEqual([], calls)
            self.assertFalse(any(Path(directory).glob("*.pending.json")))

    async def test_precomputed_evidence_bound_answer_can_enable_lore_growth(self):
        calls = []

        async def fake_grow(
            identity,
            question,
            route,
            *,
            user_question=None,
            knowledge_question=None,
            knowledge_candidates=None,
        ):
            calls.append(
                (
                    route,
                    user_question,
                    knowledge_question,
                    knowledge_candidates,
                )
            )
            return {"domains": {"knowledge": {"new_association_ids": [9]}}}

        class FakeMemory:
            raw_result = {
                "domains": {
                    "knowledge": {
                        "evidence_episodes": [
                            {
                                "id": 17,
                                "source_key": "main/example.json",
                                "text": "白子为了同伴采取行动。",
                            },
                            {
                                "id": 18,
                                "source_key": "main/example-2.json",
                                "text": "白子在危险中优先保护同伴。",
                            },
                        ]
                    }
                }
            }

        with tempfile.TemporaryDirectory() as directory:
            worker = BackgroundGrowthWorker(
                directory,
                fake_grow,
                min_chars=8,
            )
            await worker.start()
            route = route_memory_query("有哪些学生给为师发消息了？")
            path = await worker.enqueue(
                PlatformIdentity("synthetic", "teacher"),
                "有哪些学生给为师发消息了？",
                route,
                memory=FakeMemory(),
                consolidation_decision={
                    "knowledge_candidates": [
                        {
                            "claim": "白子的行动体现出对同伴的重视",
                            "premise_episode_ids": [17, 18],
                            "inference_type": "trait",
                            "confidence": 0.8,
                            "reason": "剧情行动支持稳定人物倾向",
                        }
                    ],
                    "private_candidates": [
                        {"claim": "白子今天发来消息", "reason": "当前互动"}
                    ],
                    "rejected_claims": [],
                    "knowledge_query": "验证候选推论：白子的行动体现出对同伴的重视",
                },
            )
            self.assertIsNotNone(path)
            await worker._queue.join()
            await worker.close()

            self.assertEqual(1, len(calls))
            self.assertTrue(calls[0][0].knowledge)
            self.assertIn("白子的行动", calls[0][2])
            self.assertEqual([17, 18], calls[0][3][0]["premise_episode_ids"])
            receipt = json.loads(
                next(Path(directory).glob("*.completed.json")).read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(receipt["effective_route"]["knowledge"])
            self.assertEqual(
                "白子今天发来消息",
                receipt["consolidation"]["private_candidates"][0]["claim"],
            )

    async def test_completed_lore_candidate_is_deduplicated_after_restart(self):
        calls = []

        async def fake_grow(
            identity,
            question,
            route,
            *,
            user_question=None,
            knowledge_question=None,
        ):
            calls.append(knowledge_question)
            return {"domains": {"knowledge": {"new_association_ids": []}}}

        decision = {
            "knowledge_candidates": [
                {
                    "claim": "前一事件促使角色改变后续选择",
                    "premise_episode_ids": [41, 52],
                    "inference_type": "causal",
                    "confidence": 0.84,
                }
            ],
            "private_candidates": [],
            "rejected_claims": [],
            "knowledge_query": "验证候选推论：前一事件促使角色改变后续选择",
        }
        with tempfile.TemporaryDirectory() as directory:
            route = MemoryRoute(
                False,
                False,
                True,
                "dedup_test",
                knowledge_write_policy="evidence_gated",
            )
            identity = PlatformIdentity("synthetic", "teacher")
            first_worker = BackgroundGrowthWorker(directory, fake_grow, min_chars=8)
            await first_worker.start()
            first = await first_worker.enqueue(
                identity,
                "第一次问法",
                route,
                assistant_response="第一次回答",
                consolidation_decision=decision,
            )
            self.assertIsNotNone(first)
            await first_worker._queue.join()
            await first_worker.close()

            second_worker = BackgroundGrowthWorker(directory, fake_grow, min_chars=8)
            await second_worker.start()
            duplicate = await second_worker.enqueue(
                identity,
                "完全不同的第二种问法",
                route,
                assistant_response="不同措辞的回答",
                consolidation_decision=decision,
            )
            await second_worker.close()

            self.assertIsNone(duplicate)
            self.assertEqual(1, len(calls))

    async def test_running_job_is_recovered_after_restart(self):
        calls = []

        async def fake_grow(identity, question, route):
            calls.append((identity, question, route))
            return {"domains": {"knowledge": {"new_association_ids": []}}}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = {
                "job_id": "recovered-job",
                "created_at": "2026-08-29T00:00:00+00:00",
                "fingerprint": "recovered-fingerprint",
                "platform": "telegram",
                "platform_user_id": "42",
                "display_name": "Alice",
                "question": "为什么这两段长期记忆之间存在联系？",
                "user": False,
                "knowledge": True,
                "reason": "restart_test",
            }
            (root / "recovered-job.running.json").write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
            worker = BackgroundGrowthWorker(root, fake_grow, min_chars=8)
            await worker.start()
            await worker._queue.join()
            await worker.close()

            self.assertEqual(1, len(calls))
            self.assertEqual(1, worker.completed_jobs)
            self.assertFalse(any(root.glob("*.running.json")))
            self.assertEqual(1, len(list(root.glob("*.completed.json"))))

    async def test_failed_growth_writes_a_failure_receipt(self):
        async def fake_grow(identity, question, route):
            raise RuntimeError("synthetic growth failure")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worker = BackgroundGrowthWorker(root, fake_grow, min_chars=8)
            await worker.start()
            path = await worker.enqueue(
                PlatformIdentity("telegram", "42"),
                "为什么这两段长期记忆之间存在联系？",
                MemoryRoute(False, False, True, "failure_test"),
            )
            self.assertIsNotNone(path)
            await worker._queue.join()
            await worker.close()

            self.assertEqual(1, worker.failed_jobs)
            self.assertIn("synthetic growth failure", worker.last_error)
            failed = list(root.glob("*.failed.json"))
            self.assertEqual(1, len(failed))
            receipt = json.loads(failed[0].read_text(encoding="utf-8"))
            self.assertIn("synthetic growth failure", receipt["error"])

    async def test_growth_queue_is_persistent_deduplicated_and_public_read_only(self):
        calls = []

        async def fake_grow(identity, question, route):
            calls.append((identity, question, route))
            return {"domains": {"user": {"new_association_ids": [7]}}}

        with tempfile.TemporaryDirectory() as directory:
            worker = BackgroundGrowthWorker(directory, fake_grow, min_chars=8)
            await worker.start()
            identity = PlatformIdentity("telegram", "42", "Alice")
            route = MemoryRoute(True, True, True, "test")
            question = "为什么我会把这段私人经历和剧情人物联系起来？"
            first = await worker.enqueue(identity, question, route)
            duplicate = await worker.enqueue(identity, question, route)
            self.assertIsNotNone(first)
            self.assertIsNone(duplicate)
            await worker._queue.join()
            await worker.close()

            self.assertEqual(1, len(calls))
            self.assertTrue(calls[0][2].user)
            self.assertTrue(calls[0][2].knowledge)
            self.assertFalse(calls[0][2].public)
            self.assertEqual(1, worker.completed_jobs)
            self.assertEqual(1, len(list(Path(directory).glob("*.completed.json"))))

    async def test_supplied_empty_consolidation_skips_duplicate_deep_growth(self):
        calls = []

        async def fake_grow(
            identity,
            question,
            route,
            *,
            user_question=None,
            knowledge_question=None,
            knowledge_candidates=None,
        ):
            calls.append((route, knowledge_question, knowledge_candidates))
            return {"domains": {"knowledge": {"new_association_ids": [31]}}}

        memory = RetrievedMemory(
            context="",
            raw_result={
                "domains": {
                    "knowledge": {
                        "retrieval_escalation": {"triggered": True},
                        "evidence_episodes": [
                            {"id": 10, "text": "第一段证据"},
                            {"id": 20, "text": "第二段证据"},
                        ],
                    }
                }
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            worker = BackgroundGrowthWorker(directory, fake_grow, min_chars=8)
            await worker.start()
            question = "这两段事件为什么会共同影响角色后来的选择？"
            path = await worker.enqueue(
                PlatformIdentity("synthetic", "teacher"),
                question,
                MemoryRoute(
                    False,
                    False,
                    True,
                    "model_intent_planner",
                    "standard",
                    "evidence_gated",
                ),
                memory=memory,
                consolidation_decision={
                    "knowledge_candidates": [],
                    "private_candidates": [],
                    "knowledge_query": "",
                },
            )
            self.assertIsNone(path)
            await worker.close()

        self.assertEqual([], calls)

    async def test_growth_queue_skips_short_non_relational_chat(self):
        async def fake_grow(identity, question, route):
            raise AssertionError("must not run")

        with tempfile.TemporaryDirectory() as directory:
            worker = BackgroundGrowthWorker(directory, fake_grow, min_chars=8)
            await worker.start()
            path = await worker.enqueue(
                PlatformIdentity("telegram", "42"),
                "今天天气不错",
                MemoryRoute(True, True, False, "personal"),
            )
            await worker.close()
            self.assertIsNone(path)


class CoordinatorTests(unittest.IsolatedAsyncioTestCase):
    def test_evidence_bridge_fallback_hides_internal_retrieval_wording(self):
        memory = RetrievedMemory(
            "evidence",
            raw_result={
                "domains": {
                    "knowledge": {
                        "rerank_trace": {
                            "cache_kind": "audited_association_cue",
                            "association_capsules": [
                                {
                                    "association_id": 8,
                                    "relation_key": "evidence_bridge",
                                    "relation_text": (
                                        "查询综合推论：一段证据记录甲采取行动，"
                                        "另一段记录乙受到影响；此关联只用于共同检索，"
                                        "不断言甲导致了乙。"
                                    ),
                                }
                            ],
                        },
                    }
                }
            },
        )

        reply = ConversationCoordinator._association_capsule_reply(memory)

        self.assertIn("甲采取行动", reply)
        self.assertIn("乙受到影响", reply)
        self.assertIn("不足以把二者直接判为因果", reply)
        self.assertNotIn("查询综合推论", reply)
        self.assertNotIn("共同检索", reply)

    def test_evidence_bridge_fallback_selects_slot_relevant_sentences(self):
        memory = RetrievedMemory(
            "evidence",
            raw_result={
                "domains": {
                    "knowledge": {
                        "rerank_trace": {
                            "cache_kind": "audited_association_cue",
                            "association_capsules": [
                                {
                                    "relation_key": "evidence_bridge",
                                    "relation_text": (
                                        "查询综合推论：检索证据桥：问题“综合说明”中的"
                                        "槽“袭击者”由 Episode 1 直接支持，其观察为："
                                        "阿里乌斯袭击了条约现场。老师随后受伤；"
                                        "槽“合作背景”由 Episode 2 直接支持，其观察为："
                                        "学生们建立友谊。未花与阿里乌斯存在合作；"
                                        "此边只用于共同检索，不断言因果。"
                                    ),
                                }
                            ],
                        }
                    }
                }
            },
        )

        reply = ConversationCoordinator._association_capsule_reply(memory)

        self.assertIn("阿里乌斯袭击了条约现场", reply)
        self.assertIn("未花与阿里乌斯存在合作", reply)
        self.assertNotIn("学生们建立友谊", reply)
        self.assertNotIn("Episode", reply)

    def test_bridge_boundary_rejects_positive_causation_but_allows_denial(self):
        memory = RetrievedMemory(
            "evidence",
            raw_result={
                "domains": {
                    "knowledge": {
                        "rerank_trace": {
                            "cache_kind": "audited_association_cue",
                            "association_capsules": [
                                {"relation_key": "evidence_bridge"}
                            ],
                        }
                    }
                }
            },
        )

        self.assertTrue(
            ConversationCoordinator._evidence_bridge_boundary_violation(
                "这项合作最终导致了袭击。", memory
            )
        )
        self.assertFalse(
            ConversationCoordinator._evidence_bridge_boundary_violation(
                "现有证据不能说明这项合作导致了袭击。", memory
            )
        )

    def test_audited_association_capsule_counts_as_cached_recall(self):
        route = MemoryRoute(
            False,
            False,
            True,
            "semantic_plan_cache",
            "standard",
            "evidence_gated",
            False,
        )
        memory = RetrievedMemory(
            "evidence",
            raw_result={
                "domains": {
                    "knowledge": {
                        "service_cache_hit": False,
                        "rerank_trace": {
                            "cache_kind": "audited_association_cue"
                        },
                    }
                }
            },
        )

        self.assertTrue(
            ConversationCoordinator._all_recalled_domains_cached(
                memory, route
            )
        )

    async def test_audited_capsule_skips_chat_gateway(self):
        route = MemoryRoute(
            False,
            False,
            True,
            "audited_association_cache",
            "standard",
            "evidence_gated",
            False,
        )

        class Planner:
            async def plan(self, **_kwargs):
                return PlannedChatRequest(
                    route=route,
                    retrieval_question="当前问法",
                    planner="association_cache",
                )

        class Memory:
            async def recall(self, *_args, **_kwargs):
                return RetrievedMemory(
                    "direct evidence",
                    raw_result={
                        "domains": {
                            "knowledge": {
                                "rerank_trace": {
                                    "cache_kind": "audited_association_cue",
                                    "association_capsules": [
                                        {
                                            "association_id": 8,
                                            "relation_text": "甲的选择导致乙随后采取行动。",
                                        }
                                    ],
                                }
                            }
                        }
                    },
                )

        class Fast:
            async def generate_response(self, *_args, **_kwargs):
                raise AssertionError("audited capsule should skip the gateway")

        class Main:
            async def generate_response(self, *_args, **_kwargs):
                raise AssertionError("audited capsule should avoid another gateway call")

        coordinator = ConversationCoordinator(
            chat_engine=Main(),
            fast_chat_engine=Fast(),
            memory_system=Memory(),
            system_prompt_factory=lambda context: context,
            sessions=ConversationSessionBuffer(),
            request_planner=Planner(),
        )

        reply = await coordinator.handle(
            identity=PlatformIdentity("synthetic", "teacher"),
            text="换一种问法",
        )

        self.assertIn("甲的选择导致乙随后采取行动", reply.text)
        self.assertIn("老师", reply.text)

    async def test_fast_gateway_failure_uses_direct_evidence_without_second_call(self):
        route = MemoryRoute(
            False,
            False,
            True,
            "model_intent_planner",
            "standard",
            "none",
            False,
        )

        class Planner:
            async def plan(self, **_kwargs):
                return PlannedChatRequest(
                    route=route,
                    retrieval_question="谁执行了袭击？",
                    planner="model",
                )

        class Memory:
            async def recall(self, *_args, **_kwargs):
                return RetrievedMemory(
                    "direct evidence",
                    raw_result={
                        "domains": {
                            "knowledge": {
                                "retrieval_quality": {"sufficient": False},
                                "evidence_episodes": [
                                    {
                                        "id": 1,
                                        "text": "势力甲执行了袭击。",
                                        "generation": 0,
                                        "evidence_origin": "source",
                                    },
                                    {
                                        "id": 2,
                                        "text": "乙与势力甲曾经合作。",
                                        "generation": 0,
                                        "evidence_origin": "source",
                                    },
                                ],
                            }
                        }
                    },
                )

        class Fast:
            async def generate_response(self, *_args, **_kwargs):
                raise TimeoutError("gateway unavailable")

        class Main:
            async def generate_response(self, *_args, **_kwargs):
                raise AssertionError("direct evidence should avoid a second call")

        coordinator = ConversationCoordinator(
            chat_engine=Main(),
            fast_chat_engine=Fast(),
            memory_system=Memory(),
            system_prompt_factory=lambda context: context,
            sessions=ConversationSessionBuffer(),
            request_planner=Planner(),
        )

        reply = await coordinator.handle(
            identity=PlatformIdentity("synthetic", "teacher"),
            text="谁执行了袭击？",
        )

        self.assertIn("势力甲执行了袭击", reply.text)
        self.assertIn("乙与势力甲曾经合作", reply.text)
        self.assertIn("直接资料", reply.text)

    async def test_inline_memory_footer_is_not_exposed_and_is_enqueued(self):
        class FakeMemory:
            async def recall(self, identity, question, route=None):
                return RetrievedMemory(
                    "Episode #7",
                    raw_result={
                        "domains": {
                            "knowledge": {
                                "evidence_episodes": [
                                    {"id": 7, "text": "白子持续练习骑行"},
                                    {"id": 8, "text": "白子长途骑行仍坚持完成"},
                                ]
                            }
                        }
                    },
                )

        class FastChat:
            async def generate_response(self, messages, system_prompt, **kwargs):
                self.prompt = system_prompt
                payload = {
                    "knowledge_candidates": [
                        {
                            "claim": "白子把骑行视为持续投入的活动",
                            "premise_episode_ids": [7, 8],
                            "inference_type": "trait",
                            "confidence": 0.8,
                            "reason": "行为证据支持",
                        }
                    ],
                    "private_candidates": [],
                }
                return (
                    "白子对骑行很投入。<assistant_memory>"
                    + json.dumps(payload, ensure_ascii=False)
                    + "</assistant_memory>"
                )

        class Growth:
            async def enqueue(self, identity, question, route, **kwargs):
                self.kwargs = kwargs

        growth = Growth()
        chat = FastChat()
        coordinator = ConversationCoordinator(
            chat_engine=chat,
            fast_chat_engine=chat,
            memory_system=FakeMemory(),
            system_prompt_factory=lambda context: context,
            sessions=ConversationSessionBuffer(),
            growth_worker=growth,
        )
        reply = await coordinator.handle(
            identity=PlatformIdentity("synthetic", "teacher"),
            text="你觉得白子是一个什么样的人？",
        )
        self.assertEqual("白子对骑行很投入。", reply.text)
        self.assertNotIn("assistant_memory", reply.text)
        self.assertIn("隐藏的记忆候选输出", chat.prompt)
        self.assertEqual(
            "白子把骑行视为持续投入的活动",
            growth.kwargs["consolidation_decision"]["knowledge_candidates"][0]["claim"],
        )
        self.assertIsInstance(growth.kwargs["memory"], RetrievedMemory)

    async def test_light_roleplay_uses_fast_engine_and_creative_boundary(self):
        class CreativePlanner:
            async def plan(self, **_kwargs):
                return PlannedChatRequest(
                    route=MemoryRoute(
                        True,
                        True,
                        True,
                        "model_intent_planner",
                        "light",
                        "evidence_gated",
                        True,
                    ),
                    retrieval_question="创建当前出差任务",
                    planner="model",
                )

        class FakeMemory:
            async def recall(self, identity, question, route=None):
                return RetrievedMemory(
                    "阿拜多斯旧剧情事件正文",
                    raw_result={
                        "domains": {
                            "knowledge": {
                                "evidence_episodes": [
                                    {"id": 9, "text": "优香曾经发过一条旧消息"}
                                ],
                                "evidence_concepts": [
                                    {
                                        "canonical_name": "阿拜多斯",
                                        "description": "沙漠中的学园",
                                    }
                                ],
                            }
                        }
                    },
                    domains=("knowledge",),
                )

        class MainChat:
            async def generate_response(self, *args, **kwargs):
                raise AssertionError("light route must use fast engine")

        class FastChat:
            async def generate_response(self, messages, system_prompt, **kwargs):
                self.system_prompt = system_prompt
                return "今天有一项阿拜多斯现场协助任务。"

        fast = FastChat()
        coordinator = ConversationCoordinator(
            chat_engine=MainChat(),
            fast_chat_engine=fast,
            memory_system=FakeMemory(),
            system_prompt_factory=lambda context: context,
            sessions=ConversationSessionBuffer(),
            request_planner=CreativePlanner(),
        )
        reply = await coordinator.handle(
            identity=PlatformIdentity("synthetic", "teacher"),
            text="现在有哪些出差任务？",
        )
        self.assertTrue(reply.route.creative)
        self.assertIn("新互动只属于当前用户", fast.system_prompt)
        self.assertIn("稳定推论", fast.system_prompt)
        self.assertIn("不是原作既定事实", fast.system_prompt)
        self.assertIn("阿拜多斯：沙漠中的学园", fast.system_prompt)
        self.assertNotIn("优香曾经发过一条旧消息", fast.system_prompt)

    def test_retrieval_history_is_only_added_for_elliptical_followups(self):
        coordinator = ConversationCoordinator(
            chat_engine=None,
            memory_system=None,
            system_prompt_factory=lambda context: context,
            sessions=ConversationSessionBuffer(),
        )
        identity = PlatformIdentity("telegram", "42", "Alice")
        coordinator.sessions.add(identity.key, "user", "我们刚才谈到未花。")
        coordinator.sessions.add(identity.key, "assistant", "是的。")

        standalone = "古圣堂的大爆炸由哪个分校势力直接执行，与未花有什么关系？"
        self.assertEqual(
            standalone,
            coordinator._retrieval_question(identity, standalone),
        )
        followup = coordinator._retrieval_question(identity, "她后来怎么样？")
        self.assertIn("近期对话", followup)
        self.assertIn("当前用户输入：她后来怎么样？", followup)

        coordinator.sessions.add(identity.key, "user", "我准备去阿拜多斯出差。")
        coordinator.sessions.add(identity.key, "assistant", "那我陪老师准备。")
        creative = coordinator._retrieval_question(
            identity,
            "现在有哪些出差任务？",
            MemoryRoute(
                True,
                True,
                True,
                "creative_roleplay",
                "light",
                "evidence_gated",
                True,
            ),
        )
        self.assertIn("我准备去阿拜多斯出差", creative)
        self.assertNotIn("那我陪老师准备", creative)

    async def test_coordinator_passes_composite_identity_and_persists_exchange(self):
        class FakeMemory:
            def __init__(self):
                self.calls = []

            async def recall(self, identity, question, route=None):
                self.calls.append((identity, question, route))
                return RetrievedMemory("PRIVATE EVIDENCE", domains=("user",))

        class FakeChat:
            async def generate_response(self, messages, system_prompt, task_context=""):
                self.messages = list(messages)
                self.system_prompt = system_prompt
                self.task_context = task_context
                return "reply"

        class FakeJournal:
            async def record_exchange(self, identity, user_text, assistant_text):
                self.value = (identity, user_text, assistant_text)
                return Path("ready.txt")

        class FakeWorker:
            async def enqueue(self, path):
                self.path = path

        class FakeGrowthWorker:
            async def enqueue(self, identity, question, route):
                self.value = (identity, question, route)

        memory = FakeMemory()
        chat = FakeChat()
        journal = FakeJournal()
        worker = FakeWorker()
        growth_worker = FakeGrowthWorker()
        coordinator = ConversationCoordinator(
            chat_engine=chat,
            memory_system=memory,
            system_prompt_factory=lambda context: f"PROMPT\n{context}",
            sessions=ConversationSessionBuffer(),
            journal=journal,
            ingestion_worker=worker,
            growth_worker=growth_worker,
        )
        identity = PlatformIdentity("telegram", "42", "Alice")
        reply = await coordinator.handle(identity=identity, text="你记得我吗？")
        self.assertEqual("reply", reply.text)
        self.assertEqual(identity, memory.calls[0][0])
        self.assertIn("PRIVATE EVIDENCE", chat.system_prompt)
        self.assertEqual("telegram:42:chat", chat.task_context)
        self.assertEqual(identity, journal.value[0])
        self.assertEqual(Path("ready.txt"), worker.path)
        self.assertEqual(identity, growth_worker.value[0])
        self.assertEqual(memory.calls[0][2], growth_worker.value[2])
        self.assertEqual(
            {
                "memory_recall_seconds",
                "memory_intent_seconds",
                "chat_generation_seconds",
                "memory_guard_seconds",
                "postprocess_seconds",
                "total_seconds",
            },
            set(reply.timings),
        )

    async def test_private_memory_violation_is_dynamically_rewritten(self):
        class EmptyPrivateMemory:
            async def recall(self, identity, question, route=None):
                return RetrievedMemory(
                    "===== 当前用户的私人记忆 =====\n无可靠命中",
                    raw_result={"domains": {"user": {"evidence_episodes": []}}},
                )

        class InventingChat:
            async def generate_response(self, *args, **kwargs):
                return "是不是咖啡？我记得上次见过。"

        class Audit:
            def __init__(self):
                self.calls = 0

            async def generate_response(self, *args, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    return json.dumps(
                        {
                            "passed": False,
                            "private_claims": [
                                {
                                    "claim": "喜欢咖啡",
                                    "supporting_private_evidence_ids": [],
                                },
                                {
                                    "claim": "上次见过",
                                    "supporting_private_evidence_ids": [],
                                },
                            ],
                            "lore_claims": [],
                            "violations": ["无证据时猜测饮料并虚构过去观察"],
                            "reason": "unsupported",
                        },
                        ensure_ascii=False,
                    )
                return json.dumps(
                        {
                            "passed": True,
                            "private_claims": [],
                            "lore_claims": [],
                        "violations": [],
                        "reason": "",
                    },
                    ensure_ascii=False,
                )

        class Rewrite:
            async def generate_response(self, *args, **kwargs):
                return "老师这是突然检查吗？阿洛娜不想乱猜，再告诉我一次好不好？"

        guard = PrivateMemoryResponseGuard(
            audit_engine=Audit(),
            rewrite_engine=Rewrite(),
            emergency_reply_factory=lambda: "紧急兜底",
        )

        coordinator = ConversationCoordinator(
            chat_engine=InventingChat(),
            memory_system=EmptyPrivateMemory(),
            system_prompt_factory=lambda context: context,
            private_memory_guard=guard,
            sessions=ConversationSessionBuffer(),
        )
        reply = await coordinator.handle(
            identity=PlatformIdentity("discord", "new-user"),
            text="你记得我的私人暗号和最喜欢的饮料吗？",
        )
        self.assertIn("突然检查", reply.text)
        self.assertNotEqual("紧急兜底", reply.text)
        self.assertTrue(reply.memory_guard["applied"])
        self.assertTrue(reply.memory_guard["rewritten"])
        self.assertFalse(reply.memory_guard["emergency_fallback"])
        self.assertEqual(
            {
                "contract_seconds",
                "audit_1_seconds",
                "rewrite_seconds",
                "audit_2_seconds",
                "total_seconds",
            },
            set(reply.memory_guard["timings"]),
        )

    async def test_grounded_emergency_does_not_claim_memory_is_missing(self):
        class AlwaysReject:
            async def generate_response(self, *args, **kwargs):
                return json.dumps(
                    {
                        "passed": False,
                        "violations": ["synthetic rejection"],
                        "reason": "test",
                    }
                )

        class BadRewrite:
            async def generate_response(self, *args, **kwargs):
                return "still invalid"

        memory = RetrievedMemory(
            "private",
            raw_result={
                "domains": {
                    "user": {
                        "evidence_episodes": [
                            {
                                "id": 1,
                                "text": "老师亲口说喜欢乌龙茶。",
                                "evidence_origin": "source",
                                "epistemic_status": "observed",
                                "generation": 0,
                            }
                        ]
                    }
                }
            },
        )
        guard = PrivateMemoryResponseGuard(
            audit_engine=AlwaysReject(),
            rewrite_engine=BadRewrite(),
            emergency_reply_factory=lambda: "没有找到记忆",
        )
        result = await guard.enforce(
            question="你记得我喜欢什么吗？",
            draft="invalid",
            memory=memory,
            route=MemoryRoute(True, True, False, "personal"),
            messages=[],
            system_prompt="persona",
            task_context="test",
        )

        self.assertTrue(result.emergency_fallback)
        self.assertEqual("grounded", result.contract_state)
        self.assertIn("确实记得", result.text)
        self.assertNotIn("没有找到", result.text)


class GuardValidationTests(unittest.IsolatedAsyncioTestCase):
    async def test_unsupported_lore_claim_cannot_pass_by_model_boolean(self):
        class AuditEngine:
            @staticmethod
            async def generate_response(*args, **kwargs):
                return json.dumps(
                    {
                        "passed": True,
                        "private_claims": [],
                        "lore_claims": [
                            {
                                "claim": "角色总是佩戴某件饰品",
                                "supporting_non_private_evidence_ids": [],
                            }
                        ],
                        "violations": [],
                        "reason": "模型错误地认为可以通过",
                    },
                    ensure_ascii=False,
                )

        guard = PrivateMemoryResponseGuard(
            audit_engine=AuditEngine(),
            rewrite_engine=AuditEngine(),
            emergency_reply_factory=lambda: "",
        )
        memory = RetrievedMemory(
            context="",
            raw_result={
                "domains": {
                    "user": {"evidence_episodes": []},
                    "knowledge": {
                        "evidence_episodes": [
                            {"id": 7, "text": "角色属于某组织。"}
                        ]
                    },
                }
            },
        )

        result = await guard._audit(
            question="你记得这个角色吗？",
            draft="这个角色总是佩戴某件饰品。",
            contract=guard.build_contract(memory),
            task_context="test",
        )

        self.assertFalse(result["passed"])
        self.assertTrue(
            any("缺少直接证据" in value for value in result["violations"])
        )


class LegacyRemovalTests(unittest.TestCase):
    def test_old_postgres_memory_modules_are_absent(self):
        memory_dir = Path(__file__).resolve().parents[1] / "src" / "memory"
        for filename in (
            "db.py",
            "models.py",
            "working_memory.py",
            "semantic_memory.py",
            "episodic_memory.py",
            "knowledge_memory.py",
            "consolidation.py",
        ):
            self.assertFalse((memory_dir / filename).exists(), filename)


if __name__ == "__main__":
    unittest.main()
