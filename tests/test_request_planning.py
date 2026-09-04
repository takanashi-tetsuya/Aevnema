from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.bot.chat_service import ConversationCoordinator
from src.bot.request_planning import ChatRequestPlanner, bounded_recent_messages
from src.memory.contracts import DomainRecallRequest, RetrievedMemory
from src.memory.conversation import ChatTurn, ConversationSessionBuffer
from src.memory.domain import AssociativeMemoryService
from src.memory.config import MemoryServiceConfig
from src.memory.identity import PlatformIdentity
from src.memory.intent_planner import MemoryIntentPlanner


class ContextBoundaryTests(unittest.TestCase):
    def test_recent_context_is_bounded_and_keeps_newest_turns(self):
        turns = [
            ChatTurn("user", "old-" + "a" * 20),
            ChatTurn("assistant", "middle-" + "b" * 20),
            ChatTurn("user", "newest"),
        ]
        selected = bounded_recent_messages(
            turns,
            max_messages=2,
            max_chars=12,
        )
        self.assertEqual("newest", selected[-1]["content"])
        self.assertLessEqual(
            sum(len(item["content"]) for item in selected),
            12,
        )
        self.assertNotIn("old-", str(selected))


class ModelRequestPlanningTests(unittest.IsolatedAsyncioTestCase):
    async def test_audited_association_cache_routes_without_cloud_planner(self):
        class IntentEngine:
            def __init__(self):
                self.calls = 0

            async def generate_response(self, *_args, **_kwargs):
                self.calls += 1
                raise AssertionError("intent model must not run on a cue hit")

        async def match(_text):
            return {
                "association_id": 7,
                "coverage": 0.31,
                "margin": 0.12,
                "score_kind": "local_char_idf",
            }

        engine = IntentEngine()
        planner = ChatRequestPlanner(
            sessions=ConversationSessionBuffer(20),
            intent_planner=MemoryIntentPlanner(engine),
            association_route_matcher=match,
        )

        planned = await planner.plan(
            identity=PlatformIdentity("test", "association-route"),
            current_message="换一种说法询问已经学会的关系",
        )

        self.assertEqual(0, engine.calls)
        self.assertEqual("association_cache", planned.planner)
        self.assertTrue(planned.route.knowledge)
        self.assertFalse(planned.route.user)
        self.assertEqual("standard", planned.route.intensity)
        self.assertEqual(
            "换一种说法询问已经学会的关系",
            planned.domain_requests["knowledge"].query,
        )

    async def test_failed_model_plan_is_refined_by_observed_evidence(self):
        class FailingIntentEngine:
            def __init__(self):
                self.calls = 0

            async def generate_response(self, *_args, **_kwargs):
                self.calls += 1
                raise TimeoutError("planner unavailable")

        async def embed(_text):
            return np.asarray([1.0, 0.0], dtype=np.float32)

        sessions = ConversationSessionBuffer(20)
        identity = PlatformIdentity("test", "evidence-refined-cache")
        engine = FailingIntentEngine()
        planner = ChatRequestPlanner(
            sessions=sessions,
            intent_planner=MemoryIntentPlanner(engine),
            semantic_embedder=embed,
            semantic_cache_similarity=0.6,
        )
        first = await planner.plan(
            identity=identity,
            current_message="渚为什么组建补课部？",
        )
        self.assertEqual("fallback", first.planner)
        planner.remember_semantic_plan(
            session_key=identity.key,
            current_message="渚为什么组建补课部？",
            planned=first,
            vector=np.asarray([1.0, 0.0], dtype=np.float32),
            observed_domains={
                "knowledge": {
                    "evidence_episodes": [{"id": 1}],
                    "retrieval_quality": {"sufficient": True},
                },
                "user": {
                    "evidence_episodes": [],
                    "retrieval_quality": {"sufficient": False},
                },
                "public": {
                    "evidence_episodes": [],
                    "retrieval_quality": {"sufficient": False},
                },
            },
        )

        second = await planner.plan(
            identity=identity,
            current_message="补课部真正是为查什么？",
        )

        self.assertEqual(1, engine.calls)
        self.assertEqual("semantic_cache", second.planner)
        self.assertTrue(second.route.knowledge)
        self.assertFalse(second.route.user)
        self.assertFalse(second.route.public)

    async def test_semantic_plan_cache_reuses_equivalent_model_plan(self):
        class IntentEngine:
            def __init__(self):
                self.calls = 0

            async def generate_response(self, *args, **kwargs):
                self.calls += 1
                return json.dumps(
                    {
                        "needs_memory": True,
                        "domains": {
                            "private": False,
                            "public": False,
                            "knowledge": True,
                        },
                        "queries": {"knowledge": "渚组建补课部的真实原因"},
                        "target_entities": ["渚", "补课部"],
                        "requested_relation": "原因",
                        "answer_slots": ["表面目的", "政治目的"],
                        "intensity": "standard",
                        "creative": False,
                        "knowledge_write_policy": "evidence_gated",
                    },
                    ensure_ascii=False,
                )

        probes = 0

        async def embed(_text):
            nonlocal probes
            probes += 1
            return np.asarray([1.0, 0.0], dtype=np.float32)

        sessions = ConversationSessionBuffer(20)
        identity = PlatformIdentity("test", "semantic-cache")
        engine = IntentEngine()
        planner = ChatRequestPlanner(
            sessions=sessions,
            intent_planner=MemoryIntentPlanner(engine),
            semantic_embedder=embed,
            semantic_cache_similarity=0.9,
        )
        first = await planner.plan(
            identity=identity,
            current_message="渚为什么组建补课部？",
        )
        planner.remember_semantic_plan(
            session_key=identity.key,
            current_message="渚为什么组建补课部？",
            planned=first,
            vector=np.asarray([1.0, 0.0], dtype=np.float32),
        )

        second = await planner.plan(
            identity=identity,
            current_message="渚组建补课部的原因是什么？",
        )

        self.assertEqual(1, engine.calls)
        self.assertEqual(1, probes)
        self.assertEqual("semantic_cache", second.planner)
        self.assertEqual(
            "渚组建补课部的原因是什么？",
            second.domain_requests["knowledge"].query,
        )
        self.assertEqual(
            ["渚组建补课部的原因是什么？"],
            second.domain_requests["knowledge"].intent_override[
                "search_queries"
            ],
        )
        np.testing.assert_array_equal(
            np.asarray([1.0, 0.0], dtype=np.float32),
            second.semantic_vector,
        )

    async def test_model_plan_receives_current_message_and_bounded_context(self):
        class IntentEngine:
            async def generate_response(self, messages, **kwargs):
                self.prompt = messages[0].content
                self.kwargs = kwargs
                return json.dumps(
                    {
                        "needs_memory": True,
                        "domains": {
                            "private": True,
                            "public": False,
                            "knowledge": True,
                        },
                        "queries": {
                            "private": "用户与甲的共同经历",
                            "public": "",
                            "knowledge": "甲在导入资料中的经历",
                        },
                        "target_entities": ["甲"],
                        "requested_relation": "经历",
                        "temporal_constraint": "",
                        "causal_constraint": "",
                        "answer_slots": ["私人经历", "资料事实"],
                        "intensity": "standard",
                        "creative": False,
                        "knowledge_write_policy": "evidence_gated",
                        "uncertainty_required": True,
                        "reason": "需要区分个人记忆与外部资料",
                    },
                    ensure_ascii=False,
                )

        sessions = ConversationSessionBuffer(20)
        identity = PlatformIdentity("test", "42")
        sessions.add(identity.key, "user", "很早以前的无关内容" * 20)
        sessions.add(identity.key, "assistant", "刚才我们谈到甲。")
        engine = IntentEngine()
        planner = ChatRequestPlanner(
            sessions=sessions,
            intent_planner=MemoryIntentPlanner(engine),
            context_messages=1,
            context_chars=100,
        )

        planned = await planner.plan(
            identity=identity,
            current_message="你还记得甲后来发生了什么吗？",
        )

        self.assertEqual("model", planned.planner)
        self.assertTrue(planned.route.user)
        self.assertTrue(planned.route.knowledge)
        self.assertFalse(planned.route.public)
        self.assertEqual(
            {"private", "knowledge"}, set(planned.domain_requests)
        )
        self.assertEqual(
            ["甲"],
            planned.domain_requests["knowledge"].intent_override[
                "target_entities"
            ],
        )
        self.assertIn("你还记得甲后来发生了什么吗", engine.prompt)
        self.assertIn("刚才我们谈到甲", engine.prompt)
        self.assertNotIn("很早以前的无关内容", engine.prompt)
        self.assertEqual("memory-intent", engine.kwargs["task_context"])

    async def test_planner_failure_uses_generic_fallback(self):
        class FailingEngine:
            async def generate_response(self, *args, **kwargs):
                raise TimeoutError("planner unavailable")

        sessions = ConversationSessionBuffer(20)
        identity = PlatformIdentity("test", "42")
        sessions.add(identity.key, "user", "我们刚才讨论了甲。")
        planner = ChatRequestPlanner(
            sessions=sessions,
            intent_planner=MemoryIntentPlanner(FailingEngine()),
        )

        planned = await planner.plan(
            identity=identity,
            current_message="后来呢？",
        )

        self.assertEqual("fallback", planned.planner)
        self.assertEqual("model_planner_fallback", planned.route.reason)
        self.assertTrue(planned.route.user)
        self.assertTrue(planned.route.public)
        self.assertTrue(planned.route.knowledge)
        self.assertIn("TimeoutError", planned.error)
        self.assertIn("我们刚才讨论了甲", planned.retrieval_question)

    async def test_coordinator_passes_request_local_domain_contracts(self):
        class IntentEngine:
            async def generate_response(self, *args, **kwargs):
                return json.dumps(
                    {
                        "needs_memory": True,
                        "domains": {
                            "private": False,
                            "public": False,
                            "knowledge": True,
                        },
                        "queries": {
                            "private": "",
                            "public": "",
                            "knowledge": "甲与乙的关系",
                        },
                        "target_entities": ["甲", "乙"],
                        "requested_relation": "关系",
                        "temporal_constraint": "",
                        "causal_constraint": "",
                        "answer_slots": ["关系证据"],
                        "intensity": "standard",
                        "creative": False,
                        "knowledge_write_policy": "evidence_gated",
                        "uncertainty_required": False,
                        "reason": "external evidence",
                    },
                    ensure_ascii=False,
                )

        class Memory:
            async def recall(
                self,
                identity,
                question,
                route=None,
                domain_requests=None,
            ):
                self.value = (identity, question, route, domain_requests)
                return RetrievedMemory("")

        class Chat:
            async def generate_response(self, *args, **kwargs):
                return "answer"

        sessions = ConversationSessionBuffer(20)
        request_planner = ChatRequestPlanner(
            sessions=sessions,
            intent_planner=MemoryIntentPlanner(IntentEngine()),
        )
        memory = Memory()
        coordinator = ConversationCoordinator(
            chat_engine=Chat(),
            memory_system=memory,
            system_prompt_factory=lambda context: context,
            sessions=sessions,
            request_planner=request_planner,
        )

        reply = await coordinator.handle(
            identity=PlatformIdentity("test", "42"),
            text="甲和乙是什么关系？",
        )

        request = memory.value[3]["knowledge"]
        self.assertIsInstance(request, DomainRecallRequest)
        self.assertEqual("甲与乙的关系", request.query)
        self.assertEqual("model", reply.memory.raw_result["intent_planning"]["planner"])
        self.assertIn("memory_intent_seconds", reply.timings)


class DomainRequestCacheTests(unittest.IsolatedAsyncioTestCase):
    async def test_model_planned_request_is_cacheable(self):
        class QueryEngine:
            def __init__(self):
                self.calls = 0

            def query(self, question, generate_answer=False, **kwargs):
                self.calls += 1
                self.options = kwargs
                return {
                    "question": question,
                    "evidence_episodes": [
                        {
                            "id": 1,
                            "text": "甲与乙共同完成了任务。",
                            "source_key": "sample.txt",
                            "score": 0.9,
                        }
                    ],
                    "episode_ids": [1],
                }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service = AssociativeMemoryService(
                MemoryServiceConfig(
                    engine_root=root,
                    database_path=root / "memory.db",
                    log_dir=root / "logs",
                    api_key="test",
                    recall_cache_size=4,
                )
            )
            engine = QueryEngine()
            service._query_engine = engine
            service._application = object()
            service._has_memories = True
            service._small_database_fast_path = True
            request = DomainRecallRequest(
                query="甲与乙的关系",
                intent_override={
                    "target_entities": ["甲", "乙"],
                    "search_queries": ["甲与乙的关系"],
                    "requested_relation": "关系",
                },
                followup_queries=(),
            )

            first = await service.recall("ignored", request=request)
            second = await service.recall("ignored", request=request)

            self.assertEqual(1, engine.calls)
            self.assertFalse(first.raw_result["service_cache_hit"])
            self.assertTrue(second.raw_result["service_cache_hit"])
            self.assertEqual(
                request.intent_override,
                engine.options["intent_override"],
            )
