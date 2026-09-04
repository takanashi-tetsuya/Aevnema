from __future__ import annotations

import unittest

from src.bot.adapter_support import AdapterRuntime, DebouncedDispatcher, split_message
from src.bot.dispatch import DebouncedDispatcher as DispatcherImplementation
from src.bot.messages import split_message as split_message_implementation
from src.bot.runtime import AdapterRuntime as RuntimeImplementation
from src.memory.domain import AssociativeMemoryService as DomainImplementation
from src.memory.intent_planner import MemoryIntentPlanner as PlannerImplementation
from src.memory.service import (
    AssociativeMemoryService,
    MemoryIntentPlanner,
    MemorySystem,
)
from src.memory.system import MemorySystem as SystemImplementation


class PublicFacadeTests(unittest.TestCase):
    def test_adapter_facade_exports_single_implementations(self) -> None:
        self.assertIs(AdapterRuntime, RuntimeImplementation)
        self.assertIs(DebouncedDispatcher, DispatcherImplementation)
        self.assertIs(split_message, split_message_implementation)

    def test_memory_facade_exports_single_implementations(self) -> None:
        self.assertIs(AssociativeMemoryService, DomainImplementation)
        self.assertIs(MemoryIntentPlanner, PlannerImplementation)
        self.assertIs(MemorySystem, SystemImplementation)


if __name__ == "__main__":
    unittest.main()
