"""
agent_layer/agent_base.py

INTERFACE SCAFFOLD (Section 10.1) -- not a working implementation.
Defines the goal/belief/memory agent architecture described in manuscript
Section 7.6, following the memory-and-reflection pattern of Park et al.
(2023, "Generative Agents") and the causal-graph action-legality constraint
that is CHRONOS's specific extension of that pattern.

A production implementation requires an LLM API client and a populated
knowledge_graph/ backend, both out of scope for this paper's prototype.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod


@dataclass
class MemoryEvent:
    timestamp: float
    description: str
    importance: float  # 0-1, per Park et al. (2023) reflection-memory scoring


@dataclass
class AgentState:
    goals: List[str]
    beliefs: Dict[str, Any]
    memory_stream: List[MemoryEvent] = field(default_factory=list)


class InstitutionalAgent(ABC):
    """Base class for CHRONOS actors (governments, military commands,
    economic institutions, cultural/religious bodies, scientific
    communities, international organizations) -- Section 7.6."""

    def __init__(self, name: str, initial_goals: List[str]):
        self.name = name
        self.state = AgentState(goals=initial_goals, beliefs={})

    @abstractmethod
    def ground_beliefs(self, kg_query_fn) -> None:
        """Re-ground self.state.beliefs against the current knowledge-graph
        state via retrieval-augmented generation (RAG). `kg_query_fn` is
        supplied by knowledge_graph/ at runtime."""
        raise NotImplementedError

    @abstractmethod
    def propose_action(self, causal_graph_state) -> Optional[str]:
        """Propose a next action given current beliefs and goals.

        MUST check action legality against `causal_graph_state` (the current
        SCM state from causal_engine/) before returning an action -- an
        agent cannot select an action whose preconditions the causal graph
        marks as already false. This is the mechanism described in
        Section 7.6 that distinguishes CHRONOS's agent layer from
        unconstrained LLM-agent systems (WarAgent, BattleAgent; Section 3.6)."""
        raise NotImplementedError

    def record_memory(self, event: MemoryEvent) -> None:
        self.state.memory_stream.append(event)
