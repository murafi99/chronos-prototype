"""
knowledge_graph/schema.py

INTERFACE SCAFFOLD (Section 10.1) -- not a working implementation.
Defines the 12 domain sub-graph schema described in manuscript Section 7.2,
plus the provenance-confidence edge-tagging extension (Section 7.2,
[ORIGINAL PROPOSAL]) and the counterfactual-query extension to standard
temporal-knowledge-graph reasoning (TKGR; Section 3.3).

A production implementation requires a graph-database backend (e.g. Neo4j,
TigerGraph, or an RDF store) and licensed historical datasets, both out of
scope for this paper's prototype.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List
from enum import Enum


class Domain(Enum):
    POLITICAL = "political"
    ECONOMIC = "economic"
    MILITARY = "military"
    TECHNOLOGY = "technology"
    CLIMATE = "climate"
    DISEASE = "disease"
    TRADE = "trade"
    MIGRATION = "migration"
    CULTURAL = "cultural"
    ACTOR_INSTITUTION = "actor_institution"   # cross-cutting
    GEOGRAPHIC = "geographic"                  # cross-cutting
    PROVENANCE = "provenance"                  # cross-cutting


@dataclass
class ProvenanceTag:
    """[ORIGINAL PROPOSAL], Section 7.2 -- attached to every graph edge."""
    source_type: str            # e.g. "primary_document", "secondary_scholarship", "dataset"
    corroboration_count: int    # number of independent sources agreeing
    contested: bool             # True if historiography disputes this edge


@dataclass
class TemporalEdge:
    source_node: str
    target_node: str
    relation: str
    domain: Domain
    timestamp: Optional[str]    # ISO 8601 or None if atemporal
    provenance: ProvenanceTag


class TemporalKnowledgeGraph:
    """Interface only. A working implementation would wrap a graph-database
    client and implement:
      - forecast(node, horizon)                     standard TKGR (Sec. 3.3)
      - counterfactual_query(subgraph, do_event)     Sec. 7.2 extension:
            (1) sever incoming edges to do_event's node,
            (2) re-run forecasting machinery forward from the intervention
                point rather than from the present,
            (3) return forecasting uncertainty as the counterfactual-branch
                uncertainty base rate.
    """

    def __init__(self):
        self.edges: List[TemporalEdge] = []

    def add_edge(self, edge: TemporalEdge) -> None:
        self.edges.append(edge)

    def forecast(self, node: str, horizon: str):
        raise NotImplementedError("Requires a populated graph and a TKGR model backend.")

    def counterfactual_query(self, subgraph_nodes: List[str], do_event: str):
        raise NotImplementedError("Requires a populated graph and a TKGR model backend; "
                                   "see manuscript Section 7.2 for the exact algorithm.")
