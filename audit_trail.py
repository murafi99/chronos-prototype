"""
audit_logging/audit_trail.py

INTERFACE SCAFFOLD (Section 10.1) -- not a working implementation.
Defines the record structure for CHRONOS's explainable output & audit layer
(manuscript Section 7.7): every output branch must carry a causal-path
trace, inherited provenance tags, calibration statistics, and a flag for
whether any contradiction was auto-resolved or deferred to human review.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class BranchAuditRecord:
    branch_id: str
    causal_path: List[str]                  # ordered list of node names traced back through G'
    provenance_refs: List[str]               # KG edge IDs this branch's path relied on
    confidence: float
    ece_context: Dict[str, Any] = field(default_factory=dict)   # from uncertainty/calibration.py
    contradictions_auto_resolved: List[str] = field(default_factory=list)
    contradictions_flagged_for_review: List[str] = field(default_factory=list)


class AuditLog:
    def __init__(self):
        self.records: List[BranchAuditRecord] = []

    def add(self, record: BranchAuditRecord) -> None:
        self.records.append(record)

    def needs_human_review(self) -> List[BranchAuditRecord]:
        return [r for r in self.records if r.contradictions_flagged_for_review]
