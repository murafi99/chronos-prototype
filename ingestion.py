"""
data_layer/ingestion.py

INTERFACE SCAFFOLD (Section 10.1) -- not a working implementation.
Source-connector and provenance-tagging entry point feeding
knowledge_graph/schema.py. A production implementation requires licensed
historical datasets and source-specific parsers, both out of scope here.
"""
from __future__ import annotations
from typing import Iterable
from knowledge_graph.schema import TemporalEdge


class SourceConnector:
    """Base class for a historical-data source connector (e.g. a structured
    dataset, a digitized archive, an API). Subclasses implement `.extract()`
    to yield TemporalEdge objects with a ProvenanceTag already attached."""

    def extract(self) -> Iterable[TemporalEdge]:
        raise NotImplementedError
