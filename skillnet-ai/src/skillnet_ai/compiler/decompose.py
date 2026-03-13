from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Set

from skillnet_ai.compiler.models import QueryPlan, Subgoal


CONNECTORS = {"and", "then", "after", "before", "finally"}
LEADING_PHASE_MARKERS = {
    "first",
    "next",
    "then",
    "after",
    "before",
    "finally",
    "if",
    "when",
    "while",
    "until",
}


@dataclass
class TaskDecomposer:
    def decompose(self, query_plan: QueryPlan) -> List[Subgoal]:
        raw = query_plan.normalized_query
        clauses = self._split_into_clauses(raw)
        if not clauses:
            clauses = [raw]

        subgoals: List[Subgoal] = []
        previous_id = None
        for index, clause in enumerate(clauses):
            tokens = [token for token in clause.split() if token]
            required = {
                token
                for token in tokens
                if token in query_plan.required_capabilities or token in query_plan.optional_capabilities
            }
            if not required:
                required = set(tokens[:3])
            optional = set(tokens[3:])
            subgoal_id = f"sg_{index + 1}"
            hints = self._infer_environment_hints(clause, required)
            subgoals.append(
                Subgoal(
                    subgoal_id=subgoal_id,
                    description=clause,
                    required_capabilities=required,
                    optional_capabilities=optional,
                    depends_on=[previous_id] if previous_id else [],
                    priority=index,
                    environment_hints=hints,
                )
            )
            previous_id = subgoal_id
        return subgoals

    def _split_into_clauses(self, text: str) -> List[str]:
        normalized = re.sub(r"\s+", " ", text).strip()
        if not normalized:
            return []

        sentence_candidates = []
        for segment in re.split(r"[.;!?]", normalized):
            segment = segment.strip()
            if segment:
                sentence_candidates.append(segment)

        clauses: List[str] = []
        for sentence in sentence_candidates or [normalized]:
            tokens = sentence.split()
            current: List[str] = []
            for token in tokens:
                lowered = token.lower().strip(",")
                if lowered in CONNECTORS or lowered in LEADING_PHASE_MARKERS:
                    if current:
                        clauses.append(" ".join(current).strip())
                        current = []
                    # Keep conditional markers so the clause still carries its branch meaning.
                    if lowered in {"if", "when", "while", "until"}:
                        current.append(lowered)
                    continue
                current.append(token)
            if current:
                clauses.append(" ".join(current).strip())
        return [clause for clause in clauses if clause]

    def _infer_environment_hints(self, clause: str, capabilities: Set[str]) -> dict:
        hints = {}
        if any(token in clause for token in {"buy", "purchase", "price", "product"}):
            hints["domain"] = "webshop"
        elif any(token in clause for token in {"cool", "heat", "put", "take", "stoveburner"}):
            hints["domain"] = "alfworld"
        elif any(token in clause for token in {"test", "conductivity", "substance", "measure"}):
            hints["domain"] = "scienceworld"
        if "domain" not in hints and capabilities:
            hints["domain"] = "generic"
        return hints
