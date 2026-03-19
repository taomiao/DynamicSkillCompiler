from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Set

from skillnet_ai.compiler.models import QueryPlan, Subgoal
from skillnet_ai.compiler.query import QueryOptimizer


CONNECTORS = {"and", "then", "after", "before", "finally", "otherwise", "else"}
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

SCIENCEWORLD_HINT_TOKENS = {
    "substance", "boil", "melt", "freeze", "heat", "cool", "temperature",
    "thermometer", "conductivity", "measure", "focus", "contain", "apparatu",
    "monit", "wait", "react", "liquid", "solid", "vapor", "combust",
}
ALFWORLD_HINT_TOKENS = {
    "put", "take", "mov", "clean", "cool", "heat", "stoveburn", "garbagecan",
    "drawer", "cabinet", "receptacle", "pan", "fridge", "microwave",
}


@dataclass
class TaskDecomposer:
    optimizer: QueryOptimizer = field(default_factory=QueryOptimizer)

    def decompose(self, query_plan: QueryPlan) -> List[Subgoal]:
        raw = self.optimizer.normalize_text(
            query_plan.raw_query or query_plan.normalized_query,
            preserve_delimiters=True,
        )
        clauses = self._split_into_clauses(raw)
        if not clauses:
            clauses = [query_plan.normalized_query]

        subgoals: List[Subgoal] = []
        previous_id = None
        for index, clause in enumerate(clauses):
            tokens = self.optimizer.extract_content_terms(clause)
            token_set = set(tokens)
            required = {
                token
                for token in tokens
                if token in query_plan.required_capabilities or token in query_plan.optional_capabilities
            }
            if not required:
                required = set(tokens[:3]) if tokens else set(clause.split()[:3])
            optional = set(tokens[3:]) if len(tokens) > 3 else set()
            required |= self._augment_required_capabilities(clause, token_set)
            optional |= self._augment_optional_capabilities(clause, token_set)
            subgoal_id = f"sg_{index + 1}"
            hints = self._infer_environment_hints(clause, required | optional)
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

        for marker in ("first", "next", "then", "finally", "otherwise", "else", "if", "when", "while", "until"):
            normalized = re.sub(rf"\b{marker}\b", f". {marker}", normalized)

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
        elif capabilities & {"stoveburn", "garbagecan", "drawer", "cabinet", "pan"}:
            hints["domain"] = "alfworld"
        elif capabilities & SCIENCEWORLD_HINT_TOKENS:
            hints["domain"] = "scienceworld"
        elif capabilities & ALFWORLD_HINT_TOKENS or any(
            token in clause for token in {"stoveburner", "garbagecan", "drawer", "cabinet"}
        ):
            hints["domain"] = "alfworld"
        if "domain" not in hints and capabilities:
            hints["domain"] = "generic"
        return hints

    def _augment_required_capabilities(self, clause: str, tokens: Set[str]) -> Set[str]:
        augmented: Set[str] = set()
        if "focus" in tokens:
            augmented.add("focus")
        if self._is_phase_change_clause(clause, tokens):
            augmented |= {"heat", "contain", "apparatu"}
        if tokens & {"measure", "temperature"}:
            augmented |= {"measure", "monit"}
        return augmented

    def _augment_optional_capabilities(self, clause: str, tokens: Set[str]) -> Set[str]:
        augmented = set(self.optimizer.infer_structural_capabilities(tokens))
        if self._is_phase_change_clause(clause, tokens):
            augmented |= {"monit", "wait", "state", "chang", "substance"}
        if "combust" in tokens:
            augmented |= {"heat", "fallback"}
        return augmented

    def _is_phase_change_clause(self, clause: str, tokens: Set[str]) -> bool:
        if tokens & {"boil", "melt", "freeze", "heat", "cool", "combust"}:
            return True
        lowered = clause.lower()
        return "state of matter" in lowered or ("change" in lowered and "state" in lowered)
