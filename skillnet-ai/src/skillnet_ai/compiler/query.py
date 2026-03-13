from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Set

from skillnet_ai.compiler.models import QueryPlan


STOPWORDS = {
    "a", "an", "and", "are", "at", "be", "by", "for", "from", "how", "i", "in",
    "into", "is", "it", "of", "on", "or", "please", "that", "the", "this", "to",
    "using", "with", "your", "task", "first", "next", "then", "located", "around",
    "degrees", "degree", "celsius", "called", "which", "if",
}

INTENT_KEYWORDS: Dict[str, Set[str]] = {
    "analyze": {"analyze", "inspect", "understand", "study"},
    "build": {"build", "create", "develop", "generate", "implement"},
    "retrieve": {"find", "search", "lookup", "query"},
    "transform": {"convert", "compile", "rewrite", "refactor", "optimize"},
    "evaluate": {"evaluate", "verify", "test", "benchmark", "compare"},
    "localize": {"local", "localize", "workspace", "environment"},
}

CONSTRAINT_KEYWORDS = {
    "low_latency": {"fast", "efficient", "latency", "real-time"},
    "low_token": {"token", "cheap", "cost", "budget"},
    "safe": {"safe", "secure", "sandbox"},
    "local_only": {"offline", "local", "on-prem"},
}

SYNONYM_EXPANSIONS: Dict[str, Set[str]] = {
    "put": {"place", "placer", "move", "transfer"},
    "place": {"put", "placer", "move"},
    "cool": {"cooler", "cooling", "temperature", "cold"},
    "heat": {"heater", "heating", "temperature", "warm"},
    "pick": {"take", "picker", "retrieve", "retriever"},
    "buy": {"purchase", "checkout", "order"},
    "purchase": {"buy", "checkout", "order"},
    "find": {"search", "locate", "identify"},
    "analyze": {"analysis", "inspect", "reason"},
    "test": {"evaluate", "measure", "verify"},
}


@dataclass
class QueryOptimizer:
    max_keyword_terms: int = 6

    def optimize(self, query: str) -> QueryPlan:
        normalized = self._normalize(query)
        tokens = [token for token in normalized.split() if token not in STOPWORDS]
        expanded_tokens = self._expand_tokens(tokens)
        keyword_terms = expanded_tokens[: self.max_keyword_terms]
        intents = self._infer_intents(set(expanded_tokens))
        constraints = self._infer_constraints(set(expanded_tokens))
        required = set(expanded_tokens)
        optional = self._expand_optional(expanded_tokens)
        semantic_queries = self._build_semantic_queries(normalized, keyword_terms, intents)

        return QueryPlan(
            raw_query=query,
            normalized_query=normalized,
            keyword_query=" ".join(keyword_terms),
            semantic_queries=semantic_queries,
            intents=intents,
            required_capabilities=required,
            optional_capabilities=optional,
            constraints=constraints,
        )

    def _normalize(self, query: str) -> str:
        collapsed = re.sub(r"[^a-zA-Z0-9_\-/ ]+", " ", query.lower())
        collapsed = re.sub(r"\b\d+(?:\.\d+)?\b", " ", collapsed)
        return re.sub(r"\s+", " ", collapsed).strip()

    def _infer_intents(self, tokens: Set[str]) -> List[str]:
        matched = [
            intent
            for intent, keywords in INTENT_KEYWORDS.items()
            if tokens & keywords
        ]
        return matched or ["build"]

    def _infer_constraints(self, tokens: Set[str]) -> Dict[str, str]:
        constraints: Dict[str, str] = {}
        for key, keywords in CONSTRAINT_KEYWORDS.items():
            if tokens & keywords:
                constraints[key] = "true"
        return constraints

    def _expand_optional(self, tokens: List[str]) -> Set[str]:
        optional = set(tokens)
        for left, right in zip(tokens, tokens[1:]):
            optional.add(f"{left}_{right}")
        return optional

    def _expand_tokens(self, tokens: List[str]) -> List[str]:
        expanded: List[str] = []
        seen = set()
        for token in tokens:
            if token not in seen and token not in STOPWORDS:
                seen.add(token)
                expanded.append(token)
        for token in tokens:
            for candidate in self._variants_for_token(token)[1:]:
                if candidate not in seen and candidate not in STOPWORDS:
                    seen.add(candidate)
                    expanded.append(candidate)
        return expanded

    def _variants_for_token(self, token: str) -> List[str]:
        variants = [token]
        if token.endswith("ing") and len(token) > 4:
            variants.append(token[:-3])
        if token.endswith("ed") and len(token) > 3:
            variants.append(token[:-2])
        if token.endswith("es") and len(token) > 4 and not token.endswith("sses"):
            variants.append(token[:-2])
        elif token.endswith("s") and len(token) > 3 and not token.endswith(("ss", "us")):
            variants.append(token[:-1])
        if token in SYNONYM_EXPANSIONS:
            variants.extend(sorted(SYNONYM_EXPANSIONS[token]))
        return variants

    def _build_semantic_queries(
        self,
        normalized: str,
        keyword_terms: List[str],
        intents: List[str],
    ) -> List[str]:
        semantic = [normalized]
        if keyword_terms:
            semantic.append(" ".join(keyword_terms))
        if intents and keyword_terms:
            semantic.append(f"{' '.join(intents)} {' '.join(keyword_terms[:4])}")
        return list(dict.fromkeys([item for item in semantic if item]))
