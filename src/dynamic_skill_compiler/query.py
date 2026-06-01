from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Set

from dynamic_skill_compiler.models import QueryPlan


STOPWORDS = {
    "a", "an", "and", "are", "at", "be", "by", "for", "from", "how", "i", "in",
    "into", "is", "it", "of", "on", "or", "please", "that", "the", "this", "to",
    "using", "with", "your", "task", "first", "next", "then", "located", "around",
    "degrees", "degree", "celsius", "called", "which", "if", "also", "will",
    "acceptable", "without", "actions", "action",
}

LOW_SIGNAL_CAPABILITIES = {
    "acceptable", "action", "also", "compound", "compounds", "its", "matter",
    "point", "without", "will", "cause", "caus", "matt",
}

INTENT_KEYWORDS: Dict[str, Set[str]] = {
    "analyze": {"analyze", "inspect", "understand", "study"},
    "build": {"build", "create", "develop", "generate", "implement"},
    "retrieve": {"find", "search", "lookup", "query"},
    "transform": {
        "convert", "compile", "rewrite", "refactor", "optimize", "boil",
        "melt", "freeze", "chang", "transform",
    },
    "evaluate": {
        "evaluate", "verify", "test", "benchmark", "compare", "measure",
        "monitor", "temperature", "threshold",
    },
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
    # "hot" / "cold" are state adjectives that must map to the corresponding
    # action-skill vocabulary.
    "hot": {"heat", "heater", "heating", "warm", "cook", "microwave"},
    "cold": {"cool", "cooler", "cooling", "fridge", "freez"},
    "boil": {"heat", "vaporize", "phase", "state"},
    "melt": {"heat", "liquid", "phase", "state"},
    "freeze": {"cool", "solid", "phase", "state"},
    # "clean" / "wash" are precondition verbs that should map to transform skills.
    "clean": {"wash", "sink", "sinkbasin", "rinse"},
    "wash": {"clean", "sink", "sinkbasin"},
    "pick": {"take", "picker", "retrieve", "retriever"},
    "buy": {"purchase", "checkout", "order"},
    "purchase": {"buy", "checkout", "order"},
    "find": {"search", "locate", "identify"},
    "analyze": {"analysis", "inspect", "reason"},
    "test": {"evaluate", "measure", "verify"},
    "focus": {"attention", "target"},
    "measure": {"monitor", "temperature", "threshold"},
    "container": {"pot", "beaker", "cup", "jar"},
    "monitor": {"observe", "check", "verify", "wait"},
    # Chemistry/synthesis tasks require mixing; expand so mixture-creator scores correctly.
    "chemistry": {"mix", "mixture", "react", "combine", "synthes"},
    "create": {"mix", "produc", "synthes"},
    # Lifespan comparison requires identification and ranking.
    "lifespan": {"life", "span", "compar", "longest", "shortest", "identif"},
    "life": {"lifespan", "span"},
    "span": {"lifespan", "life"},
    # Growth tasks require planting and monitoring.
    "grow": {"plant", "seed", "growth", "sprout"},
    "growth": {"grow", "plant", "seed", "sprout"},
}

PHRASE_CAPABILITY_RULES = (
    (r"\bstate of matter\b", ("state", "change", "phase", "substance")),
    (r"\bchange (?:its |the )?state(?: of matter)?\b", ("change", "phase", "transform")),
    (r"\bboiling point\b", ("boil",)),
    (r"\bmelting point\b", ("melt",)),
    (r"\bfreezing point\b", ("freeze",)),
    (r"\bwithout (?:a )?boiling point\b", ("combust", "fallback")),
    (r"\bfocus on\b", ("focus",)),
)


@dataclass
class QueryOptimizer:
    max_keyword_terms: int = 6

    def optimize(self, query: str) -> QueryPlan:
        normalized = self.normalize_text(query)
        expanded_tokens = self.extract_content_terms(query)
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

    def normalize_text(self, query: str, preserve_delimiters: bool = False) -> str:
        if preserve_delimiters:
            collapsed = re.sub(r"[^a-zA-Z0-9_\-/.,;!? ]+", " ", query.lower())
        else:
            collapsed = re.sub(r"[^a-zA-Z0-9_\-/ ]+", " ", query.lower())
        collapsed = re.sub(r"\b\d+(?:\.\d+)?\b", " ", collapsed)
        return re.sub(r"\s+", " ", collapsed).strip()

    def extract_content_terms(self, query: str) -> List[str]:
        normalized = self.normalize_text(query)
        tokens = self.normalize_capability_tokens(normalized.split())
        phrase_tokens = self._extract_phrase_capabilities(query)
        base_terms = list(dict.fromkeys(tokens + phrase_tokens))
        return self._expand_tokens(base_terms)

    def normalize_capability_tokens(self, tokens: List[str]) -> List[str]:
        normalized_tokens: List[str] = []
        for token in tokens:
            normalized = self.normalize_capability_token(token)
            if normalized:
                normalized_tokens.append(normalized)
        return normalized_tokens

    def normalize_capability_token(self, token: str) -> str:
        token = token.strip().lower()
        if not token:
            return ""
        if "_" in token:
            parts = [self.normalize_capability_token(part) for part in token.split("_")]
            parts = [part for part in parts if part]
            if not parts:
                return ""
            return "_".join(parts) if len(parts) > 1 else parts[0]
        if token in STOPWORDS or len(token) <= 2:
            return ""
        for suffix in ("ing", "ers", "er", "or", "ion", "ed", "es", "s"):
            if token.endswith(suffix) and len(token) > len(suffix) + 2:
                token = token[: -len(suffix)]
                break
        if token in STOPWORDS or token in LOW_SIGNAL_CAPABILITIES or len(token) <= 2:
            return ""
        return token

    def infer_structural_capabilities(self, tokens: Set[str]) -> Set[str]:
        inferred: Set[str] = set()
        phase_change_signal = bool(tokens & {"boil", "melt", "freeze", "heat", "cool", "state", "chang"})
        if phase_change_signal:
            inferred |= {"contain", "apparatu", "monit", "wait"}
        if tokens & {"temperature", "measure", "threshold"}:
            inferred |= {"monit", "examine"}
        if tokens & {"substance", "liquid", "solid", "vapor"}:
            inferred |= {"contain"}
        if tokens & {"combust"}:
            inferred |= {"heat", "apparatu"}
        return inferred

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
        optional |= self.infer_structural_capabilities(set(tokens))
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
        if token in SYNONYM_EXPANSIONS:
            variants.extend(
                normalized
                for normalized in (
                    self.normalize_capability_token(candidate)
                    for candidate in sorted(SYNONYM_EXPANSIONS[token])
                )
                if normalized
            )
        return variants

    def _extract_phrase_capabilities(self, query: str) -> List[str]:
        lowered = query.lower()
        extracted: List[str] = []
        for pattern, capabilities in PHRASE_CAPABILITY_RULES:
            if re.search(pattern, lowered):
                extracted.extend(
                    normalized
                    for normalized in (
                        self.normalize_capability_token(capability)
                        for capability in capabilities
                    )
                    if normalized
                )
        return list(dict.fromkeys(extracted))

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
