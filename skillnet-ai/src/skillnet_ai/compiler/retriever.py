from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol

from skillnet_ai.compiler.models import QueryPlan, SkillAsset
from skillnet_ai.searcher import SkillNetSearcher


GENERIC_CAPABILITY_STOPWORDS = {
    "scienceworld", "alfworld", "webshop", "skill", "skills", "this", "that", "these",
    "should", "when", "where", "with", "from", "into", "onto", "then", "than", "them",
    "they", "their", "have", "has", "had", "using", "used", "user", "agent", "output",
    "result", "results", "task", "tasks", "environment", "workflow", "step", "steps",
    "action", "actions", "object", "objects", "item", "items", "room", "rooms",
    "located", "around", "need", "needs", "must", "will", "can", "could", "would",
    "such", "only", "also", "after", "before", "during", "through", "between", "within",
    "across", "under", "over", "into", "your", "ours", "theirs", "first", "next",
    "and", "the", "for", "are", "but", "not", "you", "our", "its", "it", "all", "any",
}


class SkillRetriever(Protocol):
    def retrieve(self, query_plan: QueryPlan) -> List[SkillAsset]:
        ...


class InMemorySkillRetriever:
    def __init__(self, skills: List[SkillAsset]):
        self.skills = skills

    def retrieve(self, query_plan: QueryPlan) -> List[SkillAsset]:
        return list(self.skills)


@dataclass
class CompositeSkillRetriever:
    retrievers: List[SkillRetriever] = field(default_factory=list)

    def retrieve(self, query_plan: QueryPlan) -> List[SkillAsset]:
        merged: Dict[str, SkillAsset] = {}
        for retriever in self.retrievers:
            for skill in retriever.retrieve(query_plan):
                merged.setdefault(skill.skill_id, skill)
        return list(merged.values())


@dataclass
class SkillNetSearchRetriever:
    searcher: SkillNetSearcher = field(default_factory=SkillNetSearcher)
    limit: int = 20
    threshold: float = 0.7

    def retrieve(self, query_plan: QueryPlan) -> List[SkillAsset]:
        skills: Dict[str, SkillAsset] = {}
        for mode, query in self._queries(query_plan):
            try:
                results = self.searcher.search(
                    q=query,
                    mode=mode,
                    limit=self.limit,
                    threshold=self.threshold,
                )
            except Exception:
                continue

            for result in results:
                skill = SkillAsset(
                    skill_id=self._safe_id(getattr(result, "skill_url", None), getattr(result, "skill_name", "skill")),
                    name=getattr(result, "skill_name", "unknown"),
                    description=getattr(result, "skill_description", "") or "",
                    category=getattr(result, "category", None),
                    source="skillnet_search",
                    location=getattr(result, "skill_url", None),
                    capabilities=self._extract_capabilities(
                        getattr(result, "skill_name", ""),
                        getattr(result, "skill_description", "") or "",
                    ),
                    quality_scores=self._quality_scores(getattr(result, "evaluation", None)),
                    token_cost=0.0,
                    execution_cost=max(0.0, 10.0 - float(getattr(result, "stars", 0)) / 50.0),
                    latency_ms=200.0,
                    instructions=[getattr(result, "skill_description", "") or ""],
                )
                skills.setdefault(skill.skill_id, skill)
        return list(skills.values())

    def _queries(self, query_plan: QueryPlan) -> List[tuple]:
        queries = []
        if query_plan.keyword_query:
            queries.append(("keyword", query_plan.keyword_query))
        for item in query_plan.semantic_queries:
            queries.append(("vector", item))
        deduped = []
        seen = set()
        for query in queries:
            if query in seen:
                continue
            seen.add(query)
            deduped.append(query)
        return deduped

    def _safe_id(self, location: Optional[str], name: str) -> str:
        return re.sub(r"[^a-z0-9_.-]+", "-", (location or name).lower()).strip("-")

    def _extract_capabilities(self, name: str, description: str) -> set[str]:
        return _extract_capabilities(name, description)

    def _quality_scores(self, evaluation: Optional[dict]) -> Dict[str, float]:
        if not isinstance(evaluation, dict):
            return {}
        mapping = {"good": 1.0, "average": 0.6, "poor": 0.2}
        scores: Dict[str, float] = {}
        for key, value in evaluation.items():
            if not isinstance(value, dict):
                continue
            level = str(value.get("level", "")).lower()
            if level in mapping:
                scores[key] = mapping[level]
        return scores


@dataclass
class LocalSkillLibraryRetriever:
    skills_dir: str

    def retrieve(self, query_plan: QueryPlan) -> List[SkillAsset]:
        if not os.path.isdir(self.skills_dir):
            return []

        assets: Dict[str, SkillAsset] = {}
        relationships = self._load_relationships()

        for entry in os.scandir(self.skills_dir):
            if not entry.is_dir():
                continue
            skill_md = os.path.join(entry.path, "SKILL.md")
            description = self._read_description(skill_md)
            instructions = self._read_instructions(skill_md, description)
            asset = SkillAsset(
                skill_id=entry.name,
                name=entry.name,
                description=description,
                source="local_library",
                location=entry.path,
                capabilities=self._extract_capabilities(entry.name, description, instructions),
                token_cost=self._estimate_token_cost(skill_md),
                execution_cost=1.0,
                latency_ms=50.0,
                instructions=instructions,
            )
            assets[asset.skill_id] = asset

        for relation in relationships:
            source = relation.get("source")
            target = relation.get("target")
            relation_type = relation.get("type")
            if source not in assets or target not in assets:
                continue
            if relation_type == "depend_on":
                assets[source].dependencies.add(target)
            elif relation_type == "compose_with":
                assets[source].composes_with.add(target)
            elif relation_type == "similar_to":
                assets[source].similar_to.add(target)
            elif relation_type == "belong_to":
                assets[target].contains.add(source)

        return list(assets.values())

    def _load_relationships(self) -> List[dict]:
        path = os.path.join(self.skills_dir, "relationships.json")
        if not os.path.isfile(path):
            return []
        try:
            with open(path, "r", encoding="utf-8") as file:
                data = json.load(file)
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _read_description(self, skill_md: str) -> str:
        if not os.path.isfile(skill_md):
            return "No description provided."
        try:
            with open(skill_md, "r", encoding="utf-8", errors="ignore") as file:
                content = file.read()
        except Exception:
            return "No description provided."
        frontmatter = re.search(r"^---\n(.*?)\n---", content, re.DOTALL)
        if frontmatter:
            desc = re.search(r"description:\s*(.+)$", frontmatter.group(1), re.MULTILINE)
            if desc:
                return desc.group(1).strip().strip("'\"")
        for line in content.splitlines():
            cleaned = line.strip()
            if cleaned and not cleaned.startswith("#"):
                return cleaned
        return "No description provided."

    def _read_instructions(self, skill_md: str, description: str) -> List[str]:
        if not os.path.isfile(skill_md):
            return [description]
        try:
            with open(skill_md, "r", encoding="utf-8", errors="ignore") as file:
                content = file.read()
        except Exception:
            return [description]

        body = re.sub(r"^---\n.*?\n---\n?", "", content, flags=re.DOTALL)
        instructions: List[str] = []
        for raw_line in body.splitlines():
            cleaned = raw_line.strip()
            if not cleaned:
                continue
            if cleaned.startswith("#"):
                continue
            cleaned = re.sub(r"^[-*]\s*", "", cleaned)
            cleaned = re.sub(r"^\d+\.\s*", "", cleaned)
            cleaned = cleaned.replace("**", "").replace("`", "")
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            if not cleaned or cleaned.lower() == description.lower():
                continue
            instructions.append(cleaned)

        if not instructions:
            return [description]

        merged = [description]
        for instruction in instructions:
            if instruction not in merged:
                merged.append(instruction)
            if len(merged) >= 14:
                break
        return merged

    def _extract_capabilities(self, name: str, description: str, instructions: List[str] | None = None) -> set[str]:
        return _extract_capabilities_from_skill(name, description, instructions)

    def _estimate_token_cost(self, skill_md: str) -> float:
        try:
            with open(skill_md, "r", encoding="utf-8", errors="ignore") as file:
                content = file.read()
        except Exception:
            return 0.0
        return max(1.0, len(content.split()) / 50.0)


# Compound word expansions: maps a token found in skill text to additional capability tokens
# that the QueryOptimizer would produce for the equivalent task phrase.
#
# Two categories:
#   1. Compound split: "lifespan" (1 word in skill) → {"life","span"} (2 words in task query)
#   2. Form canonicalization: "chemically" (skill) → {"chemistry"} (task query canonical form)
#
# All target tokens use the SAME normalization rules as QueryOptimizer.normalize_capability_token
# so they will match task required_capabilities without extra transformation.
_COMPOUND_SPLITS: dict[str, tuple[str, ...]] = {
    # Compound splits (one skill word → multiple task words)
    "lifespan": ("life", "span"),
    "lifecycle": ("life", "cycle"),
    "nonliving": ("non", "living"),
    "nonrenewable": ("non", "renewable"),
    # Chemistry variant forms → canonical "chemistry" (what QueryOptimizer produces from task)
    "chemically": ("chemistry",),
    "chemical": ("chemistry",),
    "chemicals": ("chemistry",),
    # Electrical variants → "electric"
    "electrical": ("electric",),
    "electricity": ("electric",),
    # Comparative adjective stems → base form that task uses
    "shortest": ("short",),
    "longest": ("long",),
    "smallest": ("small",),
    "largest": ("large",),
    "youngest": ("young",),
    "oldest": ("old",),
    # Growth/plant action forms
    "planted": ("plant",),
    "planting": ("plant",),
    "growing": ("grow",),
    "seeding": ("seed",),
    # Paint-mixing variants
    "mixing": ("mix",),
    "mixture": ("mix",),
    "mixed": ("mix",),
}


def _normalize_token(token: str) -> str:
    """Normalize a single token using the same suffix rules as QueryOptimizer.

    Keeping both normalizers in sync ensures that tokens extracted from skill text
    will match the required_capabilities produced by QueryOptimizer for task queries.
    """
    token = token.strip().lower()
    if len(token) <= 2 or token in GENERIC_CAPABILITY_STOPWORDS:
        return ""
    for suffix in ("ing", "ers", "er", "or", "ion", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            token = token[: -len(suffix)]
            break
    return token if len(token) > 2 and token not in GENERIC_CAPABILITY_STOPWORDS else ""


def _expand_compounds(token: str) -> list[str]:
    """Return the token plus any component parts from the compound-split table."""
    parts = _COMPOUND_SPLITS.get(token)
    if parts:
        return [token] + list(parts)
    return [token]


def _extract_capabilities(name: str, description: str) -> set[str]:
    return _extract_capabilities_from_skill(name, description, None)


def _extract_capabilities_from_skill(
    name: str,
    description: str,
    instructions: list[str] | None,
) -> set[str]:
    """Extract capability tokens from skill name, description, and optionally instruction body.

    Using instruction text (up to the first 8 lines) broadens the vocabulary so that
    task tokens that don't appear in the short description can still match the skill.
    Compound words like 'lifespan' are split into parts ('life', 'span') so they match
    task queries that spell them as separate words.
    """
    # Combine name + description + leading instruction lines for richer vocabulary.
    instruction_excerpt = ""
    if instructions:
        instruction_excerpt = " ".join(instructions[:8])
    text = f"{name} {description} {instruction_excerpt}".lower().replace("-", " ").replace("_", " ")
    raw_tokens = [
        token for token in re.split(r"[^a-z0-9]+", text)
        if token and not token.isdigit()
    ]

    # Normalize tokens and expand compound words.
    normalized_tokens: list[str] = []
    for token in raw_tokens:
        for expanded in _expand_compounds(token):
            normalized = _normalize_token(expanded)
            if normalized:
                normalized_tokens.append(normalized)

    capabilities = set(normalized_tokens)
    # Add adjacent bigrams for multi-word concept matching.
    for left, right in zip(normalized_tokens, normalized_tokens[1:]):
        if left != right:
            capabilities.add(f"{left}_{right}")
    return capabilities
