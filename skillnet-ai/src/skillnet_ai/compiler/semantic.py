"""
SemanticSoftMatcher — description-level skill relevance scoring using dense embeddings.

Instead of comparing individual capability tokens (which produces poor signal with
sentence models), this module embeds the *full task description* against each skill's
*name + description* string and returns a cosine similarity in [0, 1].

The similarity is used as a semantic bonus on top of the keyword coverage score inside
SkillUtilityScorer._coverage():

    coverage_final = keyword_coverage + semantic_bonus(task, skill) * semantic_weight

This naturally handles vocabulary gaps that keyword expansion cannot:
  - "put a hot potato in fridge." → heat-object-with-appliance scores high
  - "put a clean knife in drawer." → clean-object scores high
  - "Find me machine wash, cold men's sweaters" → NO heating/science skills score high

Architecture
------------
- Embeds skill descriptions once at warm_up() time; results are persisted to a JSONL
  disk cache so repeat runs incur zero API cost for previously seen text.
- Task queries are embedded on first use and cached in memory (typically unique per run).
- All similarity computations use numpy for efficiency.
- Fails gracefully: if the API is unreachable, score() returns 0.0 and scoring falls
  back to pure keyword coverage.

Usage
-----
    matcher = SemanticSoftMatcher.from_openai(
        api_key="sk-...",
        base_url="http://...",           # optional
        cache_dir="~/.skillnet/emb",    # optional
    )

    # Pre-embed all skills once (call after loading the skill library).
    assets = LocalSkillLibraryRetriever(...).retrieve(None)
    matcher.warm_up_skills(assets)

    # Inject into the compiler; scoring is transparent.
    compiler = DynamicSkillCompiler(retriever=..., soft_matcher=matcher)

Tuning
------
semantic_bonus_weight (default 0.30):
    Controls how much the semantic similarity can boost a skill's coverage score.
    A value of 0.30 means a perfectly relevant skill (sim=1.0) gains +0.30 on its
    coverage score.  In practice scores are 0.25–0.50, so the effective bonus is
    +0.04 to +0.09 for the most relevant skills.

semantic_floor (default 0.20):
    Cosine similarities below this value are ignored (treated as background noise).
    Calibrated so that unrelated skills (sim ≈ 0.15–0.22) contribute nothing.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

if TYPE_CHECKING:
    from skillnet_ai.compiler.models import SkillAsset


class SemanticSoftMatcher:
    """Description-level semantic similarity scorer.

    Parameters
    ----------
    embed_fn:
        Callable accepting a list of strings and returning a list of float vectors.
    cache_dir:
        Directory for the persistent embedding cache. Pass None to skip disk caching.
    semantic_bonus_weight:
        Maximum additive bonus on keyword coverage score (clipped to keep total ≤ 1).
    semantic_floor:
        Cosine similarities below this threshold contribute zero bonus.
    """

    def __init__(
        self,
        embed_fn: Callable[[List[str]], List[List[float]]],
        cache_dir: Optional[str] = "~/.skillnet/emb",
        semantic_bonus_weight: float = 0.30,
        semantic_floor: float = 0.20,
    ) -> None:
        self._embed_fn = embed_fn
        self._bonus_weight = semantic_bonus_weight
        self._floor = semantic_floor
        # skill_id  → embedding vector
        self._skill_cache: Dict[str, List[float]] = {}
        # skill_id → canonical text used for embedding
        self._skill_text: Dict[str, str] = {}
        # task text → embedding vector  (in-memory only, tasks change each run)
        self._task_cache: Dict[str, List[float]] = {}
        self._cache_file: Optional[str] = None
        if cache_dir:
            self._cache_file = os.path.join(
                os.path.expanduser(cache_dir), "skill_embeddings.jsonl"
            )
            os.makedirs(os.path.dirname(self._cache_file), exist_ok=True)
            self._load_disk_cache()

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_openai(
        cls,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "text-embedding-3-small",
        cache_dir: str = "~/.skillnet/emb",
        **kwargs,
    ) -> "SemanticSoftMatcher":
        """Create a matcher backed by the OpenAI embeddings endpoint."""
        import openai

        client = openai.OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY", ""),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL") or None,
        )

        def _embed(texts: List[str]) -> List[List[float]]:
            resp = client.embeddings.create(model=model, input=texts)
            return [item.embedding for item in resp.data]

        return cls(embed_fn=_embed, cache_dir=cache_dir, **kwargs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def warm_up_skills(self, assets: list) -> None:
        """Pre-embed all skill descriptions in one batched API call.

        Call once after loading the skill library.  Subsequent calls with the same
        skill IDs hit the in-memory cache instantly.

        Parameters
        ----------
        assets : list[SkillAsset]
        """
        to_embed_ids: List[str] = []
        to_embed_texts: List[str] = []
        for asset in assets:
            text = _skill_text(asset)
            # Re-embed if the description has changed since last cache write.
            if asset.skill_id not in self._skill_cache or self._skill_text.get(asset.skill_id) != text:
                to_embed_ids.append(asset.skill_id)
                to_embed_texts.append(text)
                self._skill_text[asset.skill_id] = text

        if to_embed_ids:
            self._batch_embed_skills(to_embed_ids, to_embed_texts)

    def score_bonus(self, task_query: str, skill_asset) -> float:
        """Return a semantic bonus in [0, semantic_bonus_weight].

        The bonus is added to the keyword coverage score.  Skills whose description
        is semantically relevant to the task receive a non-zero bonus; irrelevant
        skills receive 0.

        Parameters
        ----------
        task_query : str
            The raw task text (e.g. "put a hot potato in fridge.").
        skill_asset : SkillAsset
        """
        sim = self._similarity(task_query, skill_asset)
        above_floor = max(0.0, sim - self._floor)
        scale = 1.0 - self._floor  # map [floor, 1.0] → [0, 1]
        normalised = above_floor / scale if scale > 0 else 0.0
        return normalised * self._bonus_weight

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _similarity(self, task_query: str, skill_asset) -> float:
        """Cosine similarity between task text and skill description text."""
        skill_id = skill_asset.skill_id
        skill_emb = self._skill_cache.get(skill_id)
        if skill_emb is None:
            # Skill wasn't warmed up; embed on-the-fly.
            text = _skill_text(skill_asset)
            self._skill_text[skill_id] = text
            self._batch_embed_skills([skill_id], [text])
            skill_emb = self._skill_cache.get(skill_id)
        if skill_emb is None:
            return 0.0

        task_emb = self._task_cache.get(task_query)
        if task_emb is None:
            try:
                result = self._embed_fn([task_query])
                task_emb = result[0]
                self._task_cache[task_query] = task_emb
            except Exception:
                return 0.0

        return _cosine(task_emb, skill_emb)

    def _batch_embed_skills(
        self, skill_ids: List[str], texts: List[str], chunk_size: int = 100
    ) -> None:
        for i in range(0, len(texts), chunk_size):
            batch_ids = skill_ids[i : i + chunk_size]
            batch_texts = texts[i : i + chunk_size]
            try:
                embeddings = self._embed_fn(batch_texts)
                for sid, emb in zip(batch_ids, embeddings):
                    self._skill_cache[sid] = emb
                    self._append_to_disk(sid, self._skill_text.get(sid, ""), emb)
            except Exception:
                pass

    def _load_disk_cache(self) -> None:
        if not self._cache_file or not os.path.exists(self._cache_file):
            return
        try:
            with open(self._cache_file, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    sid = entry.get("skill_id")
                    emb = entry.get("embedding")
                    text = entry.get("text", "")
                    if sid and emb:
                        self._skill_cache[sid] = emb
                        self._skill_text[sid] = text
        except Exception:
            pass

    def _append_to_disk(self, skill_id: str, text: str, embedding: List[float]) -> None:
        if not self._cache_file:
            return
        try:
            with open(self._cache_file, "a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps(
                        {"skill_id": skill_id, "text": text, "embedding": embedding}
                    )
                    + "\n"
                )
        except Exception:
            pass


# ------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------

def _skill_text(asset) -> str:
    """Canonical text representation of a skill for embedding."""
    return f"{asset.name}: {asset.description}"


def _cosine(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors."""
    if _HAS_NUMPY:
        va = np.array(a, dtype=np.float32)
        vb = np.array(b, dtype=np.float32)
        denom = (np.linalg.norm(va) * np.linalg.norm(vb))
        return float(va @ vb / (denom + 1e-9))
    # Pure-Python fallback
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb + 1e-9)
