from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


RELATION_TYPES = {"similar_to", "belong_to", "compose_with", "depend_on"}


@dataclass(frozen=True)
class SkillRelation:
    source: str
    target: str
    relation_type: str
    weight: float = 1.0
    reason: str = ""

    def __post_init__(self) -> None:
        if self.relation_type not in RELATION_TYPES:
            raise ValueError(f"Unsupported relation_type: {self.relation_type}")


@dataclass
class SkillAsset:
    skill_id: str
    name: str
    description: str
    category: Optional[str] = None
    source: str = "unknown"
    location: Optional[str] = None
    capabilities: Set[str] = field(default_factory=set)
    dependencies: Set[str] = field(default_factory=set)
    contains: Set[str] = field(default_factory=set)
    composes_with: Set[str] = field(default_factory=set)
    similar_to: Set[str] = field(default_factory=set)
    quality_scores: Dict[str, float] = field(default_factory=dict)
    token_cost: float = 0.0
    execution_cost: float = 0.0
    latency_ms: float = 0.0
    instructions: List[str] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)

    def normalized_capabilities(self) -> Set[str]:
        if self.capabilities:
            return {cap.strip().lower() for cap in self.capabilities if cap.strip()}
        text = f"{self.name} {self.description}".lower()
        tokens = {
            token.strip(".,:;()[]{}")
            for token in text.split()
            if len(token.strip(".,:;()[]{}")) > 2
        }
        return {token for token in tokens if token}

    def mean_quality(self) -> float:
        if not self.quality_scores:
            return 0.5
        return sum(self.quality_scores.values()) / len(self.quality_scores)


@dataclass
class QueryPlan:
    raw_query: str
    normalized_query: str
    keyword_query: str
    semantic_queries: List[str]
    intents: List[str]
    required_capabilities: Set[str]
    optional_capabilities: Set[str]
    constraints: Dict[str, str] = field(default_factory=dict)


@dataclass
class Subgoal:
    subgoal_id: str
    description: str
    required_capabilities: Set[str]
    optional_capabilities: Set[str] = field(default_factory=set)
    depends_on: List[str] = field(default_factory=list)
    priority: int = 0
    environment_hints: Dict[str, str] = field(default_factory=dict)


@dataclass
class SkillFragment:
    fragment_id: str
    skill_id: str
    title: str
    content: str
    capabilities: Set[str]
    action_schema: Optional[str] = None
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)
    example_actions: List[str] = field(default_factory=list)
    token_cost: float = 0.0
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class SkillGraph:
    skills: Dict[str, SkillAsset]
    relations: List[SkillRelation]

    def neighbors(self, skill_id: str, relation_type: Optional[str] = None) -> List[str]:
        return [
            relation.target
            for relation in self.relations
            if relation.source == skill_id
            and (relation_type is None or relation.relation_type == relation_type)
        ]


@dataclass
class LocalEnvironment:
    cwd: str = "."
    workspace_root: str = "."
    python_bin: str = "python"
    shell: str = "sh"
    os_name: str = "unknown"
    available_bins: Set[str] = field(default_factory=set)
    env_vars: Dict[str, str] = field(default_factory=dict)
    benchmark: str = "generic"
    action_space: Set[str] = field(default_factory=set)
    object_vocabulary: Set[str] = field(default_factory=set)


@dataclass
class CompiledSkill:
    asset: SkillAsset
    selected_fragments: List[SkillFragment]
    assigned_subgoals: List[str]
    localized_instructions: List[str]
    utility_score: float
    selected_reason: str


@dataclass
class CompilationMetrics:
    candidate_count: int = 0
    selected_count: int = 0
    edge_count_before: int = 0
    edge_count_after: int = 0
    estimated_token_cost_before: float = 0.0
    estimated_token_cost_after: float = 0.0
    estimated_execution_cost_before: float = 0.0
    estimated_execution_cost_after: float = 0.0
    coverage_score: float = 0.0
    redundancy_reduction: float = 0.0
    subgoal_count: int = 0
    covered_subgoal_count: int = 0
    fragment_count_before: int = 0
    fragment_count_after: int = 0
    fragment_token_cost_after: float = 0.0


@dataclass
class CompilerPassTrace:
    pass_name: str
    before_selected: List[str] = field(default_factory=list)
    after_selected: List[str] = field(default_factory=list)
    added: List[str] = field(default_factory=list)
    removed: List[str] = field(default_factory=list)
    dropped_delta: Dict[str, str] = field(default_factory=dict)


@dataclass
class CompiledSkillPackage:
    query_plan: QueryPlan
    subgoals: List[Subgoal]
    graph: SkillGraph
    compiled_skills: List[CompiledSkill]
    execution_order: List[str]
    metrics: CompilationMetrics
    dropped_skills: Dict[str, str]
    notes: List[str] = field(default_factory=list)
    pass_traces: List[CompilerPassTrace] = field(default_factory=list)
