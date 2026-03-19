# Dynamic Skill Compiler Design

## Goal

Dynamic Skill Compiler (DSC) adds a task-driven compilation layer on top of SkillNet. Instead of directly using all retrieved skills, DSC compiles a query-specific, dependency-complete, low-redundancy, localized skill package.

## Compiler Pipeline

1. Query optimization
   - Normalize the raw query.
   - Extract compact keyword queries for retrieval.
   - Generate semantic queries for broader recall.
   - Infer intents and constraints such as low-token, local-only, or evaluation-heavy tasks.

2. Query understanding
   - Convert the query into a `QueryPlan`.
   - Separate required capabilities from optional capabilities.
   - Preserve task constraints for downstream scoring.

3. Skill retrieval
   - Retrieve candidates from the local skill library.
   - Retrieve candidates from SkillNet search.
   - Merge and deduplicate candidates into a unified pool.

4. Skill graph construction
   - Preserve declared `depend_on`, `belong_to`, `compose_with`, `similar_to` edges.
   - Infer additional `similar_to` edges via capability overlap.
   - Build a multi-relational graph over all candidates.

5. Task-aware compilation
   - Score each candidate by task coverage, quality, cost, and latency.
   - Remove low-relevance nodes.
   - Collapse `similar_to` clusters by keeping the best utility path.
   - Prune broad parent skills when more specific children satisfy the task.
   - Reintroduce required dependencies to maintain executability.
   - Trim isolated low-value nodes.

6. Localization
   - Rewrite generic commands and placeholders against the active workspace.
   - Resolve `{cwd}`, `{workspace_root}`, and Python binary selection.
   - Prepare environment-specific instructions for execution.

7. Compilation output
   - A compiled subgraph.
   - An execution order.
   - A dropped-skill audit trail.
   - Cost, redundancy, and coverage metrics.

## Intended Gains Over SkillNet Baseline

Compared with the SkillNet baseline, DSC targets gains in:

- Safety and executability
  - Dependency closure after pruning prevents broken packages.
- Completeness
  - Task coverage is explicitly measured before finalizing the package.
- Maintainability
  - Redundant overlapping skills are removed, reducing graph sprawl.
- Cost-awareness
  - Candidate selection prefers lower token and lower execution cost skills when coverage is similar.
- Token efficiency
  - The compiler removes unrelated, overlapping, and overly broad skills before packaging.

## Current Implementation Scope

The current implementation provides:

- Query optimizer
- In-memory, local-library, and SkillNet-search retrievers
- Multi-relational graph builder
- Heuristic compiler with redundancy pruning and dependency repair
- Environment localization
- Unit tests for the core compiler logic

## Next Experimental Upgrades

To make the paper stronger, the next versions should add:

1. Learned or LLM-based query decomposition
2. Retrieval reranking with environment-aware signals
3. Exact subgraph optimization instead of greedy pruning
4. Token-level prompt compression for compiled skills
5. Benchmark integration into ALFWorld, ScienceWorld, and WebShop
6. Ablation studies:
   - no query optimization
   - no graph pruning
   - no localization
   - no dependency repair
   - no token-aware scoring

## Suggested Paper Framing

- Baseline: SkillNet retrieval + direct use of top-k skills
- Method: Dynamic Skill Compiler
- Main claim: compile-time optimization of skill graphs improves quality and token efficiency
- Key measured outcomes:
  - task success
  - average evaluation score over the five SkillNet dimensions
  - prompt token usage
  - selected skill count
  - graph density after compilation
  - execution latency
