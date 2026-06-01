# Dynamic Skill Compiler Design

## Goal

Dynamic Skill Compiler (DSC) is a task-driven compiler for local agent skill libraries. Instead of passing a raw top-k skill list into the agent, DSC builds a task-specific skill graph and emits a compact, dependency-aware package that is easier for the executor to use.

## Compiler Pipeline

1. Query optimization
   - Normalize the raw task.
   - Extract compact retrieval queries.
   - Generate semantic queries for broader recall.
   - Infer constraints such as low-token, local-only, or evaluation-heavy tasks.

2. Query understanding
   - Convert the task into a `QueryPlan`.
   - Separate required capabilities from optional capabilities.
   - Preserve benchmark and environment constraints for downstream scoring.

3. Skill retrieval
   - Retrieve candidates from local skill libraries.
   - Merge and deduplicate candidates into a unified pool.
   - Use semantic and lexical signals when embeddings are available.

4. Skill graph construction
   - Preserve declared `depend_on`, `belong_to`, `compose_with`, and `similar_to` edges.
   - Infer extra `similar_to` edges from capability overlap.
   - Build a multi-relational graph over all candidates.

5. Task-aware compilation
   - Score each candidate by task coverage, quality, cost, and latency.
   - Remove low-relevance nodes.
   - Collapse redundant `similar_to` clusters while keeping the best utility path.
   - Prune broad parent skills when more specific children satisfy the task.
   - Reintroduce required dependencies to maintain executability.
   - Trim isolated low-value nodes.

6. Localization and compression
   - Rewrite generic commands and placeholders against the active workspace.
   - Resolve `{cwd}`, `{workspace_root}`, and Python binary selection.
   - Select relevant fragments instead of always passing whole skill files.
   - Preserve authoritative reference guidance when quality-first mode is enabled.

7. Compilation output
   - A compiled subgraph.
   - An execution order.
   - A dropped-skill audit trail.
   - Coverage, redundancy, token-cost, and pass-trace metrics.

## Intended Gains Over Direct Retrieval

DSC targets gains in:

- Safety and executability
  - Dependency closure after pruning prevents broken packages.
- Completeness
  - Task coverage is explicitly measured before finalizing the package.
- Maintainability
  - Redundant overlapping skills are removed, reducing graph sprawl.
- Cost awareness
  - Candidate selection prefers lower token and lower execution cost skills when coverage is similar.
- Token efficiency
  - The compiler removes unrelated, overlapping, and overly broad skills before packaging.

## Current Implementation Scope

The current implementation provides:

- Query optimizer
- In-memory and local-library retrievers
- Optional semantic retrieval cache
- Multi-relational graph builder
- Heuristic compiler with redundancy pruning and dependency repair
- Environment localization
- Fragment-level skill compression
- Unit tests for the core compiler logic

## Experimental Upgrades

The benchmark harness can evaluate DSC on ScienceWorld, ALFWorld, and WebShop with these ablations:

- no query optimization
- no graph pruning
- no localization
- no dependency repair
- no fragment compression
- direct retrieval baseline

## Suggested Paper Framing

- Baseline: direct retrieval plus raw top-k skill use
- Method: Dynamic Skill Compiler
- Main claim: compile-time optimization of skill graphs improves task success and token efficiency
- Key measured outcomes:
  - task success
  - prompt token usage
  - selected skill count
  - graph density after compilation
  - execution latency
