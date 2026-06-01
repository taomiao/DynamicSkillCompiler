# Dynamic Skill Compiler

Dynamic Skill Compiler (DSC) is a task-driven compiler for agent skills. It takes a natural-language task, retrieves candidate skills from a local skill library, builds a skill graph, and emits a compact, dependency-aware skill package for execution.

## What DSC Does

- Decomposes a task into subgoals and required capabilities.
- Retrieves local skill assets and their declared relationships.
- Extracts relevant skill fragments instead of passing whole skill files.
- Builds a graph over `similar_to`, `belong_to`, `compose_with`, and `depend_on` relationships.
- Runs compiler passes to select, repair, augment, and prune the skill set.
- Produces a `CompiledSkillPackage` with selected skills, execution order, coverage metrics, and pass traces.

## Repository Layout

```text
src/dynamic_skill_compiler/   Core DSC compiler package
tests/                        Compiler and experiment integration tests
experiments/                  Benchmark runners and local skill libraries
reports/                      DSC design notes
```

The benchmark runners under `experiments/` are kept as evaluation harnesses for ScienceWorld, ALFWorld, and WebShop. They are intentionally separate from the core compiler package.

## Quick Start

```bash
python -m pip install -e .
```

```python
from dynamic_skill_compiler import (
    CompilerConfig,
    DynamicSkillCompiler,
    LocalEnvironment,
    LocalSkillLibraryRetriever,
)

retriever = LocalSkillLibraryRetriever("experiments/src/skills/scienceworld")
compiler = DynamicSkillCompiler(
    retriever=retriever,
    config=CompilerConfig(),
)

compiled = compiler.compile(
    "Your task is to measure the temperature of an unknown substance.",
    environment=LocalEnvironment(benchmark="scienceworld"),
)

print([skill.asset.name for skill in compiled.compiled_skills])
print(compiled.execution_order)
print(compiled.metrics.coverage_score)
```

## Tests

```bash
PYTHONPATH=src .venv-experiments/bin/python -m unittest \
  tests/test_dynamic_skill_compiler.py \
  tests/test_experiment_skill_module.py

PYTHONPATH=src:experiments/src .venv-experiments/bin/python \
  experiments/test_runtime_execution_guard.py
```

## Branches

- `main`: standalone DSC codebase.
- `codex/v0321`: latest DSC development branch mirrored into `main`.
