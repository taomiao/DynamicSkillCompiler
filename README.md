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

Install from a local checkout:

```bash
python -m pip install -e .
```

After the package is published to PyPI, users can install it with:

```bash
python -m pip install dynamic-skill-compiler
```

On the first interactive `dsc` run, DSC asks whether to configure OpenAI
embeddings for semantic optimization. You can skip this and use local lexical
optimization only, or enter an API key/base URL to enable semantic matching.

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

## CLI

The package installs a `dsc` command for quick local compilation:

```bash
dsc "Measure the temperature of the unknown object." \
  --skills-dir experiments/src/skills/scienceworld \
  --pretty
```

The command prints a JSON summary containing selected skills, execution order,
coverage metrics, compiler pass traces, and dropped-skill reasons.

Configure semantic optimization explicitly:

```bash
dsc --configure
```

You can also use environment variables or CLI flags:

```bash
OPENAI_API_KEY=sk-... OPENAI_BASE_URL=https://api.example.com/v1 \
  dsc "Measure temperature" --skills-dir path/to/skills --semantic on
```

Use `--semantic off` to force local lexical optimization without any prompt.

## Build And Publish

Build source and wheel distributions:

```bash
python -m pip install ".[dev]"
python -m build
python -m twine check dist/*
```

Publish to TestPyPI first:

```bash
python -m twine upload --repository testpypi dist/*
```

Publish to PyPI:

```bash
python -m twine upload dist/*
```

Use a PyPI API token via `TWINE_USERNAME=__token__` and
`TWINE_PASSWORD=pypi-...` or through your local `.pypirc`.
Package metadata author is `taomiao`.

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -p "test_*.py"

PYTHONPATH=src:experiments/src .venv-experiments/bin/python \
  experiments/test_runtime_execution_guard.py
```

## Branches

- `main`: standalone DSC codebase.
- `codex/v0321`: latest DSC development branch mirrored into `main`.
