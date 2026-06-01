# DSC Benchmark Experiments

This directory contains the benchmark harnesses used to evaluate Dynamic Skill Compiler (DSC) on ScienceWorld, ALFWorld, and WebShop. The harnesses are intentionally kept outside the core package so the compiler can remain a small reusable library under `src/dynamic_skill_compiler`.

## Layout

```text
experiments/
├── alfworld/          # external ALFWorld checkout
├── ScienceWorld/      # external ScienceWorld checkout
├── WebShop/           # external WebShop checkout
├── src/               # shared benchmark skill/evaluation utilities
├── alfworld_run.py
├── scienceworld_run.py
└── webshop_run.py
```

## Setup

Install the common experiment dependencies from this directory:

```bash
cd experiments
pip install -r requirements.txt
```

Each benchmark still needs its official environment and data:

- ALFWorld: clone `https://github.com/alfworld/alfworld.git` into `experiments/alfworld` and set `ALFWORLD_DATA`.
- ScienceWorld: clone `https://github.com/allenai/ScienceWorld.git` into `experiments/ScienceWorld` and use the local Java/OpenJDK setup required by that project.
- WebShop: clone `https://github.com/princeton-nlp/WebShop.git` into `experiments/WebShop` or `experiments/webshop`.

Configure model access before running:

```bash
export API_KEY=YOUR_API_KEY
export BASE_URL=YOUR_API_BASE_URL
```

## Running Baseline vs DSC

Run commands from `experiments/`:

```bash
# ALFWorld
python alfworld_run.py --model o4-mini --split dev --max_workers 10 --exp_name alf_baseline --use_skill --skill_strategy baseline
python alfworld_run.py --model o4-mini --split dev --max_workers 10 --exp_name alf_dsc --use_skill --skill_strategy dsc

# ScienceWorld
python scienceworld_run.py --model o4-mini --split test --max_workers 5 --exp_name sci_baseline --use_skill --skill_strategy baseline
python scienceworld_run.py --model o4-mini --split test --max_workers 5 --exp_name sci_dsc --use_skill --skill_strategy dsc --compiler_min_relevance 0.15 --compiler_preserve_top_k 3

# WebShop
python webshop_run.py --model o4-mini --max_workers 3 --exp_name web_baseline --use_skill --skill_strategy baseline
python webshop_run.py --model o4-mini --max_workers 3 --exp_name web_dsc --use_skill --skill_strategy dsc
```

Useful DSC compiler knobs:

- `--compiler_min_relevance`: pruning threshold.
- `--compiler_preserve_top_k`: always keep at least the top-k scored skills.
- `--compiler_similar_prune_margin`: only prune `similar_to` alternatives when the score gap exceeds this margin.
- `--compiler_keep_parent_if_better_by`: only drop parent/container skills when the child is better by this margin.
- `--compiler_coverage_weight`, `--compiler_quality_weight`, `--compiler_cost_weight`, `--compiler_latency_weight`: scoring weights.

## Summaries

Aggregate finished runs into JSON and Markdown:

```bash
python summarize_results.py \
  --results-root results/scienceworld/o4-mini \
  --json-out results/scienceworld/o4-mini/summary.json \
  --md-out results/scienceworld/o4-mini/summary.md
```

## Skill Evolution Dry Run

DSC can mine failed trajectories and write staged skill-improvement proposals without directly changing the official skill library:

```bash
python evolve_skills.py \
  --result-dir results/webshop/o4-mini/web_dsc \
  --max-cases 10 \
  --min-cases-per-proposal 2 \
  --write-staging \
  --staging-dir results/evolution_staging/webshop_smoke \
  --overwrite
```

The output includes `proposals.json`, `proposals.md`, and one `evo-*/PATCH_PLAN.md` per staged proposal. Promote a proposal only if validation improves task reward without unacceptable token or regression cost.
