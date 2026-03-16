# 2026-03-16 SkillNet Baseline Alignment

## Purpose

Align the experiment-facing baseline name with the paper's `+SkillNet` setting.

- `skill_strategy=skillnet` is now the canonical user-facing name.
- `skill_strategy=baseline` is preserved only as a compatibility alias and is normalized to `skillnet` inside `SkillModule`.
- The underlying retrieval path remains the same benchmark-specific local skill library used by the open-source experiment runners:
  - `experiments/src/skills/scienceworld`
  - `experiments/src/skills/alfworld`
  - `experiments/src/skills/webshop`

## Code Alignment

- Explicit `skillnet` strategy added to:
  - [scienceworld_run.py](/Users/taomiao/codes/DynamicSkillCompiler/SkillNet/experiments/scienceworld_run.py)
  - [alfworld_run.py](/Users/taomiao/codes/DynamicSkillCompiler/SkillNet/experiments/alfworld_run.py)
  - [webshop_run.py](/Users/taomiao/codes/DynamicSkillCompiler/SkillNet/experiments/webshop_run.py)
- Strategy normalization and clean baseline path:
  - [skill.py](/Users/taomiao/codes/DynamicSkillCompiler/SkillNet/experiments/src/skill.py)
- Result summarization now reports legacy `baseline` runs as `skillnet`:
  - [summarize_results.py](/Users/taomiao/codes/DynamicSkillCompiler/SkillNet/experiments/summarize_results.py)

## Validation Run

Run:
- [dev_paper_align_skillnet_dev3_20260316_skill_skillnet](/Users/taomiao/codes/DynamicSkillCompiler/SkillNet/experiments/results/scienceworld/o4-mini/dev_paper_align_skillnet_dev3_20260316_skill_skillnet)

Setting:
- benchmark: `ScienceWorld dev`
- tasks: `3`
- model: `o4-mini`
- max_steps: `30`
- max_workers: `1`
- skill_strategy: `skillnet`

Result:
- success: `3/3`
- avg_reward: `44.0`
- avg_steps: `17.33`

Per-task:
- `variation 418`: reward `24`, steps `25`
- `variation 186`: reward `100`, steps `15`
- `variation 326`: reward `8`, steps `12`

## Full Reference SkillNet Baseline

The closest complete formal reference run already in the repo is:
- [dev_formal_ab20_parallel_20260314_baseline_skill_baseline](/Users/taomiao/codes/DynamicSkillCompiler/SkillNet/experiments/results/scienceworld/o4-mini/dev_formal_ab20_parallel_20260314_baseline_skill_baseline)

This run used the same underlying local benchmark-specific SkillNet retrieval path, but predates the explicit `skillnet` label.

Setting:
- benchmark: `ScienceWorld dev`
- tasks: `20`
- model: `o4-mini`
- max_steps: `30`
- max_workers: `4`
- recorded strategy label: `baseline` (now interpreted as `skillnet`)

Result:
- success: `15/20`
- avg_reward: `64.15`
- avg_steps: `17.30`

## Recommendation

For future paper tables and result folders:

- Use `skillnet` as the baseline name.
- Treat old `baseline` experiment folders as legacy SkillNet baseline runs.
- Avoid mixing `skillnet` with `ReAct / Expel / Few-Shot` unless explicitly reproducing the full paper table.
