# Evaluation Experiments

This section provides instructions for setting up the evaluation environments.

## 📂 Expected Directory Structure

To ensure the scripts can locate the environments, please organize your files as follows:

```text
SkillNet/
├── experiments/
│   ├── alfworld/          # git clone here
│   ├── ScienceWorld/      # git clone here
│   ├── WebShop/           # git clone here (or rename to webshop)
│   ├── src/
│   ├── requirements.txt
│   ├── alfworld_run.py
│   ├── scienceworld_run.py
│   └── webshop_run.py
```

## 🚀 Quick Start

We suggest configuring separate conda environments for these three datasets to avoid dependency conflicts.

### ALFWorld
1. **Clone & Setup:**
  ```bash
  cd experiments
  git clone https://github.com/alfworld/alfworld.git
  cd alfworld
  # Follow the official installation steps from the repo (https://github.com/alfworld/alfworld)
  ```
2. **Environment Variable:**
- Set `ALFWORLD_DATA` to the dataset root or edit `src/alfworld/base_config.yaml` to point to your local paths:

  ```bash
  export ALFWORLD_DATA=/path/to/alfworld_data
  ```

### ScienceWorld
1. **Clone & Setup:**
  ```bash
  cd experiments
  git clone https://github.com/allenai/ScienceWorld.git
  cd ScienceWorld
  # Refer to the ScienceWorld repository for environment setup (https://github.com/allenai/ScienceWorld)
  ```

### WebShop
1. **Clone & Setup:**
  ```bash
  cd experiments
  git clone https://github.com/princeton-nlp/WebShop.git
  cd WebShop
  # Refer to the WebShop repository for environment setup (https://github.com/princeton-nlp/WebShop)
  ```

  The evaluation script now accepts either `experiments/WebShop` or `experiments/webshop`.

---

For each environment, install common dependencies:
```bash
cd experiments
pip install -r requirements.txt
```

### Running
#### Step 1: Initialize Environment Variables
Before running the scripts, configure your API credentials:
```bash
export API_KEY=YOUR_API_KEY
export BASE_URL=YOUR_API_BASE_URL
```

#### Step 2: Execution
Run the corresponding evaluation script from the `experiments/` directory.
```python
cd experiments

# ALFWorld
python alfworld_run.py --model o4-mini --split dev --max_workers 10 --exp_name alf_test --use_skill --skill_strategy baseline
python alfworld_run.py --model o4-mini --split dev --max_workers 10 --exp_name alf_test_dsc --use_skill --skill_strategy dsc

# ScienceWorld
python scienceworld_run.py --model o4-mini --split test --max_workers 5 --exp_name sci_test --use_skill --skill_strategy baseline
python scienceworld_run.py --model o4-mini --split test --max_workers 5 --exp_name sci_test_dsc --use_skill --skill_strategy dsc --compiler_min_relevance 0.15 --compiler_preserve_top_k 3

# WebShop
python webshop_run.py --model o4-mini --max_workers 3 --exp_name web_test --use_skill --skill_strategy baseline
python webshop_run.py --model o4-mini --max_workers 3 --exp_name web_test_dsc --use_skill --skill_strategy dsc
```

#### 🛠️ Argument Descriptions
- `--model`: The name of the LLM to evaluate.

- `--split`: Data split to use (`dev` or `test`).

- `--max_workers`: Number of parallel workers for evaluation.

- `exp_name`: results save name.

- `--use_skill`: Enable the skill-augmented module.

- `--skill_strategy`: `baseline` uses the original skill retrieval path, `dsc` enables Dynamic Skill Compiler.

- `--compiler_min_relevance`: pruning threshold for DSC.

- `--compiler_preserve_top_k`: always keep at least the top-k scored skills.

- `--compiler_similar_prune_margin`: only prune `similar_to` alternatives when the score gap exceeds this margin.

- `--compiler_keep_parent_if_better_by`: only drop parent/container skills when the child is better by this margin.

- `--compiler_coverage_weight` / `--compiler_quality_weight` / `--compiler_cost_weight` / `--compiler_latency_weight`: scoring weights for DSC.

### Step 3: Summarize Results

After runs finish, aggregate them into JSON/Markdown:

```bash
python summarize_results.py \
  --results-root results/scienceworld/o4-mini \
  --json-out results/scienceworld/o4-mini/summary.json \
  --md-out results/scienceworld/o4-mini/summary.md
```
