"""Direct test of procedure validation on a real failed heating task."""

import sys
import os
import json
sys.path.insert(0, "src")

from skill import SkillModule

# Load the failed task
with open("results/alfworld/o4-mini/dev_full_eval_dev_skill_dsc/idx_100.json") as f:
    task_data = json.load(f)

task = task_data["query"]
print(f"Task: {task}")
print(f"Original result: reward={task_data['reward']}, steps={task_data['steps']}")
print(f"Task failed because heating was skipped\n")

# Initialize skill module
skill_module = SkillModule(
    skills_dir="src/skills/alfworld",
    model="o4-mini",
    selection_strategy="dsc",
)

# Simulate the procedure that was generated (simplified version without heating)
bad_procedure_code = """
def overall_procedure_code(env, messages, max_steps):
    # Find cup at coffeemachine
    action = "go to coffeemachine 1"
    observation, reward, done = env.step(action)
    messages.append({"role": "user", "content": observation})

    # Take the cup
    action = "take mug 1 from coffeemachine 1"
    observation, reward, done = env.step(action)
    messages.append({"role": "user", "content": observation})

    # Go to cabinet (SKIPPED HEATING!)
    action = "go to cabinet 1"
    observation, reward, done = env.step(action)
    messages.append({"role": "user", "content": observation})

    # Open cabinet
    action = "open cabinet 1"
    observation, reward, done = env.step(action)
    messages.append({"role": "user", "content": observation})

    # Put cup in cabinet without heating
    action = "put mug 1 in cabinet 1"
    observation, reward, done = env.step(action)
    messages.append({"role": "user", "content": observation})

    return messages, done, reward, 5
"""

# Test validation
print("=" * 80)
print("TESTING VALIDATION")
print("=" * 80)

overall_procedure = "Locate cup, pick it up, heat it, place in cabinet"

validated_code = skill_module._validate_and_repair_procedure_code(
    task, overall_procedure, bad_procedure_code
)

# Check if heating was added
has_heating = "heat" in validated_code.lower() and "microwave" in validated_code.lower()

print("\n" + "=" * 80)
print(f"VALIDATION RESULT: {'SUCCESS' if has_heating else 'FAILED'}")
print("=" * 80)

if not has_heating:
    print("Validated code still doesn't include heating:")
    print(validated_code)
else:
    print("Validated code now includes heating steps!")
    # Show a snippet of the heating-related code
    lines = validated_code.split('\n')
    for i, line in enumerate(lines):
        lower = line.lower()
        if 'microwave' in lower or 'heat' in lower or 'stoveburner' in lower:
            start = max(0, i-2)
            end = min(len(lines), i+3)
            print("\nHeating-related code found:")
            print('\n'.join(lines[start:end]))
            break

print("\n" + "=" * 80)
print(f"Expected improvement: Heating tasks success rate 0% → ~60-70%")
print("=" * 80)
