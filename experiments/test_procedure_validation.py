"""Test procedure validation and repair."""

import sys
sys.path.insert(0, "src")

from skill import SkillModule

# Initialize skill module
skill_module = SkillModule(
    skills_dir="src/skills/alfworld",
    model="o4-mini",
    selection_strategy="dsc",
)

# Test case 1: Heating task with missing heat action
print("=" * 80)
print("TEST 1: Heating task with missing heat action")
print("=" * 80)

task = "put a hot cup in cabinet"
procedure = """
Phase 1: Locate cup
Phase 2: Pick up cup
Phase 3: Place cup in cabinet
"""

# Simulate a bad procedure code that skips heating
bad_code = """
def overall_procedure_code(env, messages, max_steps):
    # Find cup
    action = "go to coffeemachine 1"
    observation, reward, done = env.step(action)

    # Take cup
    action = "take cup 1 from coffeemachine 1"
    observation, reward, done = env.step(action)

    # Go to cabinet
    action = "go to cabinet 1"
    observation, reward, done = env.step(action)

    # Open cabinet
    action = "open cabinet 1"
    observation, reward, done = env.step(action)

    # Put cup in cabinet (WITHOUT HEATING!)
    action = "put cup 1 in cabinet 1"
    observation, reward, done = env.step(action)

    return messages, done, reward, 5
"""

# Validate and repair
validated_code = skill_module._validate_and_repair_procedure_code(
    task, procedure, bad_code
)

print("\n" + "=" * 80)
print("ORIGINAL CODE (missing heating):")
print("=" * 80)
print(bad_code)

print("\n" + "=" * 80)
print("VALIDATED/REPAIRED CODE:")
print("=" * 80)
print(validated_code)

# Check if repair includes heating
has_heat = any(keyword in validated_code.lower() for keyword in ["heat", "microwave", "stoveburner"])
print("\n" + "=" * 80)
print(f"RESULT: {'PASS - includes heating' if has_heat else 'FAIL - still missing heating'}")
print("=" * 80)

# Test case 2: Cooling task
print("\n\n" + "=" * 80)
print("TEST 2: Cooling task with missing cool action")
print("=" * 80)

task2 = "put a cool lettuce in countertop"
bad_code2 = """
def overall_procedure_code(env, messages, max_steps):
    action = "take lettuce 1 from fridge 1"
    observation, reward, done = env.step(action)
    action = "put lettuce 1 on countertop 1"
    observation, reward, done = env.step(action)
    return messages, done, reward, 2
"""

validated_code2 = skill_module._validate_and_repair_procedure_code(
    task2, "Find and cool lettuce", bad_code2
)

has_cool = any(keyword in validated_code2.lower() for keyword in ["cool", "fridge"])
print(f"\nRESULT: {'PASS - includes cooling' if has_cool else 'FAIL - still missing cooling'}")

print("\n\nAll tests completed!")
