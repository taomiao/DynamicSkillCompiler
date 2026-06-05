import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from dynamic_skill_compiler.cli import main


class CliTest(unittest.TestCase):
    def test_cli_compiles_local_skill_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            skills_dir = Path(tmpdir) / "skills"
            skill_dir = skills_dir / "temperature-measurer"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                """---
description: Measure object temperature.
---

# Temperature Measurer

- Find the thermometer.
- Measure the target object's temperature.
""",
                encoding="utf-8",
            )

            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = main(
                    [
                        "Measure the temperature of the unknown object.",
                        "--skills-dir",
                        str(skills_dir),
                        "--semantic",
                        "off",
                    ]
                )

            self.assertEqual(exit_code, 0)
            summary = json.loads(output.getvalue())
            self.assertEqual(summary["query"], "Measure the temperature of the unknown object.")
            self.assertIn("temperature-measurer", summary["selected_skills"])


if __name__ == "__main__":
    unittest.main()
