import tempfile
import unittest
from io import StringIO
from pathlib import Path

from dynamic_skill_compiler.config import DSCConfig, load_config, prompt_for_config, save_config


class ConfigTest(unittest.TestCase):
    def test_save_and_load_openai_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            save_config(
                DSCConfig(
                    semantic_optimization=True,
                    openai_api_key="sk-test",
                    openai_base_url="https://example.test/v1",
                ),
                config_path,
            )

            loaded = load_config(config_path)

            self.assertTrue(loaded.semantic_optimization)
            self.assertEqual(loaded.openai_api_key, "sk-test")
            self.assertEqual(loaded.openai_base_url, "https://example.test/v1")

    def test_prompt_can_skip_semantic_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"

            config = prompt_for_config(
                input_stream=StringIO("n\n"),
                output_stream=StringIO(),
                path=config_path,
            )

            self.assertFalse(config.semantic_optimization)
            self.assertFalse(load_config(config_path).semantic_optimization)

    def test_prompt_saves_semantic_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"

            config = prompt_for_config(
                input_stream=StringIO("y\nsk-test\nhttps://example.test/v1\n"),
                output_stream=StringIO(),
                path=config_path,
            )

            self.assertTrue(config.semantic_optimization)
            self.assertEqual(config.openai_api_key, "sk-test")
            self.assertEqual(load_config(config_path).openai_base_url, "https://example.test/v1")


if __name__ == "__main__":
    unittest.main()
