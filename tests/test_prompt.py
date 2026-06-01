#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_gaia_manual_question import build_prompt  # noqa: E402


class PromptTests(unittest.TestCase):
    def test_prompt_hides_reasoning_but_keeps_final_answer_contract(self) -> None:
        prompt = build_prompt("What is 2 + 2?", None, "/usr/bin/python")

        self.assertIn("Do not reveal your reasoning process", prompt)
        self.assertIn("FINAL ANSWER: [YOUR FINAL ANSWER]", prompt)
        self.assertNotIn("Report your thoughts", prompt)


if __name__ == "__main__":
    unittest.main()
