#!/usr/bin/env python3

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class SubmitMatrixTests(unittest.TestCase):
    def test_tp_envs_exist_and_pin_single_tp(self) -> None:
        expected = {
            2: ("TP_LIST=\"${TP_LIST:-2}\"", "BACKEND_READY_TIMEOUT_S=\"${BACKEND_READY_TIMEOUT_S:-1800}\""),
            4: ("TP_LIST=\"${TP_LIST:-4}\"", "BACKEND_READY_TIMEOUT_S=\"${BACKEND_READY_TIMEOUT_S:-1800}\""),
            8: ("TP_LIST=\"${TP_LIST:-8}\"", "BACKEND_READY_TIMEOUT_S=\"${BACKEND_READY_TIMEOUT_S:-3600}\""),
        }

        for tp, markers in expected.items():
            with self.subTest(tp=tp):
                env_path = ROOT / "config" / f"tp{tp}_concurrency.env"
                text = env_path.read_text(encoding="utf-8")
                for marker in markers:
                    self.assertIn(marker, text)

    def test_submit_scripts_cover_tp_and_concurrency_matrix(self) -> None:
        for tp in (2, 4, 8):
            for cc in (1, 2, 4, 8):
                with self.subTest(tp=tp, cc=cc):
                    script = ROOT / "scripts" / f"submit_tp{tp}_cc{cc}.sbatch"
                    text = script.read_text(encoding="utf-8")
                    self.assertIn(f"#SBATCH -J gaia-tp{tp}-cc{cc}", text)
                    self.assertIn(f"config/tp{tp}_concurrency.env", text)
                    self.assertIn(f'export CONCURRENCY_LIST="{cc}"', text)
                    self.assertIn("run_tp4_concurrency.sh", text)

    def test_submit_scripts_are_safe_after_slurm_spools_them(self) -> None:
        for tp in (2, 4, 8):
            for cc in (1, 2, 4, 8):
                with self.subTest(tp=tp, cc=cc):
                    script = ROOT / "scripts" / f"submit_tp{tp}_cc{cc}.sbatch"
                    text = script.read_text(encoding="utf-8")
                    self.assertNotIn("BASH_SOURCE", text)
                    self.assertIn('GAIA_ROOT="${GAIA_ROOT:-', text)
                    self.assertIn(f'export CONFIG_FILE="${{GAIA_ROOT}}/config/tp{tp}_concurrency.env"', text)
                    self.assertIn('exec bash "${GAIA_ROOT}/scripts/run_tp4_concurrency.sh"', text)

    def test_runner_passes_backend_ready_timeout_to_python(self) -> None:
        text = (ROOT / "scripts" / "run_tp4_concurrency.sh").read_text(encoding="utf-8")
        self.assertIn("BACKEND_READY_TIMEOUT_S=\"${BACKEND_READY_TIMEOUT_S:-1800}\"", text)
        self.assertIn("--backend-ready-timeout-s", text)

    def test_vllm_prefix_caching_is_enabled_by_default(self) -> None:
        manage_text = (ROOT / "scripts" / "manage_vllm.sh").read_text(encoding="utf-8")
        self.assertIn('ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"', manage_text)
        self.assertIn("--enable-prefix-caching", manage_text)

        for tp in (2, 4, 8):
            with self.subTest(tp=tp):
                env_text = (ROOT / "config" / f"tp{tp}_concurrency.env").read_text(encoding="utf-8")
                self.assertIn('ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"', env_text)


if __name__ == "__main__":
    unittest.main()
