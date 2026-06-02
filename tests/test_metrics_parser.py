#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_gaia_concurrency import parse_metric_text  # noqa: E402


class MetricsParserTests(unittest.TestCase):
    def test_labeled_prometheus_metrics_are_aggregated(self) -> None:
        metrics = parse_metric_text(
            """
            # HELP vllm:request_success_total Count of successfully processed requests.
            vllm:request_success_total{engine="0",finished_reason="stop"} 10.0
            vllm:request_success_total{engine="0",finished_reason="length"} 2.0
            vllm:num_requests_running{engine="0"} 1.0
            """
        )

        self.assertEqual(metrics["vllm:request_success_total"], 12.0)
        self.assertEqual(metrics["vllm:num_requests_running"], 1.0)


if __name__ == "__main__":
    unittest.main()
