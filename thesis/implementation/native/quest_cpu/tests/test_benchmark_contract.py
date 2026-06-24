#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "bin" / "quest_runner"


class BenchmarkContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not RUNNER.exists():
            raise unittest.SkipTest("bin/quest_runner is not built; run make first")

    def run_cmd(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            list(args),
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_json_output_parses(self) -> None:
        result = self.run_cmd(str(RUNNER), "--algo", "BV", "--qubits", "4", "--json")
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["algo"], "BV")
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["one_qubit_gates"], 8)
        self.assertEqual(payload["two_qubit_gates"], 3)

    def test_invalid_algorithm_fails_before_timing_record(self) -> None:
        result = self.run_cmd(str(RUNNER), "--algo", "NOPE", "--qubits", "4", "--json")
        self.assertNotEqual(result.returncode, 0)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["status"], "failed")
        self.assertEqual(payload["time_s"], 0.0)
        self.assertEqual(payload["energy_source"], "unavailable")

    def test_old_bb_alias_is_rejected(self) -> None:
        result = self.run_cmd(str(RUNNER), "--algo", "BB", "--qubits", "4", "--json")
        self.assertNotEqual(result.returncode, 0)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["status"], "failed")

    def test_invalid_verify_selection_fails(self) -> None:
        result = self.run_cmd(str(RUNNER), "--verify", "NOT_A_SUITE")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Unknown verification selection", result.stdout)

    def test_hs_logical_qubits_map_to_allocated_qubits(self) -> None:
        result = self.run_cmd(str(RUNNER), "--algo", "HS", "--logical-qubits", "3", "--json")
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["input_qubits"], 3)
        self.assertEqual(payload["allocated_qubits"], 6)
        self.assertEqual(payload["one_qubit_gates"], 18)
        self.assertEqual(payload["two_qubit_gates"], 6)


if __name__ == "__main__":
    unittest.main()
