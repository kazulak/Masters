"""Synthetic tests only: no repository, SDK, hardware, or physical observations."""
from copy import deepcopy
import unittest

from p6_readout import summarize


def fixture():
    manifest = {"stage": "evaluation", "measurement_blocks": [1, 2, 3, 4, 5], "cells": {}}
    rows = []
    for family in ("A", "B"):
        for topology in ("1dpu_t8", "4dpu_t8"):
            cell_id = f"{family}/{topology}"
            manifest["cells"][cell_id] = {
                "family": family, "topology_id": topology, "split": "test",
                "candidates": {p: {} for p in ("g", "r", "u")},
                "selection": {"roles": {"G": "g", "F": "g", "R": "r", "U": "u"}},
            }
            for p, inclusive in (("g", 12.0), ("r", 6.0), ("u", 3.0)):
                for b in (0, 1, 2, 3, 4, 5):
                    t = inclusive if b else 99999.0
                    rows.append({"cell_id": cell_id, "path_id": p, "block": b,
                                 "round_id": "evaluation", "split": "test", "status": "success",
                                 "validation": "pass", "fallback": False,
                                 "attempt_type": "measurement" if b else "warmup",
                                 "session_open_s": 0.5, "total_wall_s": t - 1,
                                 "session_close_s": 0.5, "session_inclusive_s": t})
    return manifest, rows


class ReadoutTests(unittest.TestCase):
    def test_exact_contrast_alias_and_warmup_exclusion(self):
        m, rows = fixture()
        r = summarize(m, rows)
        primary = next(x for x in r["aggregates"] if x["group"] == "all"
                       and x["contrast"] == "R/U" and x["metric"] == "session_inclusive_s")
        self.assertAlmostEqual(primary["family_balanced_geometric_ratio"], 2)
        self.assertAlmostEqual(primary["paired_bootstrap_low"], 2)
        self.assertAlmostEqual(primary["paired_bootstrap_high"], 2)
        aliases = [x for x in r["contrasts"] if x["contrast"] == "G/F"]
        self.assertTrue(all(x["same_selected_path"] and x["ratio_of_medians"] == 1 for x in aliases))
        self.assertEqual(len(r["observations"]), 4 * 4 * 5)
        self.assertTrue(all(x["session_inclusive_s"] < 100 for x in r["observations"]))

    def test_family_weights_not_cell_frequency(self):
        m, rows = fixture()
        # Family A gains 4x in both its cells; B regresses to .25x in one cell.
        remove = "B/4dpu_t8"
        del m["cells"][remove]
        rows = [r for r in rows if r["cell_id"] != remove]
        for r in rows:
            if r["path_id"] == "r" and r["block"]:
                t = 12 if r["cell_id"].startswith("A/") else 0.75
                r.update(session_open_s=0, session_close_s=0, total_wall_s=t, session_inclusive_s=t)
        out = summarize(m, rows)
        primary = next(x for x in out["aggregates"] if x["group"] == "all"
                       and x["contrast"] == "R/U" and x["metric"] == "session_inclusive_s")
        self.assertAlmostEqual(primary["family_balanced_geometric_ratio"], 1)

    def test_missing_or_duplicate_measurement_rejected(self):
        m, rows = fixture()
        measured = next(r for r in rows if r["attempt_type"] == "measurement")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            summarize(m, [*rows, deepcopy(measured)])
        rows.remove(measured)
        with self.assertRaisesRegex(ValueError, "measurement set"):
            summarize(m, rows)

    def test_wrong_session_boundary_rejected(self):
        m, rows = fixture()
        next(r for r in rows if r["block"] == 1)["session_inclusive_s"] += 1
        with self.assertRaisesRegex(ValueError, "boundary"):
            summarize(m, rows)

    def test_repeatability(self):
        m, rows = fixture()
        self.assertEqual(summarize(m, rows), summarize(m, rows))


if __name__ == "__main__":
    unittest.main()
