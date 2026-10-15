"""Question-level statistical contracts, with hand-computed contrasts."""
import importlib
import json
import math
import pathlib
import subprocess
import sys
import unittest


def example_rows():
    return [
        dict(question_id="q1", colocated_nocomm_f1=.5, colocated_comm_f1=.5,
             distributed_nocomm_f1=0., distributed_comm_f1=1.),
        dict(question_id="q2", colocated_nocomm_f1=1., colocated_comm_f1=0.,
             distributed_nocomm_f1=.5, distributed_comm_f1=.5),
        dict(question_id="q3", colocated_nocomm_f1=0., colocated_comm_f1=1.,
             distributed_nocomm_f1=1., distributed_comm_f1=0.),
    ]


class StatisticsTests(unittest.TestCase):
    def setUp(self):
        self.stats = importlib.import_module("WEBExtension.causal_study.statistics")

    def test_interaction_is_paired_before_averaging_and_counts_question_changes(self):
        result = self.stats.analyze_e6(example_rows(), bootstrap_resamples=500, seed=9)
        self.assertEqual(result["n_questions"], 3)
        self.assertAlmostEqual(result["cell_means"]["distributed_comm_f1"], .5)
        contrast = result["contrasts"]["interaction"]
        self.assertAlmostEqual(contrast["mean"], 0.)
        self.assertEqual(contrast["help_harm_unchanged"], {"helped": 2, "harmed": 1, "unchanged": 0})
        self.assertEqual([row["interaction"] for row in result["per_question"]], [1., 1., -2.])
        self.assertEqual(result["contrasts"]["colocated_comm_effect"]["help_harm_unchanged"],
                         {"helped": 1, "harmed": 1, "unchanged": 1})

    def test_exact_sign_flip_two_sided_has_known_probability_and_assumption(self):
        rows = [dict(question_id=str(i), colocated_nocomm_f1=0., colocated_comm_f1=0.,
                     distributed_nocomm_f1=0., distributed_comm_f1=1.) for i in range(4)]
        result = self.stats.analyze_e6(rows, bootstrap_resamples=100, seed=11)
        test = result["contrasts"]["interaction"]["sign_flip_test"]
        self.assertEqual(test["method"], "exact_question_sign_flip")
        self.assertEqual(test["resamples"], 16)
        self.assertAlmostEqual(test["p_value"], .125)
        self.assertIn("symmetric", test["assumption"])
        self.assertEqual(result["contrasts"]["interaction"]["ci95"], [1., 1.])

    def test_duplicate_ids_fail_instead_of_pseudoreplication(self):
        rows = example_rows()
        rows.append(dict(rows[0], seed=99))
        with self.assertRaisesRegex(ValueError, "duplicate question_id"):
            self.stats.analyze_e6(rows)

    def test_invalid_or_missing_scores_fail_instead_of_scoring_failures_as_zero(self):
        for value in [float("nan"), float("inf"), -.01, 1.01, None, "0.5", True]:
            rows = example_rows()
            rows[0]["distributed_comm_f1"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.stats.analyze_e6(rows)
        rows = example_rows()
        del rows[0]["distributed_comm_f1"]
        with self.assertRaises(ValueError):
            self.stats.analyze_e6(rows)

    def test_no_rows_and_single_question_are_insufficient_not_conclusive(self):
        result = self.stats.analyze_e6([])
        self.assertEqual(result["status"], "INSUFFICIENT_DATA")
        self.assertIsNone(result["contrasts"]["interaction"]["ci95"])
        result = self.stats.analyze_e6(example_rows()[:1])
        self.assertEqual(result["status"], "INSUFFICIENT_DATA")
        self.assertIsNone(result["contrasts"]["interaction"]["ci95"])
        json.dumps(result, allow_nan=False)

    def test_analysis_is_reproducible_order_invariant_and_json_serializable(self):
        first = self.stats.analyze_e6(example_rows(), bootstrap_resamples=300, seed=80)
        second = self.stats.analyze_e6(list(reversed(example_rows())), bootstrap_resamples=300, seed=80)
        self.assertEqual(first, second)
        json.dumps(first, allow_nan=False)

    def test_zero_effect_sign_flip_is_one_and_does_not_claim_equivalence(self):
        rows = [dict(question_id=str(i), colocated_nocomm_f1=.5, colocated_comm_f1=.5,
                     distributed_nocomm_f1=.5, distributed_comm_f1=.5) for i in range(8)]
        result = self.stats.analyze_e6(rows, bootstrap_resamples=200)
        self.assertEqual(result["contrasts"]["interaction"]["sign_flip_test"]["p_value"], 1.)
        self.assertTrue(any("equivalence" in warning for warning in result["warnings"]))

    def test_power_uses_positive_variance_floor_and_reports_mc_uncertainty(self):
        rows = [dict(question_id=str(i), colocated_nocomm_f1=.5, colocated_comm_f1=.5,
                     distributed_nocomm_f1=.5, distributed_comm_f1=.5) for i in range(8)]
        result = self.stats.power_sensitivity(rows, effect_grid=(.05, .1), sample_sizes=(20, 100),
                                              simulations=200, seed=7)
        self.assertEqual(result["observed_interaction_sd"], 0.)
        self.assertGreaterEqual(result["sd_floor"], .25)
        self.assertEqual(result["status"], "SENSITIVITY_ONLY_DEGENERATE_PILOT")
        self.assertEqual(len(result["grid"]), 12)
        for point in result["grid"]:
            self.assertGreater(point["assumed_sd"], 0.)
            self.assertTrue(0 <= point["power_ci95"][0] <= point["estimated_power"] <= point["power_ci95"][1] <= 1)
        self.assertIn("approximation", result["test_method"])
        json.dumps(result, allow_nan=False)

    def test_simulated_power_increases_for_large_effect_and_sample(self):
        result = self.stats.power_sensitivity(example_rows(), effect_grid=(.02, .4),
                                              sample_sizes=(20, 400), simulations=500, seed=18)
        baseline = [r for r in result["grid"] if r["variance_multiplier"] == 1.]
        small = next(r for r in baseline if r["effect"] == .02 and r["n_questions"] == 20)
        large = next(r for r in baseline if r["effect"] == .4 and r["n_questions"] == 400)
        self.assertGreater(large["estimated_power"], small["estimated_power"] + .5)
        self.assertTrue(any(r["target_power"] == .8 for r in result["recommendations"]))
        self.assertTrue(any(r["target_power"] == .9 for r in result["recommendations"]))
        self.assertEqual(result, self.stats.power_sensitivity(example_rows(), effect_grid=(.02, .4),
                         sample_sizes=(20, 400), simulations=500, seed=18))

    def test_power_rejects_invalid_design_parameters(self):
        for kwargs in [dict(effect_grid=(0.,)), dict(sample_sizes=(1,)), dict(simulations=0),
                       dict(variance_multipliers=(0.,)), dict(sd_floor=0.)]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.stats.power_sensitivity(example_rows(), **kwargs)

    def test_direct_script_import_does_not_self_import_stdlib_statistics(self):
        folder = pathlib.Path(self.stats.__file__).parent
        completed = subprocess.run([sys.executable, "-c", "import statistics; print(hasattr(statistics, 'analyze_e6'))"],
                                   cwd=folder, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "True")


if __name__ == "__main__":
    unittest.main()
