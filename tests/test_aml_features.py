"""Toy-graph checks for AML features, score contexts and the two-stage prediction writer."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from feature_workspace.aml_features import _asset_parts, compute_features
from feature_workspace.prepare_transactions import COLUMNS
from feature_workspace.score_context import ScoreContext
from scripts.aml_model import scenario_groups, write_predictions


def payments(records, asset="XRP"):
    """records: (hash, timestamp, sender, receiver, usd)."""
    rows = [{"TRANSACTION_HASH": h, "TIMESTAMP": t, "SENDER": s, "RECEIVER": r,
             "SOURCE_ASSET": asset, "SOURCE_AMOUNT": usd, "DESTINATION_ASSET": asset,
             "DESTINATION_AMOUNT": usd, "SOURCE_AMOUNT_USD": usd, "DESTINATION_AMOUNT_USD": usd,
             "TRANSFER_COUNT": 1, "IS_CONVERSION": False, "CONVERSION_STEP_COUNT": 0,
             "INTERMEDIARY_COUNT": 0, "FEE_XRP": .00001} for h, t, s, r, usd in records]
    return pd.DataFrame(rows).loc[:, COLUMNS]


BACKGROUND = [("bg1", 0, "X", "Y", 5.), ("bg2", 500_000, "Y", "X", 5.)]


class AmlFeatureTests(unittest.TestCase):
    def test_time_respecting_cycle_and_chain(self):
        cycle = payments(BACKGROUND + [("ab", 1000, "A", "B", 100.), ("bc", 1060, "B", "C", 98.),
                                       ("ca", 1120, "C", "A", 97.)])
        f = compute_features(cycle)
        self.assertEqual(f.loc["ab", "cycle_hops"], 3)
        self.assertAlmostEqual(f.loc["ab", "cycle_hours"], 120 / 3600)
        self.assertEqual(f.loc["ab", "chain_fwd_len_1h"], 3)
        self.assertEqual(f.loc["ca", "chain_bwd_len_1h"], 3)
        # B->C before A->B is not a forwarded A->B->C flow, and no cycle closes.
        broken = payments(BACKGROUND + [("bc", 900, "B", "C", 98.), ("ab", 1000, "A", "B", 100.),
                                        ("ca", 1120, "C", "A", 97.)])
        g = compute_features(broken)
        self.assertTrue(np.isnan(g.loc["ab", "cycle_hops"]))
        self.assertEqual(g.loc["ab", "chain_fwd_len_1h"], 1)
        self.assertEqual(g.loc["ab", "r_out_after_1h"], 0)

    def test_amount_inconsistent_hop_breaks_chain(self):
        f = compute_features(payments(BACKGROUND + [("ab", 1000, "A", "B", 100.),
                                                    ("bc", 1060, "B", "C", 10.)]))
        self.assertEqual(f.loc["ab", "chain_fwd_len_1h"], 1)
        self.assertEqual(f.loc["ab", "r_out_after_1h"], 1)

    def test_repeated_pairs_stay_distinct_and_fan_counts(self):
        fan = payments(BACKGROUND + [(f"o{i}", 1000 + 30 * i, "S", f"R{i % 3}", 50.) for i in range(6)])
        f = compute_features(fan)
        self.assertEqual(f.loc["o0", "s_out_win_10m"], 6)
        self.assertEqual(f.loc["o0", "s_out_uniq_win_10m"], 3)
        self.assertEqual(f.loc["o0", "pair_n"], 2)
        self.assertEqual(f.loc["o3", "pair_rank"], 2)

    def test_burst_clusters_follow_linkage_scale(self):
        f = compute_features(payments(BACKGROUND + [("p", 1000, "A", "B", 10.), ("q", 1060, "B", "C", 10.),
                                                    ("z", 1260, "C", "D", 10.)]))
        self.assertEqual(f.loc["p", "burst_2m_size"], 2)    # z is 200 s after q
        self.assertEqual(f.loc["p", "burst_5m_size"], 3)
        self.assertEqual(f.loc["q", "burst_5m_r_events"], 1)

    def test_input_order_and_extra_columns_do_not_change_features(self):
        base = payments(BACKGROUND + [("ab", 1000, "A", "B", 100.), ("bc", 1060, "B", "C", 98.),
                                      ("ca", 1120, "C", "A", 97.), ("tie", 1000, "A", "C", 3.)])
        expected = compute_features(base)
        shuffled = base.sample(frac=1, random_state=3).assign(is_suspicious=1, scenario_id="leak")
        got = compute_features(shuffled)
        self.assertEqual(list(got.index), list(shuffled.TRANSACTION_HASH))
        self.assertNotIn("is_suspicious", got.columns)
        pd.testing.assert_frame_equal(got.loc[expected.index], expected)

    def test_issuer_roles_are_comparable_across_splits(self):
        a = pd.Series(["XRP", "iss1:GBP", "iss1:USD", "iss2:EUR", "iss2:USD"])
        b = pd.Series(["XRP", "zzz:GBP", "zzz:USD", "aaa:EUR", "aaa:USD"])
        self.assertEqual(_asset_parts(a)[1].tolist(), _asset_parts(b)[1].tolist())
        self.assertEqual(_asset_parts(a)[1].tolist(), [0, 1, 1, 2, 2])


class ScoreContextTests(unittest.TestCase):
    def test_aggregates_exclude_self_and_check_coverage(self):
        tx = payments(BACKGROUND + [("ab", 1000, "A", "B", 100.), ("bc", 1060, "B", "C", 98.)])
        ctx = ScoreContext(tx)
        scores = pd.Series({"bg1": .1, "bg2": .2, "ab": .9, "bc": .3})
        c = ctx.aggregate(scores)
        self.assertEqual(list(c.index), list(tx.TRANSACTION_HASH))
        self.assertAlmostEqual(c.loc["ab", "ctx_r_1h_max"], .3)     # only bc, not itself
        self.assertAlmostEqual(c.loc["bc", "ctx_chain_6h_max"], .9)
        self.assertEqual(c.loc["bg1", "ctx_s_1h_max"], 0)
        with self.assertRaises(ValueError):
            ctx.aggregate(scores.drop("bc"))


class ModelHelperTests(unittest.TestCase):
    def test_groups_and_prediction_round_trip(self):
        ann = pd.DataFrame({"scenario_id": ["s1", None, "s1"]}, index=["a", "b", "c"])
        self.assertEqual(scenario_groups(ann).tolist(), ["s1", "tx:b", "s1"])
        index = pd.Index(["h2", "h1"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.csv"
            write_predictions(path, index, np.array([0.25, 1.0]), index)
            reread = pd.read_csv(path)
            self.assertEqual(list(reread.columns), ["TRANSACTION_HASH", "score"])
            self.assertEqual(reread.TRANSACTION_HASH.tolist(), ["h2", "h1"])
            with self.assertRaises(ValueError):
                write_predictions(path, index, np.array([0.5, 1.5]), index)
            with self.assertRaises(ValueError):
                write_predictions(path, index, np.array([0.5, 0.5]), pd.Index(["h2", "h3"]))


if __name__ == "__main__":
    unittest.main()
