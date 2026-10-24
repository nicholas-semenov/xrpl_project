"""Toy-graph checks for AML features, score contexts and the two-stage prediction writer."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from feature_workspace.aml_features import _asset_parts, compute_features
from feature_workspace.prepare_transactions import COLUMNS
from feature_workspace.score_context import ScoreContext
from scripts.aml_model import extra_features, scenario_groups, write_predictions


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

    def test_local_structure_dense_motif_versus_star(self):
        # Bipartite block S1,S2 -> R1,R2 within minutes, plus a star edge from S1 to an outsider.
        dense = payments(BACKGROUND + [("a", 1000, "S1", "R1", 50.), ("b", 1060, "S1", "R2", 50.),
                                       ("c", 1120, "S2", "R1", 50.), ("d", 1180, "S2", "R2", 50.),
                                       ("star", 1200, "S1", "OUT", 50.)])
        f = compute_features(dense)
        self.assertEqual(f.loc["a", "common_neighbours_1h"], 0)   # R1 and S1 share no third wallet
        self.assertEqual(f.loc["a", "ego_edges_1h"], 5)           # S1,R1 + R2,S2,OUT: all 5 edges
        self.assertEqual(f.loc["star", "ego_edges_1h"], 3)        # S1's edges to R1,R2,OUT only
        self.assertGreater(f.loc["a", "ego_density_1h"], f.loc["star", "ego_density_1h"])
        triangle = payments(BACKGROUND + [("xy", 1000, "X1", "Y1", 10.), ("yz", 1060, "Y1", "Z1", 10.),
                                          ("xz", 1120, "X1", "Z1", 10.)])
        self.assertEqual(compute_features(triangle).loc["xy", "common_neighbours_1h"], 1)

    def test_flow_clusters_link_only_consistent_forwards(self):
        f = compute_features(payments(BACKGROUND + [("h1", 1000, "A", "B", 100.), ("h2", 2500, "B", "C", 95.),
                                                    ("h3", 4000, "C", "D", 90.), ("off", 4100, "C", "E", 5.)]))
        self.assertEqual(f.loc["h1", "flow_30m_size"], 3)          # gaps of 25 min each
        self.assertEqual(f.loc["off", "flow_30m_size"], 1)         # value 5 is not a consistent forward
        self.assertAlmostEqual(f.loc["h1", "flow_30m_value_retained"], 0.9)

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
    def test_extra_feature_modules_are_prefixed_and_aligned(self):
        tx = payments(BACKGROUND + [("ab", 1000, "A", "B", 100.)])
        with tempfile.TemporaryDirectory() as tmp:
            module = Path(tmp) / "addon.py"
            module.write_text("import numpy as np\n"
                              "def compute(tx):\n"
                              "    out = tx.set_index('TRANSACTION_HASH')[['DESTINATION_AMOUNT_USD']]\n"
                              "    return np.log(out.rename(columns={'DESTINATION_AMOUNT_USD': 'lv'})).iloc[::-1]\n")
            (frame,) = extra_features(None, tx, [module])
        self.assertEqual(list(frame.columns), ["addon__lv"])
        self.assertEqual(list(frame.index), list(tx.TRANSACTION_HASH))   # realigned to input order
        self.assertAlmostEqual(frame.loc["ab", "addon__lv"], np.log(100.))

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
