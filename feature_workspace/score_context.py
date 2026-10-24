"""Neighbourhood aggregation of first-stage suspicion scores within one split.

The neighbourhoods depend only on that split's unlabeled payments. The scores fed
in must come from models not fitted on the scored rows' own labels (nested
out-of-fold scores during validation; a train-fitted model for test rows), so the
aggregates never carry a held-out target.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from feature_workspace.aml_features import CHAIN_RATIO, _Events, _find
from feature_workspace.prepare_transactions import COLUMNS, require

BURST_SCALES = {"5m": 300, "15m": 900}


class ScoreContext:
    """Precomputed neighbour lists per payment; aggregate() is cheap per score vector."""

    def __init__(self, transactions: pd.DataFrame):
        require(set(COLUMNS) <= set(transactions.columns), "Missing canonical columns")
        require(transactions.TRANSACTION_HASH.is_unique, "Duplicate transaction hashes")
        df = transactions.loc[:, list(COLUMNS)].sort_values(
            ["TIMESTAMP", "TRANSACTION_HASH"], kind="mergesort").reset_index(drop=True)
        self.hashes = pd.Index(df.TRANSACTION_HASH)
        self.input_order = pd.Index(transactions.TRANSACTION_HASH)
        n = len(df)
        wallets = pd.Index(sorted(set(df.SENDER) | set(df.RECEIVER)))
        s = wallets.get_indexer(df.SENDER)
        r = wallets.get_indexer(df.RECEIVER)
        t = df.TIMESTAMP.to_numpy("int64")
        a = df.DESTINATION_AMOUNT_USD.to_numpy(float)
        a_src = df.SOURCE_AMOUNT_USD.to_numpy(float)
        nw = len(wallets)
        out_ev, in_ev = _Events(s, nw), _Events(r, nw)
        both = [np.sort(np.concatenate([out_ev.pos[w], in_ev.pos[w]])) for w in range(nw)]

        def wallet_window(owner: np.ndarray, e: int, W: int) -> np.ndarray:
            p = both[owner[e]]
            tp = t[p]
            sel = p[np.searchsorted(tp, t[e] - W, "left"):np.searchsorted(tp, t[e] + W, "right")]
            return sel[sel != e]

        lists: dict[str, list[np.ndarray]] = {k: [] for k in (
            "s_1h", "r_1h", "s_6h", "r_6h", "downstream_6h", "upstream_6h", "chain_6h",
            "s_out_1h", "s_in_1h", "r_in_1h", "r_out_1h")}
        succ = []
        for e in range(n):
            lists["s_1h"].append(wallet_window(s, e, 3600))
            lists["r_1h"].append(wallet_window(r, e, 3600))
            for name, events, owner in (("s_out_1h", out_ev, s), ("s_in_1h", in_ev, s),
                                        ("r_in_1h", in_ev, r), ("r_out_1h", out_ev, r)):
                near = events.span(owner[e], t, t[e] - 3600, t[e] + 3600)
                lists[name].append(near[near != e])
            lists["s_6h"].append(wallet_window(s, e, 21600))
            lists["r_6h"].append(wallet_window(r, e, 21600))
            lists["downstream_6h"].append(out_ev.span(r[e], t, t[e], t[e] + 21600, lo_open=True))
            lists["upstream_6h"].append(in_ev.span(s[e], t, t[e] - 21600, t[e], hi_open=True))
            cand = lists["downstream_6h"][-1]
            ratio = a_src[cand] / a[e]
            succ.append(cand[(ratio >= CHAIN_RATIO[0]) & (ratio <= CHAIN_RATIO[1])])
        pred = [[] for _ in range(n)]
        for e in range(n):
            for q in succ[e]:
                pred[q].append(e)
        lists["chain_6h"] = [np.concatenate([succ[e], np.asarray(pred[e], dtype="int64")])
                             for e in range(n)]

        # Burst-cluster members (same linkage rule as the tempo features), excluding self.
        links = []
        for w in range(nw):
            p = both[w]
            gaps = np.diff(t[p])
            links.extend(zip(gaps.tolist(), p[:-1].tolist(), p[1:].tolist()))
        links.sort()
        parent = np.arange(n)
        k = 0
        for name, scale in BURST_SCALES.items():
            while k < len(links) and links[k][0] <= scale:
                _, x, y = links[k]
                rx, ry = _find(parent, x), _find(parent, y)
                if rx != ry:
                    parent[ry] = rx
                k += 1
            root = np.array([_find(parent, e) for e in range(n)])
            members = pd.Series(np.arange(n)).groupby(root).apply(np.asarray).to_dict()
            lists[f"burst_{name}"] = [members[root[e]][members[root[e]] != e] for e in range(n)]

        self.csr = {}
        for name, neighbours in lists.items():
            sizes = np.array([len(x) for x in neighbours])
            indptr = np.concatenate([[0], np.cumsum(sizes)])
            indices = np.concatenate([np.asarray(x, dtype="int64") for x in neighbours]) \
                if indptr[-1] else np.array([], dtype="int64")
            self.csr[name] = (indptr, indices, sizes)

    def aggregate(self, scores: pd.Series, prefix: str = "ctx") -> pd.DataFrame:
        """Max and mean of neighbours' scores per context (0 when no neighbours)."""
        require(scores.index.is_unique and set(scores.index) == set(self.hashes),
                "Scores must cover exactly this split's transactions")
        values = scores.reindex(self.hashes).to_numpy(float)
        require(np.isfinite(values).all(), "Scores must be finite")
        out = {}
        for name, (indptr, indices, sizes) in self.csr.items():
            gathered = values[indices]
            mx = np.zeros(len(values))
            mean = np.zeros(len(values))
            nonempty = sizes > 0
            if len(gathered):
                starts = indptr[:-1][nonempty]
                mx[nonempty] = np.maximum.reduceat(gathered, starts)
                mean[nonempty] = np.add.reduceat(gathered, starts) / sizes[nonempty]
            out[f"{prefix}_{name}_max"] = mx
            out[f"{prefix}_{name}_mean"] = mean
        result = pd.DataFrame(out, index=self.hashes).loc[self.input_order]
        result.index.name = "TRANSACTION_HASH"
        return result
