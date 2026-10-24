"""Label-free AML behaviour, flow and bounded motif features, one row per transaction.

Run: .venv/bin/python -m feature_workspace.aml_features
Every split is processed separately from its own prepared transaction table. The
whole seven-day window is used as context (offline investigation is permitted by
the supplied README). Labels, scenario/cohort metadata and hashes are never inputs;
hashes only break equal-timestamp ties deterministically.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict

import numpy as np
import pandas as pd

from feature_workspace.paths import output_path
from feature_workspace.prepare_transactions import COLUMNS, SPLITS, require

INPUT_RUN = "transaction_tables_v1"
WINDOWS = {"10m": 600, "1h": 3600, "6h": 21600, "24h": 86400}
CHAIN_GAPS = {"1h": 3600, "6h": 21600}
CHAIN_RATIO = (0.5, 2.0)        # next-hop USD value relative to the previous hop
CYCLE_MAX_HOPS = 8
CYCLE_HORIZON = 6 * 3600
CYCLE_MAX_EXPANSIONS = 4000
CURRENCIES = {"XRP": 0, "USD": 1, "EUR": 2, "GBP": 3}


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Ratio with NaN where the denominator is zero; tree models handle NaN natively."""
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    out = np.full(numerator.shape, np.nan)
    ok = denominator > 0
    out[ok] = numerator[ok] / denominator[ok]
    return out


def _asset_parts(assets: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Currency code and a structural issuer code that is comparable across splits.

    Issuer addresses differ per split, so issuers are coded by role: 1 for the
    issuer of GBP, 2 for the issuer of EUR, 3 for any other issuer, 0 for XRP.
    """
    currency = assets.str.split(":").str[-1].map(CURRENCIES).fillna(-1).to_numpy(int)
    issuers = assets.where(assets.ne("XRP")).str.split(":").str[0]
    role = {}
    for code, name in ((2, "EUR"), (1, "GBP")):
        for issuer in issuers[assets.str.endswith(f":{name}")].unique():
            role[issuer] = code
    return currency, issuers.map(role).fillna(3).where(issuers.notna(), 0).to_numpy(int)


class _Events:
    """Per-wallet chronological event index for one direction (out or in)."""

    def __init__(self, owner: np.ndarray, n_wallets: int):
        positions = defaultdict(list)
        for pos, w in enumerate(owner):
            positions[w].append(pos)
        self.pos = [np.asarray(positions.get(w, []), dtype="int64") for w in range(n_wallets)]

    def span(self, wallet: int, times: np.ndarray, lo: float, hi: float,
             lo_open: bool = False, hi_open: bool = False) -> np.ndarray:
        """Edge positions of this wallet with time in the requested interval."""
        p = self.pos[wallet]
        if len(p) == 0:
            return p
        t = times[p]
        a = np.searchsorted(t, lo, "right" if lo_open else "left")
        b = np.searchsorted(t, hi, "left" if hi_open else "right")
        return p[a:b]


def _find(parent: np.ndarray, x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _tempo_features(s: np.ndarray, r: np.ndarray, t: np.ndarray, a: np.ndarray,
                    conversion: np.ndarray, nw: int) -> dict[str, np.ndarray]:
    """Local tempo and burst-cluster context, using no labels or scenario metadata.

    A burst cluster at scale D links two payments when they share a wallet and are
    consecutive in that wallet's event history with a gap of at most D seconds.
    Each payment receives the summary statistics of its connected component.
    """
    n = len(t)
    events = [[] for _ in range(nw)]
    for e in range(n):
        events[s[e]].append(e)
        events[r[e]].append(e)
    nearest = np.full(n, np.inf)
    links = []
    for evs in events:
        evs.sort(key=lambda e: (t[e], e))
        times = t[evs]
        gaps = np.diff(times)
        for i, gap in enumerate(gaps):
            x, y = evs[i], evs[i + 1]
            nearest[x] = min(nearest[x], gap)
            nearest[y] = min(nearest[y], gap)
            links.append((gap, x, y))
    links.sort()
    out = {"nearest_event_gap_min": np.where(np.isinf(nearest), np.nan, nearest / 60)}
    log_a = np.log(a)
    parent = np.arange(n)
    k = 0
    for name, scale in (("2m", 120), ("5m", 300), ("10m", 600), ("15m", 900)):
        while k < len(links) and links[k][0] <= scale:
            _, x, y = links[k]
            rx, ry = _find(parent, x), _find(parent, y)
            if rx != ry:
                parent[ry] = rx
            k += 1
        root = np.array([_find(parent, e) for e in range(n)])
        comp = pd.DataFrame({"root": root, "t": t, "la": log_a, "conv": conversion, "s": s, "r": r})
        grouped = comp.groupby("root")
        size = grouped.size()
        span = grouped.t.max() - grouped.t.min()
        wallets = pd.concat([comp[["root", "s"]].rename(columns={"s": "w"}),
                             comp[["root", "r"]].rename(columns={"r": "w"})]).drop_duplicates().groupby("root").size()
        senders = comp[["root", "s"]].drop_duplicates().groupby("root").size()
        receivers = comp[["root", "r"]].drop_duplicates().groupby("root").size()
        stats = pd.DataFrame({
            "size": size, "span_min": span / 60, "wallets": wallets,
            "tempo_min": (span / 60) / (size - 1).where(size > 1),
            "density": size / wallets, "senders": senders, "receivers": receivers,
            "conv_share": grouped.conv.mean(), "log_usd_std": grouped.la.std(),
            "log_usd_mean": grouped.la.mean()})
        rows = stats.loc[root]
        for column in stats.columns:
            out[f"burst_{name}_{column}"] = rows[column].to_numpy(float)
        # Role of each endpoint inside its cluster: its other events in the same component.
        member = pd.concat([pd.DataFrame({"root": root, "w": s}), pd.DataFrame({"root": root, "w": r})])
        per_wallet = member.groupby(["root", "w"]).size()
        for side, owner in (("s", s), ("r", r)):
            keys = pd.MultiIndex.from_arrays([root, owner])
            out[f"burst_{name}_{side}_events"] = per_wallet.reindex(keys).to_numpy(float) - 1
        out[f"burst_{name}_amount_vs_mean"] = log_a - rows["log_usd_mean"].to_numpy(float)

    # Wallet-relative tempo: k-th nearest other event and activity versus the wallet's weekly rate.
    week = max(t.max() - t.min(), 1)
    wallet_times = [t[np.asarray(evs, dtype="int64")] for evs in events]
    for side, owner in (("s", s), ("r", r)):
        kth = {k: np.full(n, np.nan) for k in (1, 2, 3)}
        burst = {w: np.zeros(n) for w in ("10m", "1h", "3h")}
        for e in range(n):
            times = wallet_times[owner[e]]
            gaps = np.sort(np.abs(times - t[e]))[1:4]  # drop this event itself
            for k in range(len(gaps)):
                kth[k + 1][e] = gaps[k] / 60
            for w, W in (("10m", 600), ("1h", 3600), ("3h", 10800)):
                count = np.searchsorted(times, t[e] + W, "right") - np.searchsorted(times, t[e] - W, "left") - 1
                burst[w][e] = count / (len(times) * 2 * W / week)
        for k in kth:
            out[f"{side}_gap_kth{k}_min"] = kth[k]
        for w in burst:
            out[f"{side}_activity_ratio_{w}"] = burst[w]
    return out


def _structure_features(s: np.ndarray, r: np.ndarray, t: np.ndarray, a: np.ndarray,
                        conversion: np.ndarray, out_ev: "_Events", in_ev: "_Events",
                        succs: list[np.ndarray]) -> dict[str, np.ndarray]:
    """Local subgraph density and amount-consistent flow clusters around each payment.

    Planted motifs are small dense subgraphs, while background payments touching a
    motif wallet are star-like; flow clusters link only chain-consistent forwards,
    so slow motifs stay separable without percolating through background traffic.
    Ego isolation, amount CV and sender/receiver counts were tried and removed
    (3-seed grouped CV: 0.6163 without vs 0.6112 with).
    """
    n = len(t)
    out: dict[str, np.ndarray] = {}

    def counterparties(w: int, lo: float, hi: float) -> set:
        return set(r[out_ev.span(w, t, lo, hi)].tolist()) | set(s[in_ev.span(w, t, lo, hi)].tolist())

    for name, W in (("30m", 1800), ("1h", 3600), ("3h", 10800), ("6h", 21600)):
        common = np.zeros(n)
        jaccard = np.zeros(n)
        ego_edges = np.zeros(n)
        ego_wallets = np.zeros(n)
        ego_tempo = np.full(n, np.nan)
        ego_conv = np.zeros(n)
        for e in range(n):
            lo, hi = t[e] - W, t[e] + W
            ns = counterparties(s[e], lo, hi) - {s[e], r[e]}
            nr = counterparties(r[e], lo, hi) - {s[e], r[e]}
            common[e] = len(ns & nr)
            union = ns | nr
            jaccard[e] = common[e] / len(union) if union else 0.0
            nodes = union | {s[e], r[e]}
            edges = set()
            for w in nodes:
                for pos in out_ev.span(w, t, lo, hi):
                    if r[pos] in nodes:
                        edges.add(int(pos))
            ego_wallets[e] = len(nodes)
            ego_edges[e] = len(edges)
            if len(edges) > 1:
                times = t[list(edges)]
                ego_tempo[e] = (times.max() - times.min()) / 60 / (len(edges) - 1)
            ego_conv[e] = conversion[list(edges)].mean()
        out[f"common_neighbours_{name}"] = common
        out[f"neighbour_jaccard_{name}"] = jaccard
        out[f"ego_edges_{name}"] = ego_edges
        out[f"ego_density_{name}"] = ego_edges / ego_wallets
        out[f"ego_tempo_min_{name}"] = ego_tempo
        out[f"ego_conversion_share_{name}"] = ego_conv

    # Flow clusters: connected components of amount-consistent forwards within a gap.
    log_a = np.log(a)
    links = sorted((int(t[q] - t[e]), e, int(q)) for e in range(n) for q in succs[e])
    parent = np.arange(n)
    k = 0
    for name, gap in (("30m", 1800), ("2h", 7200), ("6h", 21600)):
        while k < len(links) and links[k][0] <= gap:
            _, x, y = links[k]
            rx, ry = _find(parent, x), _find(parent, y)
            if rx != ry:
                parent[ry] = rx
            k += 1
        root = np.array([_find(parent, e) for e in range(n)])
        comp = pd.DataFrame({"root": root, "t": t, "la": log_a, "conv": conversion, "s": s, "r": r})
        grouped = comp.groupby("root")
        size = grouped.size()
        span = grouped.t.max() - grouped.t.min()
        wallets = pd.concat([comp[["root", "s"]].rename(columns={"s": "w"}),
                             comp[["root", "r"]].rename(columns={"r": "w"})]).drop_duplicates().groupby("root").size()
        stats = pd.DataFrame({"size": size, "span_min": span / 60, "wallets": wallets,
                              "tempo_min": (span / 60) / (size - 1).where(size > 1),
                              "conv_share": grouped.conv.mean(), "log_usd_std": grouped.la.std(),
                              "value_retained": np.exp(grouped.la.last() - grouped.la.first())})
        rows = stats.loc[root]
        for column in stats.columns:
            out[f"flow_{name}_{column}"] = rows[column].to_numpy(float)
    return out


def compute_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Compute features for one split's canonical payments, indexed by hash in input order."""
    require(set(COLUMNS) <= set(transactions.columns), "Missing canonical columns")
    require(transactions.TRANSACTION_HASH.is_unique, "Duplicate transaction hashes")
    df = transactions.loc[:, list(COLUMNS)].sort_values(
        ["TIMESTAMP", "TRANSACTION_HASH"], kind="mergesort").reset_index(drop=True)
    n = len(df)
    wallets = pd.Index(sorted(set(df.SENDER) | set(df.RECEIVER)))
    s = wallets.get_indexer(df.SENDER)
    r = wallets.get_indexer(df.RECEIVER)
    t = df.TIMESTAMP.to_numpy("int64")
    a = df.DESTINATION_AMOUNT_USD.to_numpy(float)
    a_src = df.SOURCE_AMOUNT_USD.to_numpy(float)
    require(np.isfinite(a).all() and (a > 0).all(), "Delivered USD values must be positive")
    nw = len(wallets)
    out_ev, in_ev = _Events(s, nw), _Events(r, nw)
    f: dict[str, np.ndarray] = {}

    # Group A: transaction-local attributes.
    f["log_usd"] = np.log1p(a)
    f["log_src_usd"] = np.log1p(a_src)
    f["conversion_spread"] = a_src / a - 1
    f["is_conversion"] = df.IS_CONVERSION.to_numpy(float)
    f["conversion_steps"] = df.CONVERSION_STEP_COUNT.to_numpy(float)
    f["intermediaries"] = df.INTERMEDIARY_COUNT.to_numpy(float)
    src_ccy, src_iss = _asset_parts(df.SOURCE_ASSET)
    dst_ccy, dst_iss = _asset_parts(df.DESTINATION_ASSET)
    f["src_currency"], f["src_issuer"] = src_ccy.astype(float), src_iss.astype(float)
    f["dst_currency"], f["dst_issuer"] = dst_ccy.astype(float), dst_iss.astype(float)
    f["cross_issuer_same_currency"] = ((src_ccy == dst_ccy) & (src_iss != dst_iss)).astype(float)
    # Fee, amount-decimal and hour-of-day features were dropped: 3-seed grouped-CV ablation
    # showed no gain, and they are the least likely to transfer to a different week.

    # Group B: whole-window wallet and pair behaviour.
    out_n = np.bincount(s, minlength=nw).astype(float)
    in_n = np.bincount(r, minlength=nw).astype(float)
    out_usd = np.bincount(s, weights=a_src, minlength=nw)
    in_usd = np.bincount(r, weights=a, minlength=nw)
    pairs = pd.DataFrame({"s": s, "r": r})
    out_uniq = pairs.drop_duplicates().groupby("s").size().reindex(range(nw), fill_value=0).to_numpy(float)
    in_uniq = pairs.drop_duplicates().groupby("r").size().reindex(range(nw), fill_value=0).to_numpy(float)
    out_med = pd.Series(a).groupby(s).median().reindex(range(nw)).to_numpy()
    in_med = pd.Series(a).groupby(r).median().reindex(range(nw)).to_numpy()
    first_t = np.minimum(pd.Series(t).groupby(s).min().reindex(range(nw), fill_value=np.inf).to_numpy(float),
                         pd.Series(t).groupby(r).min().reindex(range(nw), fill_value=np.inf).to_numpy(float))
    last_t = np.maximum(pd.Series(t).groupby(s).max().reindex(range(nw), fill_value=-np.inf).to_numpy(float),
                        pd.Series(t).groupby(r).max().reindex(range(nw), fill_value=-np.inf).to_numpy(float))
    for side, w in (("s", s), ("r", r)):
        f[f"{side}_out_n"], f[f"{side}_in_n"] = out_n[w], in_n[w]
        f[f"{side}_out_uniq"], f[f"{side}_in_uniq"] = out_uniq[w], in_uniq[w]
        f[f"{side}_log_out_usd"], f[f"{side}_log_in_usd"] = np.log1p(out_usd[w]), np.log1p(in_usd[w])
        f[f"{side}_flow_ratio"] = _safe_ratio(out_usd[w], in_usd[w])
        f[f"{side}_out_repeat"] = _safe_ratio(out_n[w], out_uniq[w])
        f[f"{side}_in_repeat"] = _safe_ratio(in_n[w], in_uniq[w])
        f[f"{side}_hours_since_first"] = (t - first_t[w]) / 3600
        f[f"{side}_hours_until_last"] = (last_t[w] - t) / 3600
    f["amount_vs_s_out_median"] = a / out_med[s]
    f["amount_vs_r_in_median"] = a / in_med[r]
    pair_key = s.astype("int64") * nw + r
    pair_series = pd.Series(pair_key)
    pair_n = pair_series.map(pair_series.value_counts()).to_numpy(float)
    f["pair_n"] = pair_n
    f["pair_rank_frac"] = (pair_series.groupby(pair_series).cumcount().to_numpy() + 1) / pair_n
    reverse_counts = pd.Series(pair_key).value_counts()
    f["reverse_pair_n"] = pd.Series(r.astype("int64") * nw + s).map(reverse_counts).fillna(0).to_numpy(float)
    f["pair_share_of_s_out"] = pair_n / out_n[s]
    f["pair_share_of_r_in"] = pair_n / in_n[r]
    pair_t = pd.Series(t.astype(float))
    f["pair_prev_gap_h"] = (pair_t - pair_t.groupby(pair_key).shift(1)).to_numpy() / 3600
    f["pair_next_gap_h"] = (pair_t.groupby(pair_key).shift(-1) - pair_t).to_numpy() / 3600
    pair_amount = pd.Series(a)
    pair_med = pair_amount.groupby(pair_key).transform("median").to_numpy()
    f["amount_vs_pair_median"] = a / pair_med
    pair_rank = pair_series.groupby(pair_series).cumcount().to_numpy() + 1
    f["pair_rank"] = pair_rank.astype(float)
    first_pair = (pair_rank == 1).astype(float)
    for side, owner in (("s_out", s), ("r_in", r)):
        share = np.full(n, np.nan)
        ev = out_ev if side == "s_out" else in_ev
        for e in range(n):
            near = ev.span(owner[e], t, t[e] - 86400, t[e] + 86400)
            share[e] = first_pair[near].mean()
        f[f"{side}_new_pair_share_24h"] = share

    # Group C/D: windowed flow, intermediary and motif features (per transaction loop).
    names = []
    for w in WINDOWS:
        names += [f"s_out_win_{w}", f"s_out_uniq_win_{w}", f"r_in_win_{w}", f"r_in_uniq_win_{w}",
                  f"s_in_before_{w}", f"s_in_before_usd_ratio_{w}", f"s_in_before_uniq_{w}",
                  f"r_out_after_{w}", f"r_out_after_usd_ratio_{w}", f"r_out_after_uniq_{w}"]
    names += ["s_gap_prev_in_h", "s_prev_in_ratio", "r_gap_next_out_h", "r_next_out_ratio",
              "s_gap_prev_out_h", "s_gap_next_out_h", "r_gap_prev_in_h", "r_gap_next_in_h",
              "r_similar_out_1h", "r_similar_out_24h", "r_min_reldiff_out_6h",
              "s_similar_in_1h", "s_similar_in_24h", "s_min_reldiff_in_6h",
              "r_forward_fraction_6h", "s_out_cv_1h", "r_in_cv_1h",
              "return_to_s_24h", "sg_siblings_6h", "sg_shared_dest_6h", "gs_shared_source_6h",
              "butterflies_1h", "butterflies_6h", "fanout_recipients_forward_1h",
              "s_in_fresh_senders_6h", "r_out_fresh_receivers_6h"]
    loop = {name: np.full(n, np.nan) for name in names}
    eps = 1e-12

    for e in range(n):
        se, re_, te, ae = s[e], r[e], t[e], a[e]
        for w, W in WINDOWS.items():
            so = out_ev.span(se, t, te - W, te + W)
            loop[f"s_out_win_{w}"][e] = len(so)
            loop[f"s_out_uniq_win_{w}"][e] = len(np.unique(r[so]))
            ri = in_ev.span(re_, t, te - W, te + W)
            loop[f"r_in_win_{w}"][e] = len(ri)
            loop[f"r_in_uniq_win_{w}"][e] = len(np.unique(s[ri]))
            sib = in_ev.span(se, t, te - W, te, hi_open=True)
            loop[f"s_in_before_{w}"][e] = len(sib)
            loop[f"s_in_before_usd_ratio_{w}"][e] = a[sib].sum() / ae
            loop[f"s_in_before_uniq_{w}"][e] = len(np.unique(s[sib]))
            roa = out_ev.span(re_, t, te, te + W, lo_open=True)
            loop[f"r_out_after_{w}"][e] = len(roa)
            loop[f"r_out_after_usd_ratio_{w}"][e] = a_src[roa].sum() / ae
            loop[f"r_out_after_uniq_{w}"][e] = len(np.unique(r[roa]))

        prev_in = in_ev.span(se, t, -np.inf, te, hi_open=True)
        if len(prev_in):
            loop["s_gap_prev_in_h"][e] = (te - t[prev_in[-1]]) / 3600
            loop["s_prev_in_ratio"][e] = a[prev_in[-1]] / ae
        next_out = out_ev.span(re_, t, te, np.inf, lo_open=True)
        if len(next_out):
            loop["r_gap_next_out_h"][e] = (t[next_out[0]] - te) / 3600
            loop["r_next_out_ratio"][e] = a_src[next_out[0]] / ae
        po = out_ev.span(se, t, -np.inf, te, hi_open=True)
        if len(po):
            loop["s_gap_prev_out_h"][e] = (te - t[po[-1]]) / 3600
        no = out_ev.span(se, t, te, np.inf, lo_open=True)
        if len(no):
            loop["s_gap_next_out_h"][e] = (t[no[0]] - te) / 3600
        pi = in_ev.span(re_, t, -np.inf, te, hi_open=True)
        if len(pi):
            loop["r_gap_prev_in_h"][e] = (te - t[pi[-1]]) / 3600
        ni = in_ev.span(re_, t, te, np.inf, lo_open=True)
        if len(ni):
            loop["r_gap_next_in_h"][e] = (t[ni[0]] - te) / 3600

        # Similar-value pass-through (strictly later forwards / strictly earlier receipts).
        fwd = out_ev.span(re_, t, te, te + 86400, lo_open=True)
        rel = np.abs(a_src[fwd] / ae - 1)
        loop["r_similar_out_1h"][e] = np.sum((rel <= 0.1) & (t[fwd] <= te + 3600))
        loop["r_similar_out_24h"][e] = np.sum(rel <= 0.1)
        in6 = t[fwd] <= te + 21600
        if in6.any():
            loop["r_min_reldiff_out_6h"][e] = rel[in6].min()
        back = in_ev.span(se, t, te - 86400, te, hi_open=True)
        relb = np.abs(a[back] / ae - 1)
        loop["s_similar_in_1h"][e] = np.sum((relb <= 0.1) & (t[back] >= te - 3600))
        loop["s_similar_in_24h"][e] = np.sum(relb <= 0.1)
        b6 = t[back] >= te - 21600
        if b6.any():
            loop["s_min_reldiff_in_6h"][e] = relb[b6].min()
        r_in6 = in_ev.span(re_, t, te - 21600, te)
        loop["r_forward_fraction_6h"][e] = a_src[fwd[in6]].sum() / max(a[r_in6].sum(), eps)
        so1 = out_ev.span(se, t, te - 3600, te + 3600)
        if len(so1) >= 2:
            loop["s_out_cv_1h"][e] = a[so1].std() / a[so1].mean()
        ri1 = in_ev.span(re_, t, te - 3600, te + 3600)
        if len(ri1) >= 2:
            loop["r_in_cv_1h"][e] = a[ri1].std() / a[ri1].mean()
        loop["return_to_s_24h"][e] = np.sum(r[fwd] == se)

        # Scatter-gather: siblings S->M' near this S->M that reconverge on M's later destinations.
        dests = set(r[out_ev.span(re_, t, te, te + 86400, lo_open=True)].tolist())
        sib_edges = out_ev.span(se, t, te - 21600, te + 21600)
        siblings = {m for m in r[sib_edges].tolist() if m != re_}
        loop["sg_siblings_6h"][e] = len(siblings)
        shared = 0
        if dests:
            for m in siblings:
                md = r[out_ev.span(m, t, te - 21600, te + 86400)]
                if dests.intersection(md.tolist()):
                    shared += 1
        loop["sg_shared_dest_6h"][e] = shared
        # Gather side: other senders into R near now sharing an upstream source with S.
        sources = set(s[in_ev.span(se, t, te - 86400, te, hi_open=True)].tolist())
        co_senders = {x for x in s[in_ev.span(re_, t, te - 21600, te + 21600)].tolist() if x != se}
        gshared = 0
        if sources:
            for x in co_senders:
                xs = s[in_ev.span(x, t, te - 86400 - 21600, te + 21600)]
                if sources.intersection(xs.tolist()):
                    gshared += 1
        loop["gs_shared_source_6h"][e] = gshared

        # Bipartite butterflies: S'->R' edges among R's co-senders and S's co-receivers.
        for w, W in (("1h", 3600), ("6h", 21600)):
            rs = set(r[out_ev.span(se, t, te - W, te + W)].tolist()) - {re_}
            sr = set(s[in_ev.span(re_, t, te - W, te + W)].tolist()) - {se}
            count = 0
            if rs and sr:
                for x in sr:
                    count += sum(1 for y in r[out_ev.span(x, t, te - W, te + W)].tolist() if y in rs)
            loop[f"butterflies_{w}"][e] = count

        # Fan-out with rapid forwarding: share of S's nearby recipients forwarding within 1h.
        if len(so1):
            forwarded = 0
            for pos in so1:
                if len(out_ev.span(r[pos], t, t[pos], t[pos] + 3600, lo_open=True)):
                    forwarded += 1
            loop["fanout_recipients_forward_1h"][e] = forwarded / len(so1)
        # Novel counterparties: senders into S (6h before) / receivers of R (6h after) first seen then.
        s_in6 = in_ev.span(se, t, te - 21600, te, hi_open=True)
        if len(s_in6):
            loop["s_in_fresh_senders_6h"][e] = np.mean(first_t[s[s_in6]] >= te - 21600 - 3600)
        r_out6 = out_ev.span(re_, t, te, te + 21600, lo_open=True)
        if len(r_out6):
            loop["r_out_fresh_receivers_6h"][e] = np.mean(last_t[r[r_out6]] <= te + 21600 + 3600)
    f.update(loop)

    # Amount-consistent temporal chains through this edge (dynamic programming on time order).
    succ_cache: dict[str, list[np.ndarray]] = {}
    for g, gap in CHAIN_GAPS.items():
        succs = []
        for e in range(n):
            cand = out_ev.span(r[e], t, t[e], t[e] + gap, lo_open=True)
            ratio = a_src[cand] / a[e]
            succs.append(cand[(ratio >= CHAIN_RATIO[0]) & (ratio <= CHAIN_RATIO[1])])
        succ_cache[g] = succs
        fwd_len = np.ones(n)
        for e in range(n - 1, -1, -1):
            if len(succs[e]):
                fwd_len[e] = 1 + fwd_len[succs[e]].max()
        bwd_len = np.ones(n)
        n_pred = np.zeros(n)
        for e in range(n):
            for q in succs[e]:
                n_pred[q] += 1
                bwd_len[q] = max(bwd_len[q], bwd_len[e] + 1)
        f[f"chain_fwd_len_{g}"] = fwd_len
        f[f"chain_bwd_len_{g}"] = bwd_len
        f[f"chain_total_len_{g}"] = fwd_len + bwd_len - 1
        f[f"chain_successors_{g}"] = np.array([len(x) for x in succs], dtype=float)
        f[f"chain_predecessors_{g}"] = n_pred
        if g == "6h":
            next_gap = np.full(n, np.nan)
            prev_gap = np.full(n, np.nan)
            for e in range(n):
                if len(succs[e]):
                    gap = (t[succs[e][0]] - t[e]) / 60
                    next_gap[e] = gap
                    for q in succs[e]:
                        lag = (t[q] - t[e]) / 60
                        if np.isnan(prev_gap[q]) or lag < prev_gap[q]:
                            prev_gap[q] = lag
            f["chain_next_gap_min"], f["chain_prev_gap_min"] = next_gap, prev_gap

    # Bounded amount-consistent cycle search returning value to the original sender.
    succs = succ_cache["6h"]
    cycle_hops = np.full(n, np.nan)
    cycle_hours = np.full(n, np.nan)
    cycle_value = np.full(n, np.nan)
    for e in range(n):
        origin, start = s[e], t[e]
        frontier, seen, expansions = [e], {e}, 0
        for hop in range(2, CYCLE_MAX_HOPS + 1):
            nxt = []
            for q in frontier:
                for p in succs[q]:
                    if p in seen or t[p] > start + CYCLE_HORIZON:
                        continue
                    seen.add(p)
                    expansions += 1
                    if r[p] == origin:
                        cycle_hops[e], cycle_hours[e] = hop, (t[p] - start) / 3600
                        cycle_value[e] = a[p] / a[e]
                        break
                    nxt.append(p)
                if not np.isnan(cycle_hops[e]):
                    break
            if not np.isnan(cycle_hops[e]) or not nxt or expansions > CYCLE_MAX_EXPANSIONS:
                break
            frontier = nxt
    f["cycle_hops"], f["cycle_hours"], f["cycle_value_ratio"] = cycle_hops, cycle_hours, cycle_value

    # Neighbourhood context: strongest motif signal among this wallet pair's nearby events.
    for name in ("chain_total_len_1h", "butterflies_1h", "sg_shared_dest_6h"):
        values = f[name]
        s_max = np.zeros(n)
        r_max = np.zeros(n)
        for e in range(n):
            so = out_ev.span(s[e], t, t[e] - 3600, t[e] + 3600)
            ri = in_ev.span(r[e], t, t[e] - 3600, t[e] + 3600)
            s_max[e] = np.nanmax(values[so]) if len(so) else 0
            r_max[e] = np.nanmax(values[ri]) if len(ri) else 0
        f[f"s_nearby_max_{name}"] = s_max
        f[f"r_nearby_max_{name}"] = r_max

    f.update(_tempo_features(s, r, t, a, df.IS_CONVERSION.to_numpy(float), nw))
    f.update(_structure_features(s, r, t, a, df.IS_CONVERSION.to_numpy(float), out_ev, in_ev,
                                 succ_cache["6h"]))

    result = pd.DataFrame(f, index=pd.Index(df.TRANSACTION_HASH, name="TRANSACTION_HASH"))
    result = result.loc[transactions.TRANSACTION_HASH]
    values = result.to_numpy()
    require(not np.isinf(values).any(), "Infinite feature values")
    require(len(result) == len(transactions) and result.index.is_unique, "Feature row coverage")
    return result.astype("float64")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="aml_features_v2")
    parser.add_argument("--splits", nargs="+", default=list(SPLITS))
    args = parser.parse_args()
    summary = {}
    for split in args.splits:
        start = time.perf_counter()
        source = output_path(f"{INPUT_RUN}/{split}/transactions.parquet")
        transactions = pd.read_parquet(source)
        features = compute_features(transactions)
        destination = output_path(f"{args.run_name}/{split}/features.parquet")
        destination.parent.mkdir(parents=True, exist_ok=True)
        features.reset_index().to_parquet(destination, index=False)
        reloaded = pd.read_parquet(destination).set_index("TRANSACTION_HASH")
        require(reloaded.index.equals(features.index), "Feature hash alignment changed on reload")
        require(np.allclose(reloaded.to_numpy(), features.to_numpy(), equal_nan=True),
                "Feature values changed on reload")
        summary[split] = {"rows": len(features), "features": features.shape[1],
                          "seconds": round(time.perf_counter() - start, 1)}
        print(split, json.dumps(summary[split]), flush=True)
    output_path(f"{args.run_name}/summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
