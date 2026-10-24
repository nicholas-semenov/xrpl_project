"""scenario_recon track: candidate motif-instance reconstruction features (label-free).

compute(transactions) -> DataFrame indexed by TRANSACTION_HASH, float columns prefixed "sr_".

Idea: planted motif instances (and their decoys) are built from wallet pairs that never
interact outside the instance, while background/relay traffic is dominated by recurring
relationships. Instances are reconstructed as connected components of *non-recurring*
payments linked through consecutive same-wallet events within a gap D (multi-scale).
Each payment gets robust instance statistics (consecutive-gap distribution, shape,
value flow, isolation from other traffic) and its role inside the instance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REC_H = 6 * 3600            # a pair "recurs" if it also occurs more than 6 h away
SCALES = (("3m", 180), ("7m", 420), ("15m", 900), ("30m", 1800), ("60m", 3600))


def _find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _recurring(s, r, t, H=REC_H):
    a, b = np.minimum(s, r), np.maximum(s, r)
    key = a.astype(np.int64) * 1_000_003 + b
    df = pd.DataFrame({"k": key, "t": t})
    g = df.groupby("k").t
    tmin, tmax = g.transform("min").to_numpy(), g.transform("max").to_numpy()
    cnt = g.transform("size").to_numpy()
    return ((t - tmin) > H) | ((tmax - t) > H), cnt


def _cluster_stats(root, mem, s, r, t, la, conv, usd_in, usd_out, src_asset, ext_counts):
    """Statistics per component for member payments (mem boolean)."""
    idx = np.where(mem)[0]
    d = pd.DataFrame({"root": root[idx], "e": idx, "t": t[idx], "s": s[idx], "r": r[idx], "la": la[idx],
                      "conv": conv[idx], "uin": usd_in[idx], "uout": usd_out[idx], "sa": src_asset[idx]})
    d = d.sort_values(["root", "t", "e"])
    g = d.groupby("root", sort=False)
    size = g.size()
    span = g.t.max() - g.t.min()
    # consecutive gaps inside instance
    gp = d.t.diff().to_numpy()
    first = d.root.ne(d.root.shift()).to_numpy()
    gp[first] = np.nan
    gd = pd.DataFrame({"root": d.root.to_numpy(), "g": gp / 60}).dropna()
    gg = gd.groupby("root").g
    gstats = pd.DataFrame({"gap_med": gg.median(), "gap_min": gg.min(), "gap_max": gg.max(),
                           "gap_cv": gg.std() / gg.mean()})
    ws = pd.concat([pd.DataFrame({"root": d.root, "w": d.s, "o": 1, "i": 0, "vo": d.uout, "vi": 0.0}),
                    pd.DataFrame({"root": d.root, "w": d.r, "o": 0, "i": 1, "vo": 0.0, "vi": d.uin})])
    pw = ws.groupby(["root", "w"]).agg(o=("o", "sum"), i=("i", "sum"), vo=("vo", "sum"), vi=("vi", "sum"))
    pw["deg"] = pw.o + pw.i
    pr = pw.reset_index().groupby("root")
    src = pw[(pw.o > 0) & (pw.i == 0)].reset_index().groupby("root")
    snk = pw[(pw.i > 0) & (pw.o == 0)].reset_index().groupby("root")
    wallets = pr.size()
    n_src = src.size().reindex(size.index).fillna(0)
    n_snk = snk.size().reindex(size.index).fillna(0)
    v_src = src.vo.sum().reindex(size.index)
    v_snk = snk.vi.sum().reindex(size.index)
    hub = pr.deg.max()
    # flow-consistent links: member f sent by the receiver of member e, later
    fe = d[["root", "r", "t"]].rename(columns={"r": "w", "t": "te"})
    ff = d[["root", "s", "t"]].rename(columns={"s": "w", "t": "tf"})
    fl = fe.merge(ff, on=["root", "w"])
    fl = fl[fl.tf > fl.te]
    flow_links = fl.groupby("root").size().reindex(size.index).fillna(0)
    # reciprocated wallet pairs
    pp = d[["root", "s", "r"]].drop_duplicates()
    rp = pp.merge(pp.rename(columns={"s": "r", "r": "s"}), on=["root", "s", "r"])
    recip = rp.groupby("root").size().reindex(size.index).fillna(0) / 2
    # isolation: other events of instance wallets during the instance span (+-15 min)
    ext = pd.Series(ext_counts, index=size.index) if ext_counts is not None else None
    st = pd.DataFrame({
        "size": size, "wallets": wallets, "span_min": span / 60,
        "density": size / wallets, "n_src": n_src, "n_snk": n_snk,
        "internal_share": (wallets - n_src - n_snk) / wallets, "hub_share": hub / (2 * size),
        "flow_links": flow_links / size, "recip": recip,
        "retention": v_snk / v_src, "conv_share": g.conv.mean(), "la_std": g.la.std(), "la_mean": g.la.mean(),
        "n_src_asset": g.sa.nunique()})
    st = st.join(gstats)
    return st, pw


def compute(transactions: pd.DataFrame) -> pd.DataFrame:
    tx = transactions.reset_index(drop=True)
    wl = pd.Index(pd.unique(np.r_[tx.SENDER.to_numpy(), tx.RECEIVER.to_numpy()]))
    s = wl.get_indexer(tx.SENDER).astype(np.int64)
    r = wl.get_indexer(tx.RECEIVER).astype(np.int64)
    t = tx.TIMESTAMP.to_numpy(float)
    n, nw = len(t), len(wl)
    usd_out = tx.SOURCE_AMOUNT_USD.to_numpy(float)
    usd_in = tx.DESTINATION_AMOUNT_USD.to_numpy(float)
    la = np.log(np.maximum(usd_out, 1e-9))
    conv = tx.IS_CONVERSION.to_numpy(float)
    src_asset = pd.factorize(tx.SOURCE_ASSET)[0]

    rec, pair_cnt = _recurring(s, r, t)
    novel = ~rec
    out = {"sr_recurring": rec.astype(float)}

    # all-event lists per wallet (for isolation counts)
    ev_all = pd.DataFrame({"w": np.r_[s, r], "e": np.r_[np.arange(n), np.arange(n)], "t": np.r_[t, t]})
    ev_all = ev_all.sort_values(["w", "t", "e"], kind="mergesort")
    w_ptr = np.searchsorted(ev_all.w.to_numpy(), np.arange(nw + 1))
    w_times = ev_all.t.to_numpy()

    idx = np.where(novel)[0]
    ev = pd.DataFrame({"w": np.r_[s[idx], r[idx]], "e": np.r_[idx, idx], "t": np.r_[t[idx], t[idx]]})
    ev = ev.sort_values(["w", "t", "e"], kind="mergesort")
    w, e, tt = ev.w.to_numpy(), ev.e.to_numpy(), ev.t.to_numpy()
    same = w[1:] == w[:-1]
    gap = np.where(same, tt[1:] - tt[:-1], np.inf)
    order = np.argsort(gap, kind="mergesort")
    order = order[np.isfinite(gap[order])]
    parent = np.arange(n)
    k = 0
    prev_size = None
    for name, D in SCALES:
        while k < len(order) and gap[order[k]] <= D:
            j = order[k]
            x, y = _find(parent, e[j]), _find(parent, e[j + 1])
            if x != y:
                parent[y] = x
            k += 1
        root = np.array([_find(parent, q) for q in range(n)])
        # isolation: per instance, events of its wallets within [t0-15m, t1+15m] that are not members
        st, pw = _cluster_stats(root, novel, s, r, t, la, conv, usd_in, usd_out, src_asset, None)
        tr = pd.DataFrame({"root": root[idx], "t": t[idx]}).groupby("root").t.agg(["min", "max"])
        pwr = pw.reset_index()[["root", "w", "deg"]].join(tr, on="root")
        lo = w_ptr[pwr.w.to_numpy()]
        hi = w_ptr[pwr.w.to_numpy() + 1]
        cnt = np.empty(len(pwr))
        tmin, tmax = pwr["min"].to_numpy() - 900, pwr["max"].to_numpy() + 900
        for i in range(len(pwr)):
            seg = w_times[lo[i]:hi[i]]
            cnt[i] = np.searchsorted(seg, tmax[i], "right") - np.searchsorted(seg, tmin[i], "left")
        pwr["ext"] = cnt - pwr.deg.to_numpy()
        ext = pwr.groupby("root").ext.sum()
        st["ext_ratio"] = ext.reindex(st.index) / (2 * st["size"])
        rows = st.reindex(root)
        cols = ["size", "wallets", "span_min", "density", "n_src", "n_snk", "internal_share", "hub_share",
                "flow_links", "recip", "retention", "conv_share", "la_std", "gap_med", "gap_min", "gap_max",
                "gap_cv", "ext_ratio", "n_src_asset"]
        for c in cols:
            v = rows[c].to_numpy(float)
            v[~novel] = np.nan
            out[f"sr_{name}_{c}"] = v
        out[f"sr_{name}_amt_vs_mean"] = np.where(novel, la - rows["la_mean"].to_numpy(float), np.nan)
        # role: endpoint degree inside the instance
        degs = pw.deg
        for side, owner in (("s", s), ("r", r)):
            kk = pd.MultiIndex.from_arrays([root, owner])
            v = degs.reindex(kk).to_numpy(float)
            v[~novel] = np.nan
            out[f"sr_{name}_{side}_deg"] = v
        size_now = rows["size"].to_numpy(float)
        if prev_size is not None:
            out[f"sr_{name}_growth"] = np.where(novel, size_now / prev_size, np.nan)
        prev_size = size_now
    return pd.DataFrame(out, index=pd.Index(tx.TRANSACTION_HASH, name="TRANSACTION_HASH")).astype(float)
