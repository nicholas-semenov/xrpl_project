"""Rank-average several prediction CSVs into one submission (equal weights by default).

Run: .venv/bin/python scripts/blend.py OUT.csv IN1.csv IN2.csv [...]
Each input must cover exactly the same TRANSACTION_HASH set; scores are converted to
rank percentiles, averaged, and the result stays in [0, 1] with test-file row order.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.aml_model import write_predictions  # noqa: E402

out, *inputs = sys.argv[1:]
frames = [pd.read_csv(p).set_index("TRANSACTION_HASH").score for p in inputs]
index = frames[0].index
assert all(set(f.index) == set(index) and f.index.is_unique for f in frames), "Hash sets differ"
blend = np.mean([f.loc[index].rank(pct=True).to_numpy() for f in frames], axis=0)
# Break rank-average ties with a tiny share of the mean raw score, then rescale into [0, 1].
blend = blend + 1e-6 * np.mean([f.loc[index].to_numpy() for f in frames], axis=0)
blend = (blend - blend.min()) / (blend.max() - blend.min())
write_predictions(Path(out), index, blend, index)
print(f"wrote {out}: {len(index)} rows, distinct {pd.Series(blend).nunique()}")
