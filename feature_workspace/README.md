# Feature preparation workspace

Keep feature preparation separate from the supplied datasets and the verified baseline.

- Python feature modules belong in this package. The unused `notebooks/` directory was removed.
- Intermediate tables, feature matrices and experiment outputs belong in `generated/`. Its contents are ignored by Git, apart from the directory placeholder.
- Source training inputs remain at `../data/traning_data/train/`. Open source files for reading only; write derived data under `generated/`. Do not overwrite, move or rename source files.
- `paths.py` exposes `TRAIN_INPUT` and `output_path(relative_filename)`. The output helper rejects absolute paths, parent traversal outside `generated/`, and existing symlinks that lead outside it. Future feature writers must use the helper; it does not prevent unrelated code from opening arbitrary paths.
- Keep `TRANSACTION_HASH` as an alignment key. Do not use labels, family, difficulty, scenario/cohort IDs or raw identifiers as predictive features. Preserve the baseline's split isolation in feature experiments.

Example paths, run from the repository root:

```python
from feature_workspace.paths import TRAIN_INPUT, output_path

input_file = TRAIN_INPUT / "transactions.parquet"
feature_file = output_path("train_features.parquet")
# Read input_file; save the derived feature table to feature_file.
```

## Transaction preparation

From the repository root, using the dependencies in `requirements-baseline.txt`:

```sh
.venv/bin/python -m feature_workspace.prepare_transactions
.venv/bin/python -m unittest discover -s tests -v
```

The command separately reads transactions, transfers and market rates from train, practice validation and test. No labels or annotation metadata are read. Original files and permissions are unchanged. The measured baseline stays in `scripts/baseline.py` and `outputs/baseline_wallet_seed0/`; it is not retrained.

Outputs for each split are under `generated/transaction_tables_v1/{train,validation,test}/`:

- `transactions.parquet`: exactly one row per original transaction, in its original order, containing the 15 columns below.
- `transfer_legs.parquet`: all original detailed transfer records and their Arrow column types, sorted by `(TRANSACTION_HASH, TRANSFER_INDEX)`.

`generated/transaction_tables_v1/summary.json` and the aggregate reports `../reports/transaction_preparation.json` / `.md` contain counts, step distributions, top ten asset pairs (all payments and conversions separately), sample calculations, integrity checks and source/output hashes.

| Column | Definition |
|---|---|
| `TRANSACTION_HASH` | Original transaction key; never a predictive measurement |
| `TIMESTAMP` | Original Unix timestamp, integral seconds |
| `SENDER` | Transaction `ACCOUNT`, checked against the first leg |
| `RECEIVER` | JSON `EXTRA_TRANSACTION_FIELDS.Destination`, checked against the last leg |
| `SOURCE_ASSET` | First leg's `XRP` or `issuer:currency` identity |
| `SOURCE_AMOUNT` | Actual first-leg amount in whole XRP/token units |
| `DESTINATION_ASSET` | Last leg's full asset identity |
| `DESTINATION_AMOUNT` | Actual last-leg amount in whole units, checked against delivered JSON |
| `SOURCE_AMOUNT_USD` | Source amount × source asset's hourly reference price |
| `DESTINATION_AMOUNT_USD` | Destination amount × destination asset's hourly reference price |
| `TRANSFER_COUNT` | Number of executed legs |
| `IS_CONVERSION` | Whether any adjacent full asset identity changes |
| `CONVERSION_STEP_COUNT` | Number of adjacent asset identity changes, including issuer changes |
| `INTERMEDIARY_COUNT` | Number of distinct path addresses excluding the original sender and final receiver |
| `FEE_XRP` | Burned network fee, separately normalized from `SOURCE_FEE` drops |

`VALUE` is verified against `SOURCE_VALUE`: XRP `SOURCE_VALUE / 1,000,000 = VALUE`; IOU `SOURCE_VALUE = VALUE`. Amounts use already-normalized `VALUE` without dividing again. Source and destination prices use their own asset identities at `TIMESTAMP // 3600 * 3600`; each `PRICE_MICRO_USD` is divided by 1,000,000. Missing/duplicate rates and inconsistent units fail instead of being guessed. USD leg values are never summed; `SendMax` is never used as spending. Floating values retain input precision without intentional rounding; numeric comparisons allow only floating-point tolerance.

Conversion count is independent of transfer count: same-asset extra legs count as zero changes; `A → B → A` counts as two changes despite equal endpoint assets. These cases are tested even though every adjacent leg in the supplied splits changes asset. Intermediaries in the current conversions are liquidity providers, not evidence of AML layering.

Checks cover unique keys, exact leg coverage, matching ledger metadata, contiguous indices, continuous executed paths, endpoints, delivered JSON, units, fees, hourly price coverage, and finite outputs. Every destination USD value is cross-checked against the supplied `payments.py`. Both output Parquet tables are read back and compared. Existing tables with different content are refused: use a new `--run-name transaction_tables_v2` for future changed definitions.

These transaction-local fields do not learn statistics from other rows. When adding wallet-history features later, retain split isolation from the baseline and never combine synthetic worlds.

## Wallet transaction graphs

Build all three independent payment graphs with:

```sh
.venv/bin/python -m feature_workspace.build_wallet_graph
.venv/bin/python -m unittest discover -s tests -v
```

The builder reads only the prepared `transaction_tables_v1/{split}/transactions.parquet` tables for construction. Original transaction keys, senders, destinations and timestamps are read separately for validation. Detailed transfer legs and labels are not parsed to construct the graph. Every graph uses its complete supplied seven-day payment history.

A node is a unique `SENDER` or `RECEIVER` address. An edge is one directed `SENDER → RECEIVER` payment, identified by `TRANSACTION_HASH`. Repeated payments between a wallet pair remain separate edges. All 15 canonical columns are retained as edge attributes, including the 11 requested fields. `TRANSFER_COUNT` and `INTERMEDIARY_COUNT` remain attributes only; they never expand a payment into additional edges or intermediary nodes. A liquidity-provider address may still be a node if it independently appears as an actual user-payment endpoint.

Lossless graph files live under `generated/wallet_graphs_v1/{train,validation,test}/`:

- `edges.parquet`: the 15 canonical fields, sorted by `(TIMESTAMP, TRANSACTION_HASH)`. Original numeric values and types are preserved exactly.
- `nodes.parquet`: one row per wallet, sorted by address. Contains `WALLET`, `IN_DEGREE`, `OUT_DEGREE`, `TOTAL_DEGREE`, `UNIQUE_IN_NEIGHBORS`, `UNIQUE_OUT_NEIGHBORS`, `INCOMING_USD`, `OUTGOING_USD`, `TRANSACTION_COUNT`, `FIRST_TIMESTAMP`, `LAST_TIMESTAMP`.
- `manifest.json`: split identity, edge key/order, input fingerprint, graph counts and output hashes.

`generated/wallet_graphs_v1/summary.json` and `../reports/wallet_graphs.json` / `.md` contain measured statistics, degree summaries, transaction-count histograms, provenance and integrity results. Generated row-level artifacts stay outside Git. Existing differing graph tables/manifests are refused; use `--run-name wallet_graphs_v2` for changed definitions.

`IN_DEGREE` and `OUT_DEGREE` count transaction edges, while unique-neighbor columns count distinct counterparties. `TRANSACTION_COUNT` counts distinct incident transactions. A self-payment is one graph edge and one incident transaction but contributes one to each directional degree. Sink/source-only wallets are preserved. Incoming USD sums delivered values; outgoing USD sums spent values. These sums need not be equal for conversions, and neither includes the separately recorded XRP network fees.

Reusable access:

```python
from feature_workspace.build_wallet_graph import load_wallet_graph
from feature_workspace.paths import output_path

graph = load_wallet_graph(output_path("wallet_graphs_v1/train"), "train")
wallet = graph.nodes.WALLET.iloc[0]
outgoing_events = graph.outgoing(wallet)
incoming_events = graph.incoming(wallet)
later_events = graph.outgoing(wallet, after=1788220800)
# Optional through=... gives an inclusive upper time bound.
```

`build_wallet_graph(dataframe, split)` also builds the representation from a canonical subset, so future validation can rebuild fit and holdout contexts separately. Queries return chronological copies; do not mutate the graph's internal `edges` or `nodes` tables. Hash ordering resolves equal timestamps deterministically, not causally: a strictly later path hop must have a strictly greater timestamp. The query's `after` boundary enforces that rule. No path enumeration or AML pattern detection is implemented.

Validation checks every original transaction appears exactly once, all prepared attributes remain equal by hash, directions match `ACCOUNT`/`Destination`, degree and volume aggregates reconcile, and split wallet/hash intersections are empty. Rebuilding from shuffled inputs and reloading saved files produce identical graphs. Source and earlier preparation artifacts are fingerprinted before/after. Eight new graph tests cover hand-calculated statistics, parallel/reverse/self edges, sink wallets, time windows, label neutrality, cross-split rejection, invalid inputs, original transaction checks and corrupted artifacts.

Full train graphs are for offline analysis. For future supervised validation, rebuild graph contexts from the isolated fit/holdout payment subsets rather than using full train wallet statistics as held-out model features. No model training, label-derived statistic or pattern identifier is introduced by this graph stage.

## AML behaviour, flow and motif features

```sh
.venv/bin/python -m feature_workspace.aml_features          # all three splits, ~15 s each
```

`aml_features.compute_features(transactions)` takes one split's canonical table (the 15 columns above) and returns one float row per `TRANSACTION_HASH`, in input order. It reads no labels, annotations or legs. Hashes only break equal-timestamp ties. Each split is processed alone, using its whole seven-day window (offline context is allowed by the supplied README). Output: `generated/aml_features_v1/{split}/features.parquet` (198 features), re-read and compared after writing.

| Group | Examples |
|---|---|
| Transaction | log USD sent/delivered, conversion spread/steps, intermediaries, currency, issuer *role* (GBP issuer = 1, EUR issuer = 2, XRP = 0). Issuer addresses differ per split, so raw addresses are never used |
| Wallet / pair (whole week) | in/out counts, distinct counterparties, USD volumes, flow ratio, repeat ratio, amount vs wallet/pair median, pair count/rank/gaps, reverse pair, new-pair share within ±24 h |
| Windows (10 m, 1 h, 6 h, 24 h) | sender out / receiver in counts and distinct counterparties; sender receipts *before* and receiver forwards *after* (count, USD ratio, distinct) |
| Pass-through | gaps to adjacent events, similar-value (±10%) forwards/receipts, forwarded fraction within 6 h, amount CV within ±1 h |
| Motifs (bounded) | amount-consistent temporal chains (next hop within 1 h/6 h, 0.5–2× value) via DP; cycles back to the sender (≤8 hops, ≤6 h, ≤4,000 expansions); scatter-gather and gather-scatter reconvergence; bipartite butterflies; fan-out recipients forwarding within 1 h; nearby motif maxima |
| Tempo / burst clusters | k-th nearest event gaps, activity vs the wallet's weekly rate, and connected components of payments linked by consecutive same-wallet events ≤ 2/5/10/15 min apart (size, span, tempo, wallets, density, conversion share, amount spread, endpoint roles) |

Strictly later hops need strictly greater timestamps. Ratios with zero denominators are NaN (handled natively by the model). Fee, amount-decimal and hour-of-day features were removed after a 3-seed grouped-CV ablation showed no gain; they are also the least likely to carry over to another week.

`score_context.ScoreContext(transactions)` precomputes, per payment, its neighbour lists within the same split: same-wallet events within ±1 h/±6 h (also split by direction), receiver forwards and sender receipts within 6 h, amount-consistent chain neighbours, and burst-cluster members (5/15 min), always excluding the payment itself. `aggregate(scores)` returns the max and mean of the neighbours' scores. The scores passed in must not be fitted on the scored rows' labels. `scripts/aml_model.py` guarantees this with nested out-of-fold scores.

Tests: `tests/test_aml_features.py` covers toy cycles/chains with causal ordering, amount-inconsistent hops, repeated pairs and fans, burst linkage scales, invariance to input order and to extra label columns, issuer-role coding, context self-exclusion and coverage, scenario grouping, and the prediction CSV round trip.
