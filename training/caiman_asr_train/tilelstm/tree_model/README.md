# `tree_model` — a learned config-ranking heuristic for the TileLang LSTM autotuner

A gradient-boosted **decision-tree** model (one per-backend **xgboost** GBDT) that ranks
autotune configs by predicted speed, so the autotuner compiles + benchmarks only the handful
most likely to be optimal. The package is **self-contained** and imports only
`caiman_asr_train.tilelstm.autotune` and the generic `caiman_asr_train.autotune`
(bench + cache I/O). It splits into two subpackages: **`core/`** is the
production path (spec measurement, featurization, data generation, training, and the deployable
ranker), **`aux/`** is a visualization layer (throughput plots). `core` never
imports `aux` — the dependency only ever points `aux → core`, keeping the production
path independent of the plotting code. It sits beside the autotuner it consumes,
importable as `caiman_asr_train.tilelstm.tree_model`.

Headline: in the realistic regime — some shapes benched on the target GPU — the model
surfaces a within-3%-of-optimal config in the top handful of its ranking for all but
**~3 of 5,857** (GPU, backend, H, batch) cells (grouped k-fold on the target GPU, mean over
5 seeds; see [Grouped-CV results](#grouped-cv-results)). All seven boosters share one
hyperparameter config (384 trees × 31 leaves, lr 0.111).

## What it does (the K metric)

For each `(GPU, backend, H, tune_batch)` shape, rank every candidate config by the model's
predicted within-shape **log-slowdown** (`log(latency) − log(shape_best)`), then ask: how far
down that ranking is the first config measured within N% of the true optimum? That depth is
**K** (tie-aware). Lower = a near-optimal config surfaces sooner = cheaper autotuning. The bar is
**K ≤ 25** for the six non-split-K backends and **K ≤ 75** for `bwd_s` (its config space is ~10×
larger). The kernels are fixed — this buys tune-time + correctness, not kernel speed.

## Folder layout

**`core/`** — the production path (never imports `aux`):

| file                  | role                                                                                                                                                                                                                                       |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `core/specs.py`       | GPU spec scalars. `host_spec()` **measures** gemm/cuda/mem_bw + queries sms/L2/arch on the running GPU; every sweep embeds its host's spec and the bundle carries them — no hard-coded table.                                              |
| `core/features.py`    | `featurize(configs, backend, H, B, spec)` → numeric matrix (one row per config): config knobs + shape + GPU scalars + derived geometry, the validated 21-column feature set.                                                               |
| `core/gen_data.py`    | **Generate input data** — sweep the autotuner on a GPU → `data/full_ranked_<gpu>.jsonl.gz`, and (by default) the `data/custom_lstm_<gpu>.jsonl.gz` plot baseline.                                                                          |
| `core/train.py`       | **Build the model** — fit one xgboost GBDT per backend (one shared `PARAMS` config) → `model_bundle.joblib` (in the cache); also hosts the shared `data_files`/`build_shapes`/`fit_on` primitives and prints an in-sample K readout.       |
| `core/prune.py`       | **Consume the model** — `make_tree_ranker(k)` returns the ranker you pass to `tune(..., config_ranker=...)` to prune the candidate set.                                                                                                    |
| `core/fingerprint.py` | Bundle identity shared by `train` (writes) and `prune` (reads): `BUNDLE_REL` (cache path) + `source_data_fingerprint()` (SHA-256 of the source + data that gates a rebuild). A leaf module, so it breaks the `prune`↔`train` import cycle. |
| `core/metrics.py`     | K metrics — `k_metric` (ranking depth) + `ExcessStats` (`Σ(K-1)²`, heavy-tail sensitive). Bundle-neutral; used by `train`'s in-sample K readout.                                                                                           |

**`aux/`** — visualization (imports from `core`, never the reverse):

| file          | role                                                                                                  |
| ------------- | ----------------------------------------------------------------------------------------------------- |
| `aux/plot.py` | **Plot** — throughput-vs-batch PNGs per GPU/direction; TileLang backends + dashed CustomLSTM overlay. |

All sweep/config data lives out-of-tree at `/datasets/tile_tree_data` (on the datasets drive) —
`read_data_dir()` reads from it for training, the fingerprint, and the plots. The in-tree `data/`
dir is a development fallback `read_data_dir()` uses when the datasets drive is empty. See
`data/README.md`.

## Environment

Everything runs inside the repo container (`myrtle/rnnt`, has numpy + xgboost). **Launch it
once** from the repo `training/` dir — this drops you into an interactive shell with a GPU
attached, the repo mounted at `/workspace/training`, and `caiman_asr_train` already importable:

```bash
# from the repo training/ dir
./scripts/docker/launch.sh $(mktemp -d) $(mktemp -d) $(mktemp -d)
```

The three `mktemp -d` args are throwaway mounts the launcher requires; this package ignores them.
**Every command below is run from inside that container, from `/workspace/training`.** Data
generation and training need a GPU; plotting is CPU-only.

## Running a healthy full sweep (operations)

The full multi-H sweep (`--hidden 512 768 1024 1536 --b-max 2048`) is millions of configs and
has a few sharp edges that have taken hosts offline. Read this before launching a full run.

- **Raise `vm.max_map_count` first — the #1 pitfall.** tilelang `dlopen`s every compiled kernel,
  and the mmap regions accumulate across shapes. A single split-K `bwd_s` shape is ~5–10k configs
  (it scales with H: ~5k at H=512, ~10k at H=768+), i.e. tens of thousands of mmaps — so the default
  limit of **65530** fills after roughly one big `bwd_s` shape and the process then **hangs** (~0%
  CPU, GPU idle, `MemoryError` / `failed to map segment from shared object` / `can't start new thread` — note it is **not** out of RAM). Fix before starting: `sudo sysctl -w vm.max_map_count=1048576`. It's a global limit checked at `mmap` time, so it applies **live to an
  already-running process** (no restart); it only relaxes a ceiling. Persist it in
  `/etc/sysctl.d/`. With no root, run under a watchdog that restarts on a genuine hang — a fresh
  process resets the map count and replays the bench cache (cache hits only re-elaborate TIR; they
  don't recompile/`dlopen`), so each restart makes forward progress.
- **Keep compile parallelism low.** `CAIMAN_LSTM_PAR_COMPILE_WORKERS` (default 16) sets the
  parallel-compile fan-out. Use **4–8** for the full sweep: 16 can OOM-deadlock the wide H=1536
  compile, and the storm of nvcc subprocesses / open file handles can exhaust host resources and
  take the whole box offline. It is a **safety knob, not a speed knob** — the sweep is bench-bound,
  and benching is sequential regardless.
- **Give the process plenty of RAM.** `bwd_s` holds thousands of compiled kernels resident per
  shape (tens of GB). If it's memory-starved (e.g. a low docker `--memory` cap) you get
  `MemoryError` + `failed to map segment`, and — worse — those load failures are **cached as
  permanent config-infeasibility**, silently dropping configs from `bwd_s` rankings. If that
  happens: clear the bench cache and re-run with more memory.
- **One bench process per GPU; for multi-GPU split by `--tune-batches`.** Bench timing must be the
  *only* GPU workload or the latencies are polluted (a co-resident job corrupts do_bench-style
  timing). To use N GPUs, split the **batch buckets** disjointly (interleave the progressive
  log-refine bucket list even/odd for ~50/50 balance), give each a distinct `--tag`, run one per
  GPU, and merge the tagged `full_ranked_*` files at the end. **Do not split by `--backends`** —
  `bwd_s` dominates the work, so a backend split badly imbalances the GPUs.
- **Don't pass `--n-repeat`.** The adaptive per-batch default must match the committed data's
  schedule, or the tree's training labels become inconsistent across GPUs. (There is no `--lean`
  flag; `lean_bench` is already the default timer.)
- **Detect hangs by CPU%/GPU%, not by logs.** tqdm bars (`Parallel Compiling`, `Benchmarking …`)
  use carriage returns, so `docker logs --since`/byte-growth checks false-diagnose a healthy run as
  hung. Use `docker stats --no-stream`: a real hang is **CPU < ~15% AND the dedicated GPU < ~5%**
  sustained (small-B split-K bench is CPU-bound/GPU-idle; large-B bench is GPU-bound/CPU-low — you
  need both signals). Also watch for **rogue co-resident processes**: ≥2 compute procs on a GPU you
  own means someone else landed on it → stop (timings are polluted). Read progress with
  `zcat file | grep -c '"record": "shape"'` — the `.jsonl.gz` is mid-write, so Python's `gzip`
  reader throws `EOFError`.
- **Expect a front-loaded, slow start.** The log-refine B order does B=1/2 first, where `n_repeat`
  is maxed (256) *and* the split-K config sets are largest — the opening bucket is alarmingly slow,
  then it accelerates hard into large B. Partial runs are still log-uniform and cache-resumable, so
  don't panic at the opening rate.

## Workflow

### 1. Generate input data for a new GPU

On the target GPU machine, sweep the autotuner to produce that GPU's full per-config ranking:

```bash
python -X utf8 -m caiman_asr_train.tilelstm.tree_model.core.gen_data
# -> data/full_ranked_<gpu>.jsonl.gz  (TileLang training data)
# -> data/custom_lstm_<gpu>.jsonl.gz  (CustomLSTM baseline for the plots)
```

By default the same run ALSO benches the CustomLSTM fused kernels (the `rnnt_ext` recurrent
scan, the dashed reference line the plots overlay) and writes `custom_lstm_<gpu>.jsonl.gz`.
Use `--skip-custom` for the TileLang sweep only, or `--custom-only` to (re)generate just the
baseline. For a multi-GPU host, see
[Running a healthy full sweep](#running-a-healthy-full-sweep-operations) — split by
`--tune-batches` (not `--backends`) and run one `--custom-only` pass for the baseline.

The sweep's parallel-compile fan-out defaults to 16 workers, overridable via
`CAIMAN_LSTM_PAR_COMPILE_WORKERS` — **lower it to 4–8 for the full sweep** (16 can OOM-deadlock
H=1536 / exhaust host handles; see the operations section above). The run is long but memoised on
disk, so reruns are cheap. The GPU is identified from
`torch.cuda.get_device_name`, and the header embeds the host's measured `host_spec()` — every
file is self-describing, so a new GPU needs no code edit anywhere. Point `--out-dir` at
`/datasets/tile_tree_data` (or generate locally, then move the finished sweeps there) — that
datasets-drive location is the production home training and the plots read via `read_data_dir()`.

### 2. Train the model

```bash
# needs a GPU; ~minutes
python -X utf8 -m caiman_asr_train.tilelstm.tree_model.core.train
```

Fits one `xgboost.XGBRegressor` per backend — all from the one shared `PARAMS` config
(384 trees × 31 leaves, lr 0.111), seven separately-fitted boosters — on every
`full_ranked_*.jsonl.gz` in the resolved read dir (`read_data_dir()`: `/datasets/tile_tree_data`
when populated, else in-tree `data/`), target = within-shape log-slowdown, features =
`featurize`'s 21 columns. The bundle is written to the tilelang cache root (`$TILELANG_CACHE_DIR` /
`~/.tilelang/cache`), never in-tree — a trained artifact built on demand; its layout is
`{fingerprint, booster_kind, xgboost_version, trained_h, trained_specs, boosters}`. Each backend's
model is stored as raw **UBJ** booster bytes (`booster.save_raw("ubj")`), not a pickled estimator
— UBJ is version-stable, so an xgboost upgrade can't break deserialization (column order is pinned
by the fingerprint, not a stored feature-name list; `trained_specs` carries each sweep's embedded
spec so the ranker featurizes a trained GPU exactly as training did). The `fingerprint` is a
sha256 of the featurization source + training data: `prune` rebuilds the bundle automatically when
it no longer matches (no manual version bump). `prune` also warns on an xgboost-version mismatch
(advisory only — UBJ is stable), and when asked to rank a GPU outside `trained_specs` or an H
outside `trained_h` (the ranking is then an extrapolation and may mis-prune the top-K).

After fitting, `train` prints an **in-sample** within-3% K `ExcessStats` table (per backend +
overall, with each backend's actual tree count) — a quick sanity readout of how tightly the
fitted models rank the data they trained on. It is optimistic (in-sample, no hold-out).

### 3. Consume it — prune the autotune space in production

Pass the ranker as the `config_ranker=` kwarg to `tune` (no global state):

```python
from caiman_asr_train.tilelstm.autotune.tuners import tune
from caiman_asr_train.tilelstm.tree_model.core.prune import make_tree_ranker

ranker = make_tree_ranker(k=50)   # loads the bundle + measures the running GPU once
kernel, ranked = tune(backend, H, device, tune_batch, lo, hi, config_ranker=ranker)
```

The ranker (`(configs, backend, H, tune_batch) -> configs`) featurizes each candidate with the
bundle's trained spec for the running GPU — or its **measured** `host_spec()` when the GPU is
outside the training set — predicts log-slowdown, and keeps the top-`k` — so the autotuner only
compiles/benches those. This is already wired into `autotune.select.select_fastest`, so
production autotuning uses it automatically.

### 4. Plot throughput graphs

Achieved-throughput-vs-batch PNGs (one figure per GPU per direction): the TileLang backends'
best-tuned config at each batch, with the CustomLSTM fused kernel overlaid dashed. Reads the
sweeps from `read_data_dir()` by default (`/datasets/tile_tree_data` when populated, else the
committed in-tree `data/` — both the `full_ranked_*` and `custom_lstm_*` files `gen_data`
writes), so it needs no GPU — only `matplotlib`:

```bash
# from the repo training/ dir (CPU-only, in the container)
python -X utf8 -m caiman_asr_train.tilelstm.tree_model.aux.plot --out-dir /tmp/plots
#   --data-dir DIR    read sweeps from elsewhere (default: read_data_dir() -- /datasets/tile_tree_data or data/)
#   --gpu 4090 a100   restrict to some GPUs (substring match on device name / filename)
#   --direction fwd   forward only (default: both fwd + bwd)
#   --best-only       collapse each panel to the fastest backend per B (writes *_best.png)
```

Writes `lstm_kernel_<dir>_flops_<device>[...].png` per GPU/direction into `--out-dir`. To
plot a bespoke set, point `--data-dir` at any folder of `full_ranked_*` / `custom_lstm_*`
files (e.g. a scratch dir you copied a subset into).

## Grouped-CV results

Quality is the tie-aware **K** metric (depth of the first within-3% config in the ranking) on
whole-shape-cell grouped cross-validation. `kfold_target` (bench some of the target GPU's cells,
predict the rest) is the primary parity metric; `logo` (leave-one-GPU-out) is the harder
secondary. `over_bar` = held-out cells with K above the bar (25, or 75 for `bwd_s`); `rms` =
sqrt(mean (K−1)²), the continuous tail-sensitive metric; both mean over seeds 0–4. Every
backend is 384 trees × 31 leaves (23,424 params) — the one shared config, selected as the
only evaluated shared config whose worst-seed K tails stay inside the production margin on
all seven backends (bwd_s included, its bar having zero margin):

| backend |     cells | kfold over_bar (xgb / HGBR) | kfold rms (xgb / HGBR) | logo over_bar (xgb / HGBR) |
| ------- | --------: | :-------------------------: | :--------------------: | :------------------------: |
| fwd_c   |       937 |           0.2 / 0           |     1.131 / 1.149      |         12.8 / 4.4         |
| fwd_ct  |       636 |            0 / 0            |     1.300 / 1.464      |         4.4 / 4.8          |
| bwd_c   |     1,113 |           0 / 0.4           |     2.073 / 2.058      |        96.2 / 89.8         |
| fwd_p   |       977 |           0.2 / 0           |     1.585 / 1.599      |        49.6 / 49.2         |
| fwd_pt  |       976 |          2.2 / 0.6          |     2.044 / 1.920      |       138.2 / 111.6        |
| bwd_p   |       974 |          0.4 / 2.2          |     2.049 / 2.523      |       121.6 / 117.6        |
| bwd_s   |       244 |           0 / 1.0           |     5.660 / 8.356      |        93.8 / 64.2         |
| **all** | **5,857** |        **3.0 / 4.2**        |           —            |             —              |

`kfold_target` over-bar total is 3.0 cells vs the baseline's 4.2 (mean over 5 seeds) — all but
~3 of the 5,857 cells surface a within-3% config inside the K bar. Worst-seed K_max stays ≤ 35
on the six non-split-K backends (production keeps top-50). `bwd_s` has zero margin at bar 75
and meets the strict rule: **over_bar 0 across all 5 seeds, worst-seed K_max 54 vs the
baseline's 150** — the shared config IS the config that won `bwd_s` in the per-backend sweep
(the tail bar is what forces the low learning rate everywhere). The known trades vs baseline,
accepted for the single-config simplification: seed-mean rms is a shade above baseline on
`bwd_c` (+0.7%) and `fwd_pt` (+6.5% — its over_bar 2.2 vs 0.6, but worst K_max 32, so every
over-bar cell still sits inside production's top-50 cut). On `logo` (the extrapolation regime,
accepted as the secondary) it trades on `fwd_c`/`fwd_pt`/`bwd_s` (mildly on `bwd_c`/`bwd_p`),
is at par on `fwd_p`, and wins on `fwd_ct`.

## Notes

- **Model size:** 163,968 parameters / 2,688 trees / ~2.55 MiB bundle (7 × 384 trees × 31
  leaves; one shared config) — a 2.1× parameter and 3.4× byte reduction vs the previous sklearn
  HGBR baseline (~350k parameters / 2,800 trees / ~8.66 MiB joblib-gz).
- **Serialization:** the bundle stores each backend as raw UBJ booster bytes (version-stable), so
  the `xgboost==3.3.0` pin is a soft floor (a version mismatch only warns).
- **Features:** the production set is `featurize`'s 21 columns: config knobs +
  `k_splits/red_M/res_kc` (bwd_s split-K) + `gemm_flops/arch_sm` (cross-GPU) + the minimal-4
  derived geometry (`wave_quant, smem_bytes, reread_bytes, padded_flops`). The derived geometry
  (evaluated at the target GPU's spec) is what enables cross-GPU generalisation.
- **Regime matters:** the tree fits/interpolates far better once it has some target-GPU data, but
  it can't extrapolate a feature beyond its training range — which is why `bwd_s` (whose optimum
  scales with SM count) is the weakest backend on a GPU with no measured data, and is fully
  solved once any target-GPU data is available.
