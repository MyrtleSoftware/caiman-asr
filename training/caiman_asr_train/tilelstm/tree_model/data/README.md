# Training data — full per-config kernel-bench rankings

One `full_ranked_<gpu>.jsonl.gz` per GPU (gzipped JSONL). Each is the **complete** measured
ranking the tree trains on and is scored against: every feasible autotune config for every
`(backend, H, tune_batch)` shape, with its tune-batch latency. The one exception is
`full_ranked_ci.jsonl.gz`: a tiny subset of a real 4090 sweep that lets cold CI checkouts
exercise the load/train path without mounting the datasets drive.

Format (gzipped; one JSON object per line — `zcat` to read):

```
{"record":"header","device":"<torch device name>", ...}
{"record":"shape","backend":"fwd_c","H":1024,"tune_batch":128,
 "ranked":[{"config":{...get_tir kwargs...},"latency_ms":0.123}, ...]}   # ascending latency
```

`backend` is a `Kernel` code (see `../../autotune/kernel.py`): `fwd_c`/`fwd_p`/`fwd_ct`/`fwd_pt`
(cooperative / persistent forward, the `_t` variants gate-saving) and `bwd_c`/`bwd_s`/`bwd_p`
(cooperative / split-K / persistent backward).

Each file carries the `GpuSpec` of the host it was benched on in its header (`"spec"`,
embedded by `gen_data`) — the files are fully self-describing; there is no spec table in
the code. Add a new GPU's sweep to `/datasets/tile_tree_data` as `full_ranked_<name>.jsonl.gz`
and `build_shapes` picks it up automatically.

## Where the data lives (read vs write)

The finished sweeps are ~50 MB and belong **out-of-tree**, on the datasets drive:
`/datasets/tile_tree_data` inside the container. `read_data_dir()` (the package
`__init__`) resolves the READ location for training, the bundle fingerprint, and the
plots' default:

- **`/datasets/tile_tree_data`** — the production location, used whenever it contains at
  least one `full_ranked_*.jsonl.gz`.
- **this folder** — the development/CI fallback when the external dir is absent or empty.
  It contains only the tiny CI sweep, which is enough for smoke tests but is not a
  representative model-training dataset.

`gen_data`'s default **write** target stays this folder: freshly generated sweeps land
here (or wherever `--out-dir` points); move the finished files to
`/datasets/tile_tree_data` once they are final. Don't leave the same GPU's sweep in both
places at different versions — whichever directory `read_data_dir()` resolves to wins
wholesale (the two are never merged).

| file                         | GPU                             | device header                                  |
| ---------------------------- | ------------------------------- | ---------------------------------------------- |
| `full_ranked_ci.jsonl.gz`    | CI subset of RTX 4090 (sm_89)   | `NVIDIA GeForce RTX 4090`                      |
| `full_ranked_4090.jsonl.gz`  | RTX 4090 (sm_89)                | `NVIDIA GeForce RTX 4090`                      |
| `full_ranked_a100.jsonl.gz`  | A100-SXM4-80GB (sm_80)          | `NVIDIA A100-SXM4-80GB`                        |
| `full_ranked_sm120.jsonl.gz` | RTX PRO 6000 Blackwell (sm_120) | `NVIDIA RTX PRO 6000 Blackwell Server Edition` |

**Regenerate** for any GPU by running
`python -m caiman_asr_train.tilelstm.tree_model.core.gen_data` on that machine — see the
top-level README.
