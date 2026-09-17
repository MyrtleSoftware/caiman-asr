# Kernel-agnostic TileLang autotuning

Generic machinery for tuning any TileLang kernel. The TileLang LSTM
(`caiman_asr_train/tilelstm/`) is the reference client: every contract below
has a worked example there. The TileLang transducer
(`caiman_asr_train/tile_transducer/autotune/`) is the second client. Nothing in this
package may import from either — the dependency points one way only.

| Module           | What it gives you                                                                                                                                 |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `device.py`      | GPU stats: `DeviceLimits`, `device_limits(device)`, `resident_blocks(...)` (occupancy), `device_name(device)` (stable cache-key id)               |
| `bench.py`       | `lean_bench` (cold-cache CUDA-event median timing), `n_repeat_for`, `tl_dtype` (torch dtype -> tilelang string), `tl_log`                         |
| `feasibility.py` | Tile-legality filters: `mnk_atom_ok` / `wmma_ok` (WMMA atom + warp partition), `threads_are_warp`, `smem_fits`, `grid_resident`, `regs_per_block` |
| `candidates.py`  | `config_product(base, **axes)` cross-products and `divisors(n)` for building candidate tile sets                                                  |
| `sweep.py`       | `autotune_select` — the tune driver with its per-config bench cache                                                                               |
| `cache_io.py`    | Process-safe atomic `cache_read` / `cache_write` under the tilelang cache root                                                                    |

## How `autotune_select` works

For one backend at one shape, `autotune_select`:

1. Elaborates every config via `factory.get_tir(**cfg)` and keys each by its
   structural TIR hash (name- and free-var-invariant, so warm re-tunes hit
   across processes).
2. Partitions into bench-cache hits and the to-bench delta — an unchanged
   config set re-benches nothing; adding one config benches one config.
3. Parallel-compiles the delta with `tilelang.jit.par_compile`. A compile or
   launch failure is expected infeasibility: it is cached as
   `latency_ms: null` and never retried.
4. Times each survivor with `lean_bench` and validates its numerics — a wrong
   result raises, crashing the tune loudly by design.
5. Ranks ascending by latency, then compiles, re-validates, and `finalize`s the
   winner. Returns `(kernel, ranked)`, or `None` when nothing launched (the
   caller skips that backend).

## Wiring a new kernel

`tilelstm/autotune/tuners.py` is the worked example of all of the below.

1. **Factory** — a `tilelang.jit`-decorated kernel factory. The sweep calls
   `factory.get_tir(**cfg)` / `factory.compile(**cfg)` and reads `out_idx`,
   `execution_backend`, `target`, `target_host`, `verbose`, `pass_configs`,
   and `compile_flags` off it.
2. **Candidates** — build the raw cross-product with
   `candidates.config_product` (and `divisors` for K-chunk axes). Configs must
   contain only primitives (int/float/str/bool/None); pass dtypes through
   `bench.tl_dtype`. See `tilelstm/autotune/config_candidates.py`.
3. **Feasibility filters** — compose, in roughly this order:
   - `threads_are_warp`, `mnk_atom_ok`, `wmma_ok` (pass the gemm's real column
     width via `n_block_mult`);
   - your kernel's own shape constraints (cf. `_tile_ok` /
     `_epilogue_reduction_ok` in `tilelstm/autotune/configs.py`);
   - a per-config SMEM-bytes estimator for your kernel's buffer layout
     (cf. `tilelstm/autotune/configs_smem.py`) fed to `smem_fits`;
   - residency: `device_limits` + `grid_resident` / `resident_blocks`, with a
     calibrated per-thread register floor for `regs_per_block` (the LSTM uses
     56 — an optimistic floor, so the pre-prune never drops a config that
     would actually launch).
4. **Bench plumbing** — allocate tune-shape buffers once, then hand
   `autotune_select` three callbacks: `make_thunk(kernel, cfg)` (zero-arg
   bench closure), `validate(kernel, cfg)` (run on fresh outputs and
   `assert_close` against a reference; `AssertionError` must propagate), and
   `finalize(kernel, cfg)` (adapt to your uniform call contract). Cf.
   `buffers.py` / `wrappers.py` / `tuners.py` in `tilelstm/autotune/`.
5. **Cache key** — put only bench *conditions* in `cache_key`
   (`device_name(device)`, tune batch, bench params). Everything about the
   kernel body and shape is already captured by each config's TIR hash.
6. **Selection policy** (optional) — an `@cache`'d loop that tunes several
   backends and keeps the fastest, cf. `tilelstm/autotune/select.py`.

## Caching

Everything lives under the tilelang cache root (`$TILELANG_CACHE_DIR`, default
`~/.tilelang/cache`). Bench entries are one JSON file per config:
`caiman_autotune_bench/<sha256(cache_key)>/<tir_hash>.json` (TIR hashes make it
safe to share across kernels). Env vars:

- `TILELANG_CACHE_DIR` — move the cache root (tests use a `tmp_path`).
- `TILELANG_DISABLE_CACHE` — tilelang's global disable, honored here too.
- `CAIMAN_AUTOTUNE_DISABLE_CACHE` — disables only these caches.

## Testing your kernel's tuner

- Cap the feasible grid in a fixture the way
  `tests/conftest.py::cap_autotune_configs` does (swap the registry entry with
  `dataclasses.replace` and clear the `@cache` memos), so CI compiles a
  handful of configs instead of the full grid.
- CUDA-free sweep tests: fake the factory and monkeypatch
  `sweep._par_compile` / `sweep.lean_bench` / `sweep._tir_hash` — see
  `tests/autotune/test_sweep.py`.
- Isolate the disk cache with `TILELANG_CACHE_DIR=<tmp>`.
- Run in docker from the repo root:

```bash
cd training
TTY=false COMMAND="python -X utf8 -m pytest tests/autotune" \
    ./scripts/docker/launch.sh "$(mktemp -d)" "$(mktemp -d)" "$(mktemp -d)"
```

## Known seams

- `autotune_select`'s `H` / `tune_batch` parameters are log context named
  after the LSTM's shape axes; pass whatever identifies your shape.
- `sweep._CONST_CFG_KEYS` / `sweep._CFG_ABBREV` are log-formatting defaults
  tuned to the LSTM's config keys; unknown keys pass through unabbreviated.
