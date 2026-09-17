# Config & checkpoint hierarchy

This note describes how the YAML **config**, the in-memory model **config**, the
training **checkpoint**, and the **hardware checkpoint** relate to one another,
and the functions that convert between them. It is aimed at developers changing
any of these layers: keep the flow below intact so the downstream Myrtle
inference server keeps loading our checkpoints.

File paths below are relative to the `training/` directory; the `rnnt/`,
`core/`, and `export/` packages live under `caiman_asr_train/`.

## The flow

```
                 configs/*.yaml  (training config, superset of everything)
                        │
          config.load() │  rnnt/config.py
                        ▼
                  cfg: dict ────────────────────────────────────────┐
                        │                                           │
          config.rnnt() │  (validate_and_fill against RNNT kwargs)  │ inference_only_config()
                        ▼                                           │  export/hardware_ckpt.py
              rnnt kwargs: dict                                     ▼
                        │                          RNNTInferenceConfigSchema  core/export_schema.py
       rnnt/model.RNNT  │  (kwargs -> ConfigRNNT)  (drops training-only fields, extra="forbid")
                        ▼                                           │
                  ConfigRNNT  core/model.py                         │  .model_dump()
                        │                                           ▼
        core/model.RNNT │  (builds nn.Modules)              inference config: dict
                        ▼                                           │
                  RNNT (nn.Module)                                  │
                        │                                           │
   Checkpointer.save()  │  export/checkpointer.py                   │
                        ▼                                           │
          training checkpoint  (*.pt)                               │
   { state_dict, ema_state_dict, optimizer, tokenizer_kw,           │
     raw_prefix_token_map, logmel_norm_weight, ... }                │
                        │                                           │
   create_hardware_ckpt │  export/hardware_ckpt.py  ◄───────────────┘
                        ▼
          hardware checkpoint  (*.hw.pt)
   { state_dict (= ema_state_dict), rnnt_config (inference config),
     melmeans, melvars, sentpiece_model, ngram, version,
     raw_prefix_token_map, ... }
                        │
   load_hw_checkpoint() │  core/load_hw.py   (SW / inference-side entry point)
                        ▼
          Checkpoint  { model: RNNT, data_config, sentpiece_model_binary,
                        ngram, mel_stats, mau_program, version }
```

## The artifacts

| Artifact                            | Where                                                | What it is                                                                                                                                                                                             |
| ----------------------------------- | ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Training config**                 | `configs/*.yaml`                                     | Human-authored superset of all training and inference settings.                                                                                                                                        |
| **`cfg` dict**                      | `config.load()` (`rnnt/config.py`)                   | The YAML parsed into a dict (tokenizer labels filled in).                                                                                                                                              |
| **`ConfigRNNT`**                    | `core/model.py`                                      | The canonical, validated dataclass the model is built from. The single source of truth for model shape.                                                                                                |
| **`RNNT` (base)**                   | `core/model.py`                                      | Pure `nn.Module`. Built from a `ConfigRNNT`. Kept dependency-light (see `caiman_asr_train/core/README.md`).                                                                                            |
| **`RNNT` (training)**               | `rnnt/model.py`                                      | Subclass adding the training/loss machinery (apex joint, lr factors, `forward`). Takes **kwargs**, assembles a `ConfigRNNT`, and calls the base `__init__`.                                            |
| **Training checkpoint** (`*.pt`)    | `Checkpointer.save()` (`export/checkpointer.py`)     | `state_dict` + `ema_state_dict` + optimizer + training metadata. Used to resume training and as the source for hardware export.                                                                        |
| **Hardware checkpoint** (`*.hw.pt`) | `create_hardware_ckpt()` (`export/hardware_ckpt.py`) | Self-contained inference artifact: EMA weights, the _inference_ config, mel stats, the sentencepiece model bytes, an optional n-gram, and a `version`. This is the contract with the inference server. |
| **`Checkpoint`**                    | `core/load_hw.py`                                    | A hardware checkpoint read back into Python: a ready-to-run `RNNT` plus the data needed around it.                                                                                                     |

## Two different "schemas"

These are easy to confuse — they validate different things:

- **Config schema** — `core/export_schema.py` (`RNNTInferenceConfigSchema` and
  friends). A pydantic model that ingests the YAML config and **drops every
  training-only field** (listed per-class in `allow_ignore`) while forbidding
  unknown ones (`extra = "forbid"`). `inference_only_config()` runs this and
  `model_dump()`s the result into the hardware checkpoint's `rnnt_config`. If
  you add a YAML field, decide whether the inference server needs it: if not,
  add it to the relevant `allow_ignore`; if so, update the server too.
  `tests/export/test_hardware_ckpt.py` guards this contract against drift.
- **State-dict / tensor schema** — `export/model_schema/` (`base.json`,
  `large.json`, checked by `check_model_schema()`). Validates that the model's
  **weight tensor shapes** match a supported FPGA model variant. A checkpoint
  that doesn't match raises `CheckpointNotSupportedError` and is skipped.

## Invariants for future changes

- `ConfigRNNT` is the one place that defines model shape. Both `RNNT`
  constructors funnel through it; don't add a parallel path that builds modules
  from raw kwargs.
- `prefix_token_map` is **not** part of the inference config schema (it lives in
  `ModelSchema.allow_ignore`). It is instead persisted at the top level of the
  checkpoint as `raw_prefix_token_map`, so `load_hw.py` reads it from there and
  feeds it back into `ConfigRNNT`. Keep these two ends in sync.
- The hardware checkpoint's `state_dict` is the **EMA** weights
  (`traincp["ema_state_dict"]`), not the raw `state_dict`.
- `mau_program` is the one `Checkpoint` field this repo never writes.
  `create_hardware_ckpt()` doesn't emit it; the mau model compiler injects it
  when it turns a hardware checkpoint into a CAIMAN-ASR checkpoint (see
  `docs/src/inference/compiling_weights.md`). `load_hw.py` reads it back
  opaquely, so it is `None` for a plain hardware checkpoint.
- `load_hw_checkpoint` always reconstructs a `TorchConfig()`-backed `RNNT`.
- Bump `version` in `create_hardware_ckpt()` when the hardware checkpoint layout
  changes in a way the inference server must be aware of.
