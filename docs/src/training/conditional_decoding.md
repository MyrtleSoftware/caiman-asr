# Conditional Decoding

Having trained a conditional model by following the instructions in the [conditional training docs](conditional_training.md),
you can select the output style at validation/inference time by providing a prefix.

## What it does

- Initialize the prediction network with a control token before decoding.
- Supported prefixes:
  - `pnc` → use `<pnc>` (punctuation + casing)
  - `nopnc` → use `<nopnc>` (lowercase, no punctuation)
- Works with both greedy and beam decoders.

```admonish
Default: If `--prefix` is not set, no prefix is used and decoding behaves exactly as before (backwards compatible with existing models).
```

## Usage

### pnc

```bash
./scripts/val.sh --prefix=pnc
```

### nopnc

```bash
./scripts/val.sh --prefix=nopnc
```

```admonish
The `--prefix` option is available in both `val.sh` and `train.sh` (for on-the-fly validation during training).
```

## Notes

- Ensure your model config `.yaml` includes the control tokens in `user_symbols`, e.g. `<pnc>` and `<nopnc>`.
  - And that your tokenizer includes these user symbols and maps them to valid token IDs.
- The model must have been trained with [conditional training](conditional_training.md); otherwise decoding with a prefix will fail.
