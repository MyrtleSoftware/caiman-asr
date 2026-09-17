# Conditional Decoding

Having trained a conditional model by following the instructions in the [conditional training docs](conditional_training.md),
you can select the output style at validation/inference time by providing a prefix.

## What it does

- Initialize the prediction network with control tokens before decoding.
- Supported prefixes:
  - `pnc` → use `<pnc>` (punctuation + casing)
  - `nopnc` → use `<nopnc>` (lowercase, no punctuation)
  - `lang_XX` → use `<lang_XX>` (language code XX, e.g. `lang_en`, `lang_fr`, etc.)
- Works with both greedy and beam decoders.

```admonish
Default: If `--pnc_prefix` and `--val_lang_prefix` are not set, no
prefix is used and decoding behaves exactly as before (backwards compatible
with existing models).
```

## Usage

### pnc

```bash
./scripts/val.sh --pnc_prefix=pnc
```

### nopnc

```bash
./scripts/val.sh --pnc_prefix=nopnc
```

### Language prefix

```bash
./scripts/val.sh --pnc_prefix=pnc --val_lang_prefix=lang_fr
```

```admonish
The `--(lang|pnc)_prefix` options are available in both `val.sh` and `train.sh`
(for on-the-fly validation during training).
```

## Notes

- Ensure your model config `.yaml` includes the control tokens in `user_tokens`, e.g. `<pnc>` and `<nopnc>`.
  - And that your tokenizer includes these user symbols and maps them to valid token IDs.
- The model must have been trained with [conditional training](conditional_training.md); otherwise decoding with a prefix will fail.
