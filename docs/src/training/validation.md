# Validation

```admonish
As of v1.8, the API of `scripts/val.sh` has changed. This script now takes command line arguments instead of environment variables (`--num_gpus=8` instead of `NUM_GPUS=8`).
For backwards compatibility, the script `scripts/legacy/val.sh` still uses the former API but it doesn't support features introduced after v1.7.1, and will be removed in a future release.
```

## Validation Command

### Quick Start

To run validation, execute:

```bash
./scripts/val.sh
```

By default, a checkpoint saved at `/results/RNN-T_best_checkpoint.pt`, with the `testing-1023sp_run.yaml` model config, is evaluated on the `/datasets/LibriSpeech/librispeech-dev-clean-wav.json` manifest.

### Arguments

Customise validation by specifying the `--checkpoint`, `--model_config`, and `--val_manifests` arguments to adjust the model checkpoint, model YAML configuration, and validation manifest file(s), respectively.

To save the predictions, pass `--dump_preds` as described [here](./saving_predictions.md).

See [`args/val.py`](https://github.com/MyrtleSoftware/caiman-asr/blob/main/training/caiman_asr_train/args/val.py) and
[`args/shared.py`](https://github.com/MyrtleSoftware/caiman-asr/blob/main/training/caiman_asr_train/args/shared.py)
for the complete set of arguments and their respective docstrings.

### Further Detail

- All references and hypotheses are normalized with the Whisper normalizer before calculating WERs, as described in the [WER calculation docs](./wer_calculation.md). To switch off normalization, modify the respective config file entry to read `standardize_wer: false`.
- During validation the [state resets technique](./state_resets.md) is applied by default in order to increase the model's accuracy.
- Validating on long utterances is calibrated to not run out of memory on a single 11 GB GPU.
If a smaller GPU is used, or utterances are longer than 2 hours, refer to this [document](automatic_batch_size_reduction.md).

## Next Step

See the [hardware export documentation](./export_inference_checkpoint.md) for instructions on exporting a hardware checkpoint for inference on an accelerator.
