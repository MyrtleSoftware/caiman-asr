# Validation

## Validation Command

### Quick Start

To run validation, execute:

```bash
./scripts/val.sh
```

By default, a checkpoint saved at `/results/RNN-T_best_checkpoint.pt`, with the `testing-1023sp_run.yaml` model config, is evaluated on the `/datasets/LibriSpeech/librispeech-dev-clean.json` manifest.

### Arguments

Customise validation by specifying the `--checkpoint`, `--model_config`, and `--val_manifests` arguments to adjust the model checkpoint, model YAML configuration, and validation manifest file(s), respectively.

Predictions are saved as described [here](./saving_predictions.md).

See [`args/val.py`](https://github.com/MyrtleSoftware/caiman-asr/blob/main/training/caiman_asr_train/args/val.py) and
[`args/shared.py`](https://github.com/MyrtleSoftware/caiman-asr/blob/main/training/caiman_asr_train/args/shared.py)
for the complete set of arguments and their respective docstrings.

### Further Detail

- All references and hypotheses are normalized with the Whisper normalizer before calculating WERs, as described in the [WER calculation docs](./wer_calculation.md). To switch off normalization, modify the respective config file entry to read `standardize_wer: false`.
- During validation the [state resets technique](./state_resets.md) is applied by default in order to increase the model's accuracy.
- The model's accuracy can be improved by using [beam search](./beam_decoder.md) and an [n-gram language model](./ngram_lm.md).
- Validating on long utterances is calibrated to not run out of memory on a single 11 GB GPU.
  If a smaller GPU is used, or utterances are longer than 2 hours, refer to this [document](automatic_batch_size_reduction.md).
- By default during validation, all input audios are padded with 0.96s of silence at the end
  so that the model has time to output the final tokens.
  You can change this using the `--val_final_padding_secs` flag.

## Next Step

See the [hardware export documentation](./export_inference_checkpoint.md) for instructions on exporting a hardware checkpoint for inference on an accelerator.
