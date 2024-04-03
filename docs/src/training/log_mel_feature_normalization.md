# Log-mel feature normalization

We normalize the acoustic log mel features based on the global mean and variance recorded over the training dataset.

## Record dataset stats

The script [`generate_mel_stats.py`](https://github.com/MyrtleSoftware/caiman-asr/blob/main/training/caiman_asr_train/utils/generate_mel_stats.py) computes these statistics
and stores them in `/datasets/stats/<dataset_name+window_size>` as PyTorch tensors. For example usage see:

* `scripts/preprocess_librispeech.sh`
* `scripts/preprocess_webdataset.sh`

## Training stability

Empirically, it was found that normalizing the input activations with dataset global mean and variance makes the early stage of training unstable.
As such, the default behaviour is to move between two modes of normalization on a schedule during training. This is handled by the `MelFeatNormalizer` class and explained in the docstring below:

```
{{#include ../../../training/caiman_asr_train/data/dali/mel_normalization.py:MelFeatNormalizer_in_mdbook}}
```

## Validation

When running validation, the dataset global mean and variance are always used for normalization regardless of how far through the schedule the model is.

### Backwards compatibility

Prior to v1.9.0, the per-utterance stats were used for normalization during training (and then streaming normalization was used during inference).
To evaluate a model trained on <=v1.8.0, use the `--norm_over_utterance` flag to the `val.sh` script.
