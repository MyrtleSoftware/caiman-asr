# Training Datasets

## 50k hour dataset <a name="50k_hrs"></a>

Myrtle.ai's 50k hrs of training data is a mixture of the following open-source datasets:

* LibriSpeech-960h
* Common Voice Corpus 10.0 (version `cv-corpus-10.0-2022-07-04`)
* Multilingual LibriSpeech (MLS)
* Peoples' Speech: filtered internally to take highest quality ~10k hrs out of 30k hrs total

This data has a `maximum_duration` of 20s and a mean length of 14.67s.

If your dataset is organized in the json format, you can use [this script](https://github.com/MyrtleSoftware/caiman-asr/blob/main/training/caiman_asr_train/data/mean_json_duration.py) to calculate its mean duration.

## 10k hour dataset <a name="10k_hrs"></a>

Myrtle.ai's 10k hrs of training data is a mixture of the following open-source datasets:

* LibriSpeech-960h
* Common Voice
* 961 hours from MLS
* Peoples' Speech: A ~6000 hour subset

This data has a `maximum_duration` of 20s and a mean length of 14.02s.

The 10k hour dataset is a subset of the 50k hour dataset above but experiments indicate that models trained on it give better results on Earnings21 than those training on the 50k hour dataset.
