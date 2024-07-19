# Performance

The solution has various configurations that trade off accuracy and performance. In this page:

- **Realtime streams (RTS)** is the number of concurrent streams that can be serviced by a single accelerator using default settings
- **p99 latency** is the 99th-percentile latency to process a single 60 ms audio frame and return any predictions. Note that latency increases with the number of concurrent streams.
- **WER** is the Word Error Rate, a measure of the accuracy of the model. Lower is better.
- **Huggingface Leaderboard WER** is the WER of the model on the [Huggingface Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard). WER is averaged across 8 test datasets.

The WERs in the following section are for models trained on 13k hours of open-source data described at the bottom of this page.

## Beam search with n-gram LM

The solution supports decoding with a [beam search](./training/beam_decoder.md) (default beam width=4) with an [n-gram language model](./training/ngram_lm.md) for improved accuracy. The solution supports greedy decoding for higher throughput.

| Model   | Parameters | Decoding          | Realtime streams (RTS) | p99 latency at max RTS  | Huggingface Leaderboard WER  |
|---------|------------|-------------------|------------------------|-------------------------|------------------------------|
| `base`  | 85M        | `greedy`          | 2000                   |   25 ms                 | 13.50%                       |
| `base`  | 85M        | `beam`, `width=4` | 1300                   |   80 ms                 | 12.53%                         |
| `large` | 196M       | `greedy`          | 800                    |   25 ms                 | 12.44%                       |
| `large` | 196M       | `beam`, `width=4` | 500                    |   40 ms                 | 11.59%                       |

## State resets

State resets is a technique that improves the accuracy on long utterances (over 60s) by resetting the model's hidden state after a fixed duration. This reduces the number of real-time streams that can be supported by around 25%:

| Model   | Parameters | Decoding          | Realtime streams (RTS) | p99 latency at max RTS  | Huggingface Leaderboard WER  |
|---------|------------|-------------------|------------------------|-------------------------|------------------------------|
| `base`  | 85M        | `greedy`          | 1600                   |   45 ms                 | 13.47%                       |
| `base`  | 85M        | `beam`, `width=4` | 1200                   |   50 ms                 | 12.53%                       |
| `large` | 196M       | `greedy`          | 650                    |   55 ms                 | 12.34%                       |
| `large` | 196M       | `beam`, `width=4` | 400                    |   60 ms                 | 11.55%                       |

Note that most of the data in the Huggingface leaderboard is less than 60s long so the impact of state resets is not reflected in the leaderboard WER.

## 13k hour dataset <a name="13k_hrs"></a>

The models above were trained on 13k hrs of open-source training data consisting of:

- LibriSpeech-960h
- Common Voice 17.0
- 961 hours from MLS
- Peoples' Speech: A ~9389 hour subset filtered for transcription quality
- 155 hrs of AMI

This data has a `maximum_duration` of 20s.
