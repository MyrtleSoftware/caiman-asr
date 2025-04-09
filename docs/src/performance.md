# Performance

The solution has various configurations that trade off accuracy and performance. In this page:

- **Realtime streams (RTS)** is the number of concurrent streams that can be serviced by a single accelerator using default settings.
- **Compute latency 99th-percentile (CL99)** is the 99th-percentile compute latency, which measures how long it takes for a model to make a prediction for one audio frame. Note that CL99 increases with the number of concurrent streams.
- **User-perceived latency (UPL)** is the time difference between when the user finishes saying a word and when it is returned as a transcript by the system.
- **WER** is the Word Error Rate, a measure of the accuracy of the model. Lower is better.
- **HF Leaderboard WER** is the WER of the model on the [Huggingface Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard). WER is averaged across 8 test datasets.

The WERs in the following section are for models trained on 44k hours of mostly open-source data described at the bottom of this page.

The UPL were computed by streaming librispeech-dev-clean audio live to an FPGA backend server on-site. Please refer to this [document](./inference/user_perceived_latency.md) for more details on latencies.

## Without state resets

<!-- These use a language model for beam because it doesn't slow down inference-->

The solution supports decoding with a [beam search](./training/beam_decoder.md) (default beam width=4) with an [n-gram language model](./training/ngram_lm.md) for improved accuracy. The solution supports greedy decoding for higher throughput.

| Model   | Parameters | Decoding          | RTS  | CL99 at max RTS | median UPL | HF Leaderboard WER |
| ------- | ---------- | ----------------- | ---- | --------------- | ---------- | ------------------ |
| `base`  | 85M        | `greedy`          | 2000 | 25 ms           | 147 ms     | 11.04%             |
| `base`  | 85M        | `beam`, `width=4` | 1300 | 80 ms           | -          | 9.79%              |
| `large` | 196M       | `greedy`          | 800  | 25 ms           | -          | 9.19%              |
| `large` | 196M       | `beam`, `width=4` | 500  | 40 ms           | 158 ms     | 8.42%              |

## State resets

<!-- These use a language model for beam because it doesn't slow down inference-->

State resets is a technique that improves the accuracy on long utterances (over 60s) by resetting the model's hidden state after a fixed duration. This reduces the number of real-time streams that can be supported by around 25%:

| Model   | Parameters | Decoding          | RTS  | CL99 at max RTS | median UPL | HF Leaderboard WER |
| ------- | ---------- | ----------------- | ---- | --------------- | ---------- | ------------------ |
| `base`  | 85M        | `greedy`          | 1600 | 45 ms           | 147 ms     | 11.06%             |
| `base`  | 85M        | `beam`, `width=4` | 1200 | 50 ms           | -          | 9.69%              |
| `large` | 196M       | `greedy`          | 650  | 55 ms           | -          | 9.04%              |
| `large` | 196M       | `beam`, `width=4` | 400  | 60 ms           | 158 ms     | 7.98%              |

Note that most of the data in the Huggingface leaderboard is less than 60s long so the impact of state resets is not reflected in the leaderboard WER.

Since the UPL numbers were computed from librispeech-dev-clean, the effect of state resets is not reflected in measured latencies.

## WER test set breakdown

The WER breakdown across test sets is shown for the 'highest-throughput' and 'most-accurate' configurations in the table below:

<!-- base: meropi_2. greedy, sr=on -->

<!-- large: mirakor. beam=4, resume mirach on open1.6, ckpt avg -->

| Model Configuration | AVERAGE | AMI    | E22 (segmented) | Gigaspeech | LS test clean | LS test other | SPGISpeech | TED-LIUM | VoxPopuli |
| ------------------- | ------- | ------ | --------------- | ---------- | ------------- | ------------- | ---------- | -------- | --------- |
| base (greedy)       | 11.06%  | 15.37% | 18.20%          | 14.91%     | 4.08%         | 9.70%         | 6.27%      | 7.82%    | 12.15%    |
| large (beam=4)      | 7.98%   | 12.09% | 13.46%          | 11.20%     | 2.72%         | 6.68%         | 4.06%      | 4.85%    | 8.74%     |

## 44k hour dataset

The models above were trained on 44k hrs of mostly open-source training data consisting of:

- YODAS: 19k hour subset of YODAS manual English subset filtered for transcription quality
- Peoples' Speech: A ~9389 hour subset filtered for transcription quality
- Unsupervised Peoples' Speech: A 1.6k hour subset of unsupervised Peoples' Speech, automatically labelled
- NPTEL: 571hr subset of NPTEL2000, filtered for transcription quality
- VoxPopuli: 500 hours
- Unsupervised VoxPopuli: 8.6k hrs subset of unsupervised VoxPopuli, automatically labelled
- LibriSpeech-960h
- Common Voice 17.0: 1.7k hours
- 961 hours from MLS
- 155 hrs of AMI

Additionally, we used 550 hrs of TTS-generated speech data targeting virtual assistant use cases.

This data has a `maximum_duration` of 20s.
