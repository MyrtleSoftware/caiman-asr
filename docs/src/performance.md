# Performance

The solution has various configurations that trade off accuracy and performance. In this page:

- **Realtime streams (RTS)** is the number of concurrent streams that can be serviced by a single accelerator using default settings.
- **Compute latency 99th-percentile (CL99)** is the 99th-percentile compute latency, which measures how long it takes for a model to make a prediction for one audio frame. Note that CL99 increases with the number of concurrent streams.
- **User-perceived latency (UPL)** is the time difference between when the user finishes saying a word and when it is returned as a transcript by the system.
- **WER** is the Word Error Rate, a measure of the accuracy of the model. Lower is better.
- **HF Leaderboard WER** is the WER of the model on the [Huggingface Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard). WER is averaged across 9 test datasets.

The WERs in the following section are for models trained on 13k hours of open-source data described at the bottom of this page.

The UPL were computed by streaming librispeech-dev-clean audio live to an FPGA backend server on-site. Please refer to this [document](./inference/user_perceived_latency.md) for more details on latencies.

## Beam search with n-gram LM

The solution supports decoding with a [beam search](./training/beam_decoder.md) (default beam width=4) with an [n-gram language model](./training/ngram_lm.md) for improved accuracy. The solution supports greedy decoding for higher throughput.

| Model   | Parameters | Decoding          | RTS   | CL99 at max RTS | mean UPL | HF Leaderboard WER  |
|---------|------------|-------------------|-------|-----------------|------------|---------------------|
| `base`  | 85M        | `greedy`          | 2000  |  25 ms          | 159 ms     | 13.74%              |
| `base`  | 85M        | `beam`, `width=4` | 1300  |  80 ms          | -          | 12.78%              |
| `large` | 196M       | `greedy`          | 800   |  25 ms          | -          | 12.02%              |
| `large` | 196M       | `beam`, `width=4` | 500   |  40 ms          | 163 ms     | 11.37%              |

## State resets

State resets is a technique that improves the accuracy on long utterances (over 60s) by resetting the model's hidden state after a fixed duration. This reduces the number of real-time streams that can be supported by around 25%:

| Model   | Parameters | Decoding          | RTS   | CL99 at max RTS | mean UPL | HF Leaderboard WER  |
|---------|------------|-------------------|-------|-----------------|------------|---------------------|
| `base`  | 85M        | `greedy`          | 1600  | 45 ms           | 159 ms     | 13.70%              |
| `base`  | 85M        | `beam`, `width=4` | 1200  | 50 ms           |  -         | 12.83%              |
| `large` | 196M       | `greedy`          | 650   | 55 ms           |  -         | 11.99%              |
| `large` | 196M       | `beam`, `width=4` | 400   | 60 ms           | 163 ms     | 11.38%              |

Note that most of the data in the Huggingface leaderboard is less than 60s long so the impact of state resets is not reflected in the leaderboard WER.

Since the UPL numbers were computed from librispeech-dev-clean, the effect of state resets is not reflected in measured latencies.

## 13k hour dataset <a name="13k_hrs"></a>

The models above were trained on 13k hrs of open-source training data consisting of:

- LibriSpeech-960h
- Common Voice 17.0
- 961 hours from MLS
- Peoples' Speech: A ~9389 hour subset filtered for transcription quality
- 155 hrs of AMI

This data has a `maximum_duration` of 20s.
