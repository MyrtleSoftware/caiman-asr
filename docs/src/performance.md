# Performance

The solution has various configurations that trade off accuracy and performance. In this page:

- **Realtime streams (RTS)** is the number of concurrent streams that can be serviced by a single accelerator using default settings.
- **Compute latency 99th-percentile (CL99)** is the 99th-percentile compute latency, which measures how long it takes for a model to make a prediction for one audio frame. Note that CL99 increases with the number of concurrent streams.
- **User-perceived latency (UPL)** is the time difference between when the user finishes saying a word and when it is returned as a transcript by the system.
- **WER** is the Word Error Rate, a measure of the accuracy of the model. Lower is better.
- **HF Leaderboard WER** is the WER of the model on the [Huggingface Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard). WER is averaged across 8 test datasets.

The WERs in the following section are for models trained on 154k hours of mostly open-source data described at the bottom of this page.

The UPL were computed by streaming librispeech-dev-clean audio live to an FPGA backend server on-site. Please refer to this [document](./inference/user_perceived_latency.md) for more details on latencies.

## Without state resets

<!-- These use a language model for beam because it doesn't slow down inference-->

<!-- All WERs with <nopnc> prefix -->

The solution supports decoding with a [beam search](./training/beam_decoder.md) (default beam width=4) with an [n-gram language model](./training/ngram_lm.md) for improved accuracy. The solution supports greedy decoding for higher throughput.

| Model   | Parameters | Decoding          | RTS  | CL99 at max RTS | median UPL | HF Leaderboard WER | Earnings21 WER |
| ------- | ---------- | ----------------- | ---- | --------------- | ---------- | ------------------ | -------------- |
| `base`  | 85M        | `greedy`          | 2000 | 25 ms           | 147 ms     | 10.43%             | 20.87%         |
| `base`  | 85M        | `beam`, `width=4` | 1300 | 80 ms           | -          | 9.03%              | 14.09%         |
| `large` | 196M       | `greedy`          | 800  | 25 ms           | -          | 8.99%              | 34.26%         |
| `large` | 196M       | `beam`, `width=4` | 500  | 40 ms           | 158 ms     | 7.92%              | 23.74%         |

## State resets

<!-- These use a language model for beam because it doesn't slow down inference-->

State resets is a technique that improves the accuracy on long utterances (over 60s) by resetting the model's hidden state after a fixed duration. This reduces the number of real-time streams that can be supported by around 25%:

| Model   | Parameters | Decoding          | RTS  | CL99 at max RTS | median UPL | HF Leaderboard WER | Earnings21 WER |
| ------- | ---------- | ----------------- | ---- | --------------- | ---------- | ------------------ | -------------- |
| `base`  | 85M        | `greedy`          | 1600 | 45 ms           | 147 ms     | 10.43%             | 15.41%         |
| `base`  | 85M        | `beam`, `width=4` | 1200 | 50 ms           | -          | 9.02%              | 12.87%         |
| `large` | 196M       | `greedy`          | 650  | 55 ms           | -          | 8.99%              | 13.54%         |
| `large` | 196M       | `beam`, `width=4` | 400  | 60 ms           | 158 ms     | 7.93%              | 11.37%         |

Note that most of the data in the HuggingFace Leaderboard is less than 60s long so the impact of state resets is not reflected in the leaderboard WER.
The average duration of audio in Earnings21 is ~54 minutes, making it a more representative benchmark for evaluating the impact of state resets on long-form transcription accuracy.

Since the UPL numbers were computed from librispeech-dev-clean, the effect of state resets is not reflected in measured latencies.

## WER test set breakdown

The WER breakdown across the HuggingFace Leaderboard is shown for the 'highest-throughput' and 'most-accurate' configurations in the table below:

<!-- highest-throughput: base-ekans, greedy, sr=off -->

<!-- most-accurate: large-arbok. beam=4, sr=off -->

| Model Configuration | AVERAGE | AMI    | E22 (segmented) | Gigaspeech | LS test clean | LS test other | SPGISpeech | TED-LIUM | VoxPopuli |
| ------------------- | ------- | ------ | --------------- | ---------- | ------------- | ------------- | ---------- | -------- | --------- |
| base (greedy)       | 10.43%  | 14.82% | 17.36%          | 14.41%     | 3.83%         | 9.38%         | 6.06%      | 6.78%    | 10.83%    |
| large (beam=4)      | 7.92%   | 11.46% | 13.24%          | 11.52%     | 2.59%         | 6.66%         | 4.38%      | 4.95%    | 8.53%     |

## 154k hour dataset <a name="13k_hrs"></a>

The models above were trained on 154k hrs of mostly open-source training data consisting of:

- Emilia-YODAS: 93k hours of YODAS; filtered and transcribed with Emilia-Pipe.
- YODAS: 19k hour subset of YODAS manual English subset filtered for transcription quality.
- Peoples' Speech: A 9.4k hour subset filtered for transcription quality.
- Unsupervised Peoples' Speech: A 17k hour subset of unsupervised Peoples' Speech; automatically labelled.
- NPTEL: 570 hour subset of NPTEL2000, filtered for transcription quality.
- VoxPopuli: 500 hours.
- Unsupervised VoxPopuli: 8.6k hrs subset of unsupervised VoxPopuli; automatically labelled.
- National Speech Corpus: 980 hour subset; automatically labelled and filtered for transcription quality.
- LibriSpeech: 960 hours.
- Common Voice 17.0: 1.7k hours
- MLS: 961 hours.
- Spoken Wikipedia Corpora: 350 hours.
- AMI: 155 hours.

Additionally, we used 550 hrs of TTS-generated speech data targeting virtual assistant use cases.

This data has a `maximum_duration` of 20s.
