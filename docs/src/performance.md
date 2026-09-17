# Performance

The solution has various configurations that trade off accuracy and performance. In this page:

- **Realtime streams (RTS)** is the number of concurrent streams that can be serviced by a single accelerator using default settings.
- **Compute latency 99th-percentile (CL99)** is the 99th-percentile compute latency, which measures how long it takes for a model to make a prediction for one audio frame. Note that CL99 increases with the number of concurrent streams.
- **User-perceived latency (UPL)** is the time difference between when the user finishes saying a word and when it is returned as a transcript by the system.
- **WER** is the Word Error Rate, a measure of the accuracy of the model. Lower is better.
- **HF Leaderboard WER** is the WER of the model on the [Huggingface Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard). WER is averaged across 8 test datasets.

The WERs in the following section are for models trained on 154k hours of mostly open-source data described at the bottom of this page.

The UPL were computed by streaming librispeech-dev-clean audio live to an FPGA backend server on-site. Please refer to this [document](./inference/user_perceived_latency.md) for more details on latencies.

## Monolingual

<!-- These use a language model for beam because it doesn't slow down inference-->

<!-- All WERs with <nopnc> prefix, inf_mode=default -->

The solution supports decoding with a [beam search](./training/beam_decoder.md)
(default beam width=4) with an [n-gram language model](./training/ngram_lm.md)
for improved accuracy. The solution supports greedy decoding for higher
throughput.

Note: RTS has two values, the first/larger is without state resetting and the
latter/smaller is with state resetting. State resets is a technique that
improves the accuracy on long utterances (over 60s) by resetting the model's
hidden state after a fixed duration. This reduces the number of real-time
streams that can be supported by around 25%. All the datasets in the
HuggingFace benchmark ase sufficienatly short that state resets have only a
minor (i.e. \<.2%) effect on WER.

| Model   | Parameters | Decoding          | RTS       | CL99 at max RTS | median UPL | HF Leaderboard WER |
| ------- | ---------- | ----------------- | --------- | --------------- | ---------- | ------------------ |
| `base`  | 85M        | `greedy`          | 2000/1600 | 25 ms           | 147 ms     | 10.34%             |
| `base`  | 85M        | `beam`, `width=4` | 1300/1200 | 80 ms           | -          | 9.02%              |
| `large` | 196M       | `greedy`          | 800/650   | 25 ms           | -          | 9.07%              |
| `large` | 196M       | `beam`, `width=4` | 500/400   | 40 ms           | 158 ms     | 8.12%              |

UPL numbers were computed from librispeech-dev-clean, the effect of state resets is not reflected in measured latencies.

Monolingual models are fine-tuned variants of the multilingual models.

## Multilingual

| Model   | Parameters | Decoding          | En     | Fr     | Es     | De     | It     | Pt     |
| ------- | ---------- | ----------------- | ------ | ------ | ------ | ------ | ------ | ------ |
| `base`  | 85M        | `greedy`          | 12.73% | 16.25% | 12.06% | 15.87% | 16.24% | 20.95% |
| `base`  | 85M        | `beam`, `width=4` | 10.94% | 14.13% | 10.27% | 14.01% | 14.44% | 18.23% |
| `large` | 196M       | `greedy`          | 11.06% | 13.46% | 9.48%  | 12.81% | 13.93% | 16.66% |
| `large` | 196M       | `beam`, `width=4` | 9.73%  | 11.82% | 8.39%  | 11.71% | 12.56% | 14.93% |

- The RTM/CL99 are the same as the above monolingual table but omitted for space.
- The WER's are against the HF leader board datasets
