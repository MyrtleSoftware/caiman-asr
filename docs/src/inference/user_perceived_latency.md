# User-perceived Latency

User-perceived latency (UPL) is defined as the time difference between the instant when the speaker finishes saying a word and the instant when the word appears as a transcript on a screen. In practice, it can be measured by streaming audio to a server live and measuring the response time for each word. The following table summarizes UPL for base and large models measured by streaming librispeech-dev-clean dataset to an on-site server with an FPGA backend running at maximum RTS.

| Model / decoding | Mean UPL | p90 UPL  | p99 UPL |
|------------------|----------|----------|---------|
| base / greedy    |  159 ms  |  303 ms  | 451 ms  |
| large / beam     |  163 ms  |  326 ms  | 629 ms  |

UPL is the sum of the following latencies:

- audio frame latency (recording device)
- compute latency (model)
- network latency (network)
- emission latency (model)

## Audio Frame Latency

Streaming ASR systems buffer audio for a fixed duration before sending it to the server. The buffer length is typically set to match the audio chunk length used during training. CAIMAN-ASR models were trained on audio chunks of 60 ms. However, the end of the word can appear anytime during buffering, which means that its contribution is half the audio frame length on average (30 ms for CAIMAN-ASR).

## Compute Latency

The compute latency measures how long it takes for a model to make a prediction for one audio frame. This latency depends on the model size, accelerator backend, server load, decoding strategy, and whether state resets is turned on. The contribution of the compute latency is strictly additive. The following tables summarize 99th-percentile compute latencies (CL99) at maximum number of real-time streams for an FPGA backend and various setups.

| Model | Parameters | Decoding      | CL99  | CL99 + state resets |
|-------|------------|---------------|-------|---------------------|
| base  | 85M        | greedy        | 25 ms | 45 ms               |
| base  | 85M        | beam, width=4 | 80 ms | 50 ms               |
| large | 196M       | greedy        | 25 ms | 55 ms               |
| large | 196M       | beam, width=4 | 40 ms | 60 ms               |

## Network Latency

The network latency corresponds to sending the audio chunk to the server and receiving a response back. The contribution of the network latency is roughly equal to the round-trip response time, as measured using `ping`. In case the solution is deployed on-premise, the expected value is well below 1 ms. In the case the solution is deployed on cloud, the network latency can exceed 100 ms.

## Emission Latency

The emission latency (EL) is explained in detail in this [document](../training/emission_latency.md). Together with the network latency, it is the only latency the user can directly influence, as explained in [here](../training/delay_penalty.md). Its contribution is strictly additive. The table below summarizes emission latency for the latest models averaged across all HF Leaderboard evaluation datasets.

| Model  | Mean EL | p90 EL |
|--------|---------|--------|
| base   | 121 ms  | 281 ms |
| large  | 82 ms   | 287 ms |
