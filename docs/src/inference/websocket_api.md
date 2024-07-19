# WebSocket API for Streaming Transcription

## Connecting

To start a new stream, the connection must first be set up. A WebSocket
connection starts with a HTTP `GET` request with header fields `Upgrade: websocket` and `Connection: Upgrade` as per
[RFC6455](https://datatracker.ietf.org/doc/html/rfc6455).

```
GET /asr/v0.1/stream HTTP/1.1
Host: api.myrtle.ai
Upgrade: websocket
Connection: Upgrade
Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==
Sec-WebSocket-Protocol: stream.asr.api.myrtle.ai
Sec-WebSocket-Version: 13
```

If all is well, the server will respond in the affirmative.

```
HTTP/1.1 101 Switching Protocols
Upgrade: websocket
Connection: Upgrade
Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=
Sec-WebSocket-Protocol: stream.asr.api.myrtle.ai
```

The server will return `HTTP/1.1 400 Bad Request` if the request is invalid.

## Request Parameters

Parameters are query-encoded in the request URL.

### Content Type

| Parameter      | Required | Default |
|:---------------|:---------|:--------|
| `content_type` | Yes      | -       |

Requests can specify the audio format with the `content_type` parameter. If the content type is not
specified then the server will attempt to infer it. Currently only `audio/x-raw` is supported.

Supported content types are:

- `audio/x-raw`: Unstructured and uncompressed raw audio data. If raw audio is used then additional parameters must be provided by adding:
  - `format`: The format of audio samples. Only S16LE is currently supported
  - `rate`: The sample rate of the audio. Only 16000 is currently supported
  - `channels`: The number of channels. Only 1 channel is currently supported

As a query parameter, this would look like:

```
content_type=audio/x-raw;format=S16LE;channels=1;rate=16000
```

### Model Identifier

| Parameter | Required | Default     |
|:----------|:---------|:------------|
| `model`   | No       | `"general"` |

Requests can specify a transcription model identifier.

### Model Version

| Parameter | Required | Default    |
|:----------|:---------|:-----------|
| `version` | No       | `"latest"` |

Requests can specify the transcription model version. Can be `"latest"` or a specific version id.

### Model Language

| Parameter | Required | Default |
|:----------|:---------|:--------|
| `lang`    | No       | `"en"`  |

The [BCP47](https://en.wikipedia.org/wiki/IETF_language_tag) language tag for the speech in the audio.

### Max Number of Alternatives

| Parameter      | Required | Default |
|:---------------|:---------|:--------|
| `alternatives` | No       | `1`     |

The maximum number of alternative transcriptions to provide.

### Supported Models

| Model id | Version | Supported Languages |
|:---------|:--------|:--------------------|
| general  | v1      | en                  |

## Request Frames

For `audio/x-raw` audio, raw audio samples in the format specified in the
`format` parameter should be sent in WebSocket Binary frames without padding.
Frames can be any length greater than zero.

A WebSocket Binary frame of length zero is treated as an end-of-stream (EOS) message.

## Response Frames

Response frames are sent as WebSocket Text frames containing JSON.

```
{
  "start": 0.0,
  "end": 2.0,
  "is_provisional": false,
  "alternatives": [
    {
      "transcript": "hello world",
      "confidence": 1.0
    }
  ]
}
```

### API during greedy decoding

- `start`: The start time of the transcribed interval in seconds
- `end`: The end time of the transcribed interval in seconds
- `is_provisional`: Always false for greedy decoding (but can be true for beam decoding)
- `alternatives`: Contains at most one alternative for greedy decoding (but can be more for beam decoding)
  - `transcript`: The model predictions for this audio interval. Not cumulative, so you can get the full transcript by concatenating all the `transcript` fields
  - `confidence`: Currently unused

### API during beam decoding

When decoding with a beam search, the server will return two types of response, either 'partial' or 'final' where:

- Partial responses are hypotheses that are provisional and may be removed or updated in future frames
- Final responses are hypotheses that are complete and will not change in future frames

It is recommended to use partials for low-latency streaming applications and finals for the ultimate transcription output. If latency is not a concern you can ignore the partials and concatenate the finals.

Detection of finals is done by checking beam hypotheses for a shared prefix. Typically it takes no more than 1.5 seconds to get finals (and often they are much faster) but it is possible that two similar hypotheses with similar scores are maintained in the beam for a long period of time. As such it is always recommended to use partials with state-resets enabled (see [state-reset docs](../training/state_resets.md)).

When running the asr server, partial responses are marked with `"is_provisional": true` and finals with `"is_provisional": false` and partials can be one of the many `"alternatives: [...]"`.
During a partial response,
the alternatives are ordered from most confident to least.

At each frame,
it's guaranteed that the server will send
a final or partial response, and perhaps both, as explained
[`here`](https://github.com/MyrtleSoftware/caiman-asr/blob/main/training/caiman_asr_train/rnnt/response.py):

```
{{#include ../../../training/caiman_asr_train/rnnt/response.py:finals_partials_in_mdbook}}
```

## Closing the Connection

The client should not close the WebSocket connection, it should send an EOS
message and wait for a WebSocket Close frame from the server. Closing the
connection before receivng a WebSocket Close frame from the server may cause
transcription results to be dropped.

An end-of-stream (EOS) message can be sent by sending a zero-length binary frame.

## Errors

If an error occurs, the server will send a WebSocket Close frame, with error details in the body.

| Error Code | Details                                             |
|:-----------|:----------------------------------------------------|
| 400        | Invalid parameters passed.                          |
| 503        | Maximum number of simultaneous connections reached. |
