## Supported Dataset Formats

CAIMAN-ASR supports reading data from the following formats:

| Format  | Modes                 | Description                                                                                                      | Docs                        |
| ------- | --------------------- | ---------------------------------------------------------------------------------------------------------------- | --------------------------- |
| `JSONL` | training + validation | All audio files in a single directory hierarchy with transcripts in JSONL file(s) referencing these audio files. | [[link]](./jsonl_format.md) |
| `SHAR`  | training              | Sharded tarred audio with per-shard `JSONL` metadata                                                             | [[link]](./shar_format.md)  |

To train on your own proprietary dataset you will need to arrange for it to be in the `JSONL` or `SHAR` format.
A worked example of how to do this for the `JSONL` format is provided in [jsonl_format.md](./jsonl_format.md).
For more details on the `SHAR` format please see [shar_format.md](./shar_format.md).

```admonish
If you have a feature request to support training/validation on a different format, please open a GitHub issue.
```
