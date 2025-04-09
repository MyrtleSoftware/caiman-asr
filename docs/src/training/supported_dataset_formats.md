## Supported Dataset Formats

CAIMAN-ASR supports reading data from four formats:

| Format         | Modes                                                    | Description                                                                                                                                                                               | Docs                                       |
| -------------- | -------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------ |
| `JSON`         | training + validation                                    | All audio as wav or flac files in a single directory hierarchy with transcripts in json file(s) referencing these audio files.                                                            | [[link]](./json_format.md)                 |
| `Webdataset`   | training + validation                                    | Audio `<key>.{flac,wav}` files stored with associated `<key>.txt` transcripts in tar file shards. Format described [here](https://github.com/webdataset/webdataset#the-webdataset-format) | [[link]](./WebDataset_format.md)           |
| `Directories`  | validation                                               | Audio (wav or flac) files and the respective text transcripts are in two separate directories.                                                                                            | [[link]](./directory_of_audio_format.md)   |
| `Hugging Face` | training (using provided conversion script) + validation | [Hugging Face Hub](https://huggingface.co/docs/hub/en/datasets-overview) datasets                                                                                                         | [[link]](./hugging_face_dataset_format.md) |

To train on your own proprietary dataset you will need to arrange for it to be in the `WebDataset` or `JSON` format.
A worked example of how to do this for the `JSON` format is provided in [json_format.md](./json_format.md).
The script `hugging_face_to_json.py` converts a Hugging Face dataset to the `JSON` format; see [here](./hugging_face_dataset_format.md#converting-a-hugging-face-dataset-to-json-format) for more details.

```admonish
If you have a feature request to support training/validation on a different format, please open a GitHub issue.
```
