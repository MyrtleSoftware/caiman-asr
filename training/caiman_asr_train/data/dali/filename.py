#!/usr/bin/env python3

from beartype import beartype

from caiman_asr_train.data.dali.token_cache import unicode_to_str
from caiman_asr_train.data.decide_on_loader import DataSource


@beartype
class FileNameExtractor:
    def __init__(self, output_files: dict[str, dict] | None, data_source: DataSource):
        self.data_source = data_source
        if data_source is DataSource.JSON:
            self.id_to_fname = {
                label_and_duration["label"]: fname
                for fname, label_and_duration in output_files.items()
            }
        else:
            assert (
                output_files is None
            ), "Should not have output_files when not using json"

    def get_fnames(self, data: dict) -> list[str]:
        if self.data_source is DataSource.JSON:
            return [self.id_to_fname[int(i)] for i in data["label"]]
        else:
            # DALI has returned the fname
            return [unicode_to_str(seq) for seq in data["fname"]]
