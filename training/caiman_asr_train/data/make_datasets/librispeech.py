import argparse
import logging
from pathlib import Path
from typing import Dict, Sequence, Union

from caiman_asr_train.data.make_datasets.io import (
    download_file,
    extract_tar,
    md5_checksum,
)
from caiman_asr_train.data.make_datasets.manifest import (
    prepare_manifest,
    save_manifest,
    validate_manifest,
)


def setup_logger():
    logging.basicConfig(
        format="LibriSpeech %(levelname)s: %(asctime)s - %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
        level=logging.INFO,
    )


MD5 = {
    "dev-clean": "42e2234ba48799c1f50f24a7926300a1",
    "dev-other": "c8d0bcc9cca99d4f8b62fcc847357931",
    "test-clean": "32fa31d27d2e1cad72775fee3f4849a9",
    "test-other": "fb5a50374b501bb3bac4815ee91d3135",
    "train-clean-100": "2a93770f6d5c6c964bc36631d331a522",
    "train-clean-360": "c0e676e450a7ff2f54aeade5171606fa",
    "train-other-500": "d1a0fd59409feb2c614ce4d30c387708",
}


LIBRISPEECH_TRAIN960H = [
    "librispeech-train-clean-100-flac.json",
    "librispeech-train-clean-360-flac.json",
    "librispeech-train-other-500-flac.json",
]


def get_parser():
    parser = argparse.ArgumentParser(description="LibriSpeech utility parser")
    parser.add_argument(
        "--data_dir",
        default="/datasets/LibriSpeech",
        type=str,
        help="Directory to save data and manifests",
    )
    parser.add_argument(
        "--dataset_parts",
        nargs="+",
        default=[
            "dev-clean",
            "dev-other",
            "test-clean",
            "test-other",
            "train-clean-100",
            "train-clean-360",
            "train-other-500",
        ],
        help="Datasets parts to prepare, default=all",
    )
    parser.add_argument(
        "--source_url",
        default="https://www.openslr.org/resources/12/",
        type=str,
        help="Source URL to download dataset from, default=www.openslr.org",
    )
    parser.add_argument(
        "--force_download",
        action="store_true",
        help="Force download in case files exist",
    )
    parser.add_argument(
        "--num_jobs",
        default=8,
        type=int,
        help="Number of parallel jobs manifest preparation default=8",
    )
    parser.add_argument(
        "--use_relative_path",
        action="store_true",
        help="Use relative audio path in manifests",
    )
    parser.add_argument(
        "--output_format",
        default="json",
        choices=["json"],
        help="Output format for prepared LibriSpeech",
    )
    parser.add_argument(
        "--skip_download_data",
        action="store_true",
        help="Skip downloading data and prepare manifest from existing files",
    )
    parser.add_argument(
        "--skip_prepare_manifests",
        action="store_true",
        help="Skip preparing manifests and only download the dataset",
    )

    return parser


class LibriSpeech:
    def __init__(self, args):
        setup_logger()

        self.skip_download_data = args.skip_download_data
        self.skip_prepare_manifests = args.skip_prepare_manifests
        self.data_dir = Path(args.data_dir).absolute()
        self.dataset_parts = (
            args.dataset_parts
            if isinstance(args.dataset_parts, list)
            else [args.dataset_parts]
        )
        self.source_url = args.source_url
        self.force_download = args.force_download
        self.num_jobs = args.num_jobs
        self.use_relative_path = args.use_relative_path
        self.output_format = args.output_format

        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download_data(self) -> None:
        """
        Download and untar the dataset.

        :param source_url: str of the URL to download LibriSpeech from
        :param data_dir: str|Path, the path of the dir to storage the dataset.
        :param dataset_parts: "librispeech", "mini_librispeech",
            or a list of splits (e.g. "dev-clean") to download.
        :param force_download: Bool, if True, download the tars no matter if the tars exist.
        """
        source_url: str = self.source_url
        data_dir: Union[str, Path] = self.data_dir
        dataset_parts: Union[str, Sequence[str]] = self.dataset_parts
        force_download: bool = self.force_download

        # Download
        logging.info("Downloading LibriSpeech")
        for part in dataset_parts:
            url = source_url + part + ".tar.gz"
            filepath = data_dir / f"{part}.tar.gz"
            download_file(url=url, filepath=filepath, force_download=force_download)

        # Check MD5
        logging.info("Verifying checksums")
        for part in dataset_parts:
            filepath = data_dir / f"{part}.tar.gz"
            assert md5_checksum(filepath, MD5[part]), f"MD5 checksum failed for {part}"

        # Extract tar files
        logging.info("Extracting *.tar files")

        # Dont create another LibriSpeech subdir by unpacking
        # into `data_dir`/LibriSpeech/LibriSpeech
        if str(data_dir).endswith("LibriSpeech"):
            untar_dir = data_dir.parent

        for part in dataset_parts:
            filepath = data_dir / f"{part}.tar.gz"
            extract_tar(filepath=filepath, data_dir=untar_dir)

        logging.info("Download and extraction successful")

    def parse_trans_file(self, trans_file: Union[str, Path]) -> Dict[str, str]:
        """
        Parse trans.txt into a dictionary where file_id is key and file_path
        is value.
        """
        return {
            line.split()[0]: line.split(maxsplit=1)[1].strip()
            for line in open(trans_file)
        }

    def prepare_manifests(self):
        data_dir: Union[str, Path] = self.data_dir
        dataset_parts: Union[str, Sequence[str]] = self.dataset_parts
        use_relative_path: bool = self.use_relative_path
        num_jobs: int = self.num_jobs
        audio_ext = "flac"
        trans_ext = "trans.txt"

        manifests = {}
        for part in dataset_parts:
            logging.info(f"Parsing audio and transcripts files for `{part}`")
            subdir = data_dir / Path(part)
            trans_files = subdir.rglob(f"*.{trans_ext}")
            audio_files = subdir.rglob(f"*[0-9].{audio_ext}")
            if use_relative_path:
                audio_files = [f.relative_to(data_dir) for f in audio_files]
            audio_dict = {
                str(f.name).removesuffix(f".{audio_ext}"): f.absolute()
                for f in audio_files
            }

            # Parse trans files into a dict{file_id: transcript}
            trans_dict = {}
            for trans_file in trans_files:
                trans_dict.update(self.parse_trans_file(trans_file))

            # Check we are not missing any transcripts or audios
            valid_ids = set(audio_dict.keys()) & set(trans_dict.keys())
            if len(valid_ids) < len(audio_dict) or len(valid_ids) < len(trans_dict):
                logging.warning(
                    "It appears some (transcript, audio)-pairs are missing. "
                    "Will process only valid pairs"
                )

            input_data = [
                dict(
                    audio_file=audio_dict[valid_id],
                    transcript=trans_dict[valid_id],
                )
                for valid_id in valid_ids
            ]

            # Create manifest
            logging.info("Generating manifest")
            manifest = prepare_manifest(
                input_data, num_jobs, data_dir if use_relative_path else None
            )

            logging.info("Validating manifest")
            validate_manifest(manifest, data_dir if use_relative_path else None)
            manifests[part] = manifest

            # Save manifests
            logging.info(
                f"Saving librispeech-{part}.json manifest to disk, "
                f"contains {len(manifest)} entries"
            )
            manifest_path = self.data_dir / f"librispeech-{part}.{audio_ext}.json"
            save_manifest(manifests[part], manifest_path)

    def run(self):
        # Download and extract
        if not self.skip_download_data:
            self.download_data()

        # Prepare and save manifests
        if not self.skip_prepare_manifests:
            self.prepare_manifests()


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    librispeech = LibriSpeech(args)
    librispeech.run()
