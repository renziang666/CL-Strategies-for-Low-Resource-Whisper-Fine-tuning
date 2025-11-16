# coding=utf-8
# Copyright 2022 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Common Voice Dataset"""


import csv
import os
import json

import datasets
from datasets.utils.py_utils import size_str
from tqdm import tqdm

from .languages import LANGUAGES
from .release_stats import STATS


_CITATION = """\
@inproceedings{commonvoice:2020,
  author = {Ardila, R. and Branson, M. and Davis, K. and Henretty, M. and Kohler, M. and Meyer, J. and Morais, R. and Saunders, L. and Tyers, F. M. and Weber, G.},
  title = {Common Voice: A Massively-Multilingual Speech Corpus},
  booktitle = {Proceedings of the 12th Conference on Language Resources and Evaluation (LREC 2020)},
  pages = {4211--4215},
  year = 2020
}
"""

_HOMEPAGE = "https://commonvoice.mozilla.org/en/datasets"

_LICENSE = "https://creativecommons.org/publicdomain/zero/1.0/"

# TODO: change "streaming" to "main" after merge!
_BASE_URL = "https://huggingface.co/datasets/fsicoli/common_voice_22_0/resolve/main/"

_AUDIO_URL = _BASE_URL + "audio/{lang}/{split}/{lang}_{split}_{shard_idx}.tar"

_TRANSCRIPT_URL = _BASE_URL + "transcript/{lang}/{split}.tsv"

_N_SHARDS_URL = "https://huggingface.co/datasets/fsicoli/common_voice_22_0/raw/main/" + "n_shards.json"


class CommonVoiceConfig(datasets.BuilderConfig):
    """BuilderConfig for CommonVoice."""

    def __init__(self, name, version, **kwargs):
        self.language = kwargs.pop("language", None)
        self.release_date = kwargs.pop("release_date", None)
        self.num_clips = kwargs.pop("num_clips", None)
        self.num_speakers = kwargs.pop("num_speakers", None)
        self.validated_hr = kwargs.pop("validated_hr", None)
        self.total_hr = kwargs.pop("total_hr", None)
        self.size_bytes = kwargs.pop("size_bytes", None)
        self.size_human = size_str(self.size_bytes)
        description = (
            f"Common Voice speech to text dataset in {self.language} released on {self.release_date}. "
            f"The dataset comprises {self.validated_hr} hours of validated transcribed speech data "
            f"out of {self.total_hr} hours in total from {self.num_speakers} speakers. "
            f"The dataset contains {self.num_clips} audio clips and has a size of {self.size_human}."
        )
        super(CommonVoiceConfig, self).__init__(
            name=name,
            version=datasets.Version(version),
            description=description,
            **kwargs,
        )


class CommonVoice(datasets.GeneratorBasedBuilder):
    DEFAULT_WRITER_BATCH_SIZE = 1000

    BUILDER_CONFIGS = [
        CommonVoiceConfig(
            name=lang,
            version=STATS["version"],
            language=LANGUAGES[lang],
            release_date=STATS["date"],
            num_clips=lang_stats["clips"],
            num_speakers=lang_stats["users"],
            validated_hr=float(lang_stats["validHrs"]) if lang_stats["validHrs"] else None,
            total_hr=float(lang_stats["totalHrs"]) if lang_stats["totalHrs"] else None,
            size_bytes=int(lang_stats["size"]) if lang_stats["size"] else None,
        )
        for lang, lang_stats in STATS["locales"].items()
    ]

    def _info(self):
        total_languages = len(STATS["locales"])
        total_valid_hours = STATS["totalValidHrs"]
        description = (
            "Common Voice is Mozilla's initiative to help teach machines how real people speak. "
            f"The dataset currently consists of {total_valid_hours} validated hours of speech "
            f" in {total_languages} languages, but more voices and languages are always added."
        )
        features = datasets.Features(
            {
                "client_id": datasets.Value("string"),
                "path": datasets.Value("string"),
                "audio": datasets.features.Audio(sampling_rate=48_000),
                "sentence": datasets.Value("string"),
                "up_votes": datasets.Value("int64"),
                "down_votes": datasets.Value("int64"),
                "age": datasets.Value("string"),
                "gender": datasets.Value("string"),
                "accent": datasets.Value("string"),
                "locale": datasets.Value("string"),
                "segment": datasets.Value("string"),
                "variant": datasets.Value("string"),
            }
        )

        return datasets.DatasetInfo(
            description=description,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
            version=self.config.version,
        )

    def _split_generators(self, dl_manager):
        lang = self.config.name
        n_shards_path = dl_manager.download_and_extract(_N_SHARDS_URL)
        with open(n_shards_path, encoding="utf-8") as f:
            n_shards = json.load(f)

        audio_urls = {}
        splits = ("train", "dev", "test", "other", "invalidated")
        for split in splits:
            audio_urls[split] = [
                _AUDIO_URL.format(lang=lang, split=split, shard_idx=i) for i in range(n_shards[lang][split])
            ]
        archive_paths = dl_manager.download(audio_urls)
        local_extracted_archive_paths = dl_manager.extract(archive_paths) if not dl_manager.is_streaming else {}

        meta_urls = {split: _TRANSCRIPT_URL.format(lang=lang, split=split) for split in splits}
        meta_paths = dl_manager.download_and_extract(meta_urls)

        split_generators = []
        split_names = {
            "train": datasets.Split.TRAIN,
            "dev": datasets.Split.VALIDATION,
            "test": datasets.Split.TEST,
        }
        for split in splits:
            split_generators.append(
                datasets.SplitGenerator(
                    name=split_names.get(split, split),
                    gen_kwargs={
                        "local_extracted_archive_paths": local_extracted_archive_paths.get(split),
                        "archives": [dl_manager.iter_archive(path) for path in archive_paths.get(split)],
                        "meta_path": meta_paths[split],
                    },
                ),
            )

        return split_generators

    def _generate_examples(self, local_extracted_archive_paths, archives, meta_path):
        data_fields = list(self._info().features.keys())
        metadata = {}
        with open(meta_path, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            for row in tqdm(reader, desc="Reading metadata..."):
                if not row["path"].endswith(".mp3"):
                    row["path"] += ".mp3"
                # accent -> accents in CV 8.0
                if "accents" in row:
                    row["accent"] = row["accents"]
                    del row["accents"]
                # if data is incomplete, fill with empty values
                for field in data_fields:
                    if field not in row:
                        row[field] = ""
                metadata[row["path"]] = row

        for i, audio_archive in enumerate(archives):
            for path, file in audio_archive:
                _, filename = os.path.split(path)
                if filename in metadata:
                    result = dict(metadata[filename])
                    # set the audio feature and the path to the extracted file
                    path = os.path.join(local_extracted_archive_paths[i], path) if local_extracted_archive_paths else path
                    result["audio"] = {"path": path, "bytes": file.read()}
                    result["path"] = path
                    yield path, result