import random

from datasets import Dataset, Features, Value, load_dataset
from mteb.abstasks import AbsTaskBitextMining, CrosslingualTask, TaskMetadata

_LANGUAGES = {
    "eng_Latn": "lcsalign-en",
    "hin_Deva": "lcsalign-hi",
    "hin_Deva_cm": "lcsalign-hicm",  # has code-mixed words (not sentences)
    "hin_Latn_cm": "lcsalign-hicmrom"
}


def get_lang_pairs(langs):
    lang_pairs = {}
    for x in langs:
        for y in langs:
            if x != y:
                lang_pairs[f"{x}-{y}"] = [
                    x.replace("_cm", "").replace("_", "-"),
                    y.replace("_cm", "").replace("_", "-")
                ]
    return lang_pairs


_LANGUAGES_PAIRS = get_lang_pairs(_LANGUAGES)


class HinMixBitextMining(AbsTaskBitextMining, CrosslingualTask):
    metadata = TaskMetadata(
        name="HinMixBitextMining",
        dataset={
            "path": "kartikagg98/HINMIX_hi-en",
            "revision": "5735b2d0737980b03c50ee643ad8ac31da5d8531",
        },
        description="HINMIX is a massive parallel codemixed dataset for Hindi-English code switching.",
        reference="https://huggingface.co/datasets/kartikagg98/HINMIX_hi-en",
        type="BitextMining",
        category="s2s",
        eval_splits=["test"],
        eval_langs=_LANGUAGES_PAIRS,
        main_score="f1",
        date=None,
        form=["written"],
        domains=None,
        task_subtypes=[],
        license="apache-2.0",
        socioeconomic_status="mixed",
        annotations_creators="expert-annotated",
        dialect=None,
        text_creation="created",
        bibtex_citation="",
        n_samples={"test": 2507},
        avg_character_length=None,
    )

    def load_data(self, **kwargs):
        """Load dataset from HuggingFace hub and convert it to the standard format."""
        if self.data_loaded:
            return

        self.dataset = {}
        langs = set()
        lang_dataset = {}

        for lang_set in self.hf_subsets:
            langs = langs | set(lang_set.split("-"))

        for lang in langs:
            lang_dataset[lang] = {}
            for split in self.metadata.eval_splits:
                lang_dataset[lang][split] = load_dataset(**self.metadata_dict["dataset"],
                                                         name=_LANGUAGES[lang], split=split)
                if lang == 'hin_Deva_cm':
                    hi_data = load_dataset(**self.metadata_dict["dataset"], name=_LANGUAGES['hin_Deva'], split=split)
                    for idx, t in enumerate(lang_dataset[lang][split]['text']):
                        if random.random() > 0.5:
                            lang_dataset[lang][split]['text'][idx] = hi_data['text'][idx]

        for lang_set in self.hf_subsets:
            self.dataset[lang_set] = {}
            langs = lang_set.split("-")
            for split in self.metadata.eval_splits:
                self.dataset[lang_set][split] = Dataset.from_dict({
                    "sentence1": lang_dataset[langs[0]][split]["text"],
                    "sentence2": lang_dataset[langs[1]][split]["text"]
                }, features=Features({
                    "sentence1": Value("string"),
                    "sentence2": Value("string")
                }))

        self.data_loaded = True
