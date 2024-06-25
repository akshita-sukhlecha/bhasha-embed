import datasets
from mteb.abstasks import AbsTaskRetrieval, MultilingualTask, TaskMetadata

from mteb_custom_tasks import utils

_LANGUAGES = ['eng_Latn', 'hin_Deva', 'hin_Latn']
_LANGUAGES_PAIRS = utils.get_lang_pairs(_LANGUAGES)
_LANGUAGES_TRIPLETS = utils.get_lang_sets(_LANGUAGES)
_LANGUAGES_SETS = _LANGUAGES_PAIRS | _LANGUAGES_TRIPLETS

# Note: lang_pair "eng_Latn-hin_Deva" means corpus is in eng_Latn and queries in hin_Deva.

class MLQAPlusRetrieval(AbsTaskRetrieval, MultilingualTask):
    metadata = TaskMetadata(
        name="MLQAPlusRetrieval",
        description="""MLQA (MultiLingual Question Answering) is a benchmark dataset for evaluating cross-lingual question answering performance.
        MLQA consists of over 5K extractive QA instances (12K in English) in SQuAD format in seven languages - English, Arabic,
        German, Spanish, Hindi, Vietnamese and Simplified Chinese. MLQA is highly parallel, with QA instances parallel between
        4 different languages on average. 
        MLQA Plus additionally has hin_Latn data generated using indictrans library""",
        reference="https://huggingface.co/datasets/mlqa",
        dataset={
            "path": "AkshitaS/facebook_mlqa_plus",
            "revision": "a0dcc369fe60dadd8351975aa94299cdd4f3b37b",
        },
        type="Retrieval",
        category="s2p",
        eval_splits=["validation", "test"],
        eval_langs=_LANGUAGES_SETS,
        main_score="ndcg_at_10",
        date=("2019-01-01", "2020-12-31"),
        form=["written"],
        domains=["Encyclopaedic"],
        task_subtypes=["Question answering"],
        license="cc-by-sa-3.0",
        socioeconomic_status="mixed",
        annotations_creators="human-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""@article{lewis2019mlqa,
        title = {MLQA: Evaluating Cross-lingual Extractive Question Answering},
        author = {Lewis, Patrick and Oguz, Barlas and Rinott, Ruty and Riedel, Sebastian and Schwenk, Holger},
        journal = {arXiv preprint arXiv:1910.07475},
        year = 2019,
        eid = {arXiv: 1910.07475}
        }""",
        n_samples={"test": 158083, "validation": 15747},
        avg_character_length={
            "test": 37352.28,
            "validation": 36952.7,
        },  # average context lengths
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        dataset = {}
        langs = set()
        for lang_pair in self.hf_subsets:
            langs.update(lang_pair.split("-"))
        for lang in langs:
            dataset[lang] = datasets.load_dataset(**self.metadata_dict["dataset"], name=lang)

        self.queries, self.corpus, self.relevant_docs = utils.get_cross_lingual_retrieval_data(
            dataset, self.hf_subsets, self.metadata.eval_splits, "context", "question", "id"
        )
        self.data_loaded = True
