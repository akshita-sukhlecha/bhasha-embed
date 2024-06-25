import datasets
from mteb.abstasks import AbsTaskRetrieval, MultilingualTask, TaskMetadata

from mteb_custom_tasks import utils

_LANGUAGES = ['eng_Latn', 'hin_Deva', 'hin_Latn']
_LANGUAGES_PAIRS = utils.get_lang_pairs(_LANGUAGES)
_LANGUAGES_TRIPLETS = utils.get_lang_sets(_LANGUAGES)
_LANGUAGES_SETS = _LANGUAGES_PAIRS | _LANGUAGES_TRIPLETS

_EVAL_SPLITS = ["test"]


class BelebelePlusRetrieval(MultilingualTask, AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BelebelePlusRetrieval",
        dataset={
            "path": "facebook/belebele",
            "revision": "75b399394a9803252cfec289d103de462763db7c",
        },
        description=(
            "Belebele is a multiple-choice machine reading comprehension (MRC) dataset spanning 122 language variants "
            "(including 115 distinct languages and their scripts)"
        ),
        type="Retrieval",
        category="s2p",
        eval_splits=_EVAL_SPLITS,
        eval_langs=_LANGUAGES_SETS,
        reference="https://arxiv.org/abs/2308.16884",
        main_score="ndcg_at_10",
        license="CC-BY-SA-4.0",
        domains=["Web", "News"],
        text_creation="created",
        n_samples={"test": 103500},  # number of languages * 900
        date=("2023-08-31", "2023-08-31"),
        form=["written"],
        task_subtypes=["Question answering"],
        socioeconomic_status="mixed",
        annotations_creators="expert-annotated",
        dialect=[],
        avg_character_length={"test": 568},  # avg length of query-passage pairs
        bibtex_citation="""@article{bandarkar2023belebele,
        title={The Belebele Benchmark: a Parallel Reading Comprehension Dataset in 122 Language Variants}, 
        author={Lucas Bandarkar and Davis Liang and Benjamin Muller and Mikel Artetxe and Satya Narayan Shukla and Donald Husa and Naman Goyal and Abhinandan Krishnan and Luke Zettlemoyer and Madian Khabsa},
        year={2023},
        journal={arXiv preprint arXiv:2308.16884}
        }""",
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        dataset = {}
        langs = set()
        for lang_pair in self.hf_subsets:
            langs.update(lang_pair.split("-"))
        for lang in langs:
            dataset[lang] = {
                "test": datasets.load_dataset(**self.metadata_dict["dataset"], split=lang)
            }

        self.queries, self.corpus, self.relevant_docs = utils.get_cross_lingual_retrieval_data(
            dataset, self.hf_subsets, self.metadata.eval_splits, "flores_passage", "question", "link"
        )
        self.data_loaded = True
