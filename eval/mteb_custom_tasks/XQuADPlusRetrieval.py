import datasets
from mteb.abstasks import AbsTaskRetrieval, MultilingualTask, TaskMetadata

from mteb_custom_tasks import utils

_LANGUAGES = ['eng_Latn', 'hin_Deva', 'hin_Latn']
_LANGUAGES_PAIRS = utils.get_lang_pairs(_LANGUAGES)
_LANGUAGES_TRIPLETS = utils.get_lang_sets(_LANGUAGES)
_LANGUAGES_SETS = _LANGUAGES_PAIRS | _LANGUAGES_TRIPLETS


# Note: lang_pair "eng_Latn-hin_Deva" means corpus is in eng_Latn and queries in hin_Deva.

class XQuADPlusRetrieval(MultilingualTask, AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="XQuADPlusRetrieval",
        dataset={
            "path": "AkshitaS/google_xquad_plus",
            "revision": "cc81aa8c76cbf45f4e239d852c4a5d36cdfc6b32",
        },
        description="XQuAD is a benchmark dataset for evaluating cross-lingual question answering performance. It is repurposed retrieving relevant context for each question."
                    "XQuAD Plus additionally has hin_Latn data generated using indictrans library",
        reference="https://huggingface.co/datasets/xquad",
        type="Retrieval",
        category="s2p",
        eval_splits=["validation"],
        eval_langs=_LANGUAGES_SETS,
        main_score="ndcg_at_10",
        date=("2019-05-21", "2019-11-21"),
        form=["written"],
        domains=["Web"],
        task_subtypes=["Question answering"],
        license="CC BY-SA 4.0",
        socioeconomic_status="mixed",
        annotations_creators="human-annotated",
        dialect=[],
        text_creation="created",
        bibtex_citation="""@article{Artetxe:etal:2019,
              author    = {Mikel Artetxe and Sebastian Ruder and Dani Yogatama},
              title     = {On the cross-lingual transferability of monolingual representations},
              journal   = {CoRR},
              volume    = {abs/1910.11856},
              year      = {2019},
              archivePrefix = {arXiv},
              eprint    = {1910.11856}
        }
        @inproceedings{
              dumitrescu2021liro,
              title={LiRo: Benchmark and leaderboard for Romanian language tasks},
              author={Stefan Daniel Dumitrescu and Petru Rebeja and Beata Lorincz and Mihaela Gaman and Andrei Avram and Mihai Ilie and Andrei Pruteanu and Adriana Stan and Lorena Rosia and Cristina Iacobescu and Luciana Morogan and George Dima and Gabriel Marchidan and Traian Rebedea and Madalina Chitez and Dani Yogatama and Sebastian Ruder and Radu Tudor Ionescu and Razvan Pascanu and Viorica Patraucean},
              booktitle={Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 1)},
              year={2021},
              url={https://openreview.net/forum?id=JH61CD7afTv}
        }""",
        n_samples={"test": 1190},
        avg_character_length={"test": 788.7},
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
