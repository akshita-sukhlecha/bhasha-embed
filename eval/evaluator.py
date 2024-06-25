import json
import os.path

import mteb
from sentence_transformers import SentenceTransformer
from mteb.tasks import NFCorpus, MLQuestionsRetrieval
from mteb.tasks import XPQARetrieval, WikipediaRetrievalMultilingual
from mteb.tasks import TatoebaBitextMining
from mteb.tasks import WikipediaRerankingMultilingual, IndicReviewsClusteringP2P, SIB200ClusteringFast
from mteb.tasks import MTOPDomainClassification, MTOPIntentClassification, MultiHateClassification, \
    SIB200Classification

from mteb_custom_tasks import MLQAPlusRetrieval, XQuADPlusRetrieval, BelebelePlusRetrieval, SemRel24PlusSTS

tasks = {
    "retrieval": {
        "hin_Deva-hin_Deva": [
            XPQARetrieval(hf_subsets=["hin-hin"]),
            WikipediaRetrievalMultilingual(hf_subsets=["hi"])
        ],
        "eng_Latn-eng_Latn": [
            NFCorpus(),
            MLQuestionsRetrieval()
        ],
        "eng_Latn-hin_Deva": [
            XPQARetrieval(hf_subsets=["eng-hin"])
        ],
        "hin_Deva-eng_Latn": [
            XPQARetrieval(hf_subsets=["hin-eng"])
        ]
    },
    "bitext": {
        "hin_Deva-eng_Latn": [
            TatoebaBitextMining(hf_subsets=["hin-eng"])
        ]
    },
    "classification": {
        "hin_Deva": [
            MTOPDomainClassification(hf_subsets=["hi"]),
            MTOPIntentClassification(hf_subsets=["hi"]),
            MultiHateClassification(hf_subsets=["hin"]),
            SIB200Classification(hf_subsets=["hin_Deva"]),
        ]
    },
    "ranking": {
        "hin_Deva": [
            WikipediaRerankingMultilingual(hf_subsets=["hi"]),
        ]
    },
    "clustering": {
        "hin_Deva": [
            IndicReviewsClusteringP2P(hf_subsets=["hi"]),
            SIB200ClusteringFast(hf_subsets=["hin_Deva"]),
        ]
    },
    "STS": {
        "hin_Deva": [
            SemRel24PlusSTS(hf_subsets=["hin_Deva"])
        ],
        "hin_Latn": [
            SemRel24PlusSTS(hf_subsets=["hin_Latn"])
        ]
    }
}

################################## Add custom retrieval tasks ###################################

common_retrieval_tasks = [BelebelePlusRetrieval, MLQAPlusRetrieval, XQuADPlusRetrieval]
mix_lingual_langs_2 = [
    "eng_Latn-hin_Deva-eng_Latn",
    "eng_Latn-hin_Deva-hin_Deva",
]
for lang_set in mix_lingual_langs_2:
    tasks["retrieval"][lang_set] = [task(hf_subsets=[lang_set]) for task in common_retrieval_tasks]

langs = ["hin_Latn", "hin_Deva", "eng_Latn"]
for lang_x in langs:
    for lang_y in langs:
        lang_pair = f"{lang_x}-{lang_y}"
        lang_tasks = [task(hf_subsets=[lang_pair]) for task in common_retrieval_tasks]
        tasks["retrieval"][lang_pair] = tasks["retrieval"].get(lang_pair, []) + lang_tasks

mix_lingual_langs_3 = [
    "eng_Latn-hin_Deva-hin_Latn-eng_Latn",
    "eng_Latn-hin_Deva-hin_Latn-hin_Deva",
    "eng_Latn-hin_Deva-hin_Latn-hin_Latn"
]
for lang_set in mix_lingual_langs_3:
    tasks["retrieval"][lang_set] = [task(hf_subsets=[lang_set]) for task in common_retrieval_tasks]

#################################### Evaluation #####################################

model_id = "AkshitaS/Hinglish-embedding-base"  # "intfloat/multilingual-e5-base"
model = SentenceTransformer(model_id)
output_folder = "results/" + model_id.replace("/", "-")

results = {}
os.makedirs(output_folder, exist_ok=True)

for task_type, lang_to_tasks_map in tasks.items():
    results[task_type] = {}
    for lang_pair, task_list in lang_to_tasks_map.items():
        results[task_type][lang_pair] = {}
        for task in task_list:
            evaluation = mteb.MTEB(tasks=[task])
            if task.metadata.name == "XQuADPlusRetrieval":
                eval_split = "validation"
            else:
                eval_split = "test"
            task_results = evaluation.run(model, output_folder=output_folder, eval_splits=[eval_split],
                                          overwrite_results=True)
            results[task_type][lang_pair][task.metadata.name] = task_results[0].scores[eval_split][0]["main_score"]

            with open(output_folder + "/overall_results.json", "w") as fp:
                fp.write(json.dumps(results, indent=4))
