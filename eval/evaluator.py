import json
import os

import mteb
from sentence_transformers import SentenceTransformer
from mteb.tasks import NFCorpus, MLQuestionsRetrieval, XPQARetrieval
from mteb.tasks import TatoebaBitextMining, WikipediaRerankingMultilingual
from mteb.tasks import IndicReviewsClusteringP2P, SIB200ClusteringFast
from mteb.tasks import MTOPDomainClassification, MTOPIntentClassification, MultiHateClassification, \
    SIB200Classification

from mteb_custom_tasks import MLQAPlusRetrieval, XQuADPlusRetrieval, BelebelePlusRetrieval, SemRel24PlusSTS

tasks = {
    "retrieval": {
        "hin_Deva-hin_Deva": [
            XPQARetrieval(hf_subsets=["hin-hin"]),
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
            WikipediaRerankingMultilingual(hf_subsets=["hi"])
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


models = [
    {"name": "AkshitaS/bhasha-embed-v0", "revision": "f4aa60c473d434333c824218291f719bd6d65444"},
    {"name": "intfloat/multilingual-e5-base", "revision": "d13f1b27baf31030b7fd040960d60d909913633f"},
    {"name": "intfloat/multilingual-e5-large", "revision": "4dc6d853a804b9c8886ede6dda8a073b7dc08a81"},
    {"name": "sentence-transformers/LaBSE", "revision": "e34fab64a3011d2176c99545a93d5cbddc9a91b7"},
    {"name": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
     "revision": "79f2382ceacceacdf38563d7c5d16b9ff8d725d6"},
    {"name": "google/muril-base-cased", "revision": "afd9f36c7923d54e97903922ff1b260d091d202f"},
    {"name": "ai4bharat/IndicBERTv2-MLM-Sam-TLM", "revision": "605bfff4a100307f97197c899e6e28c366b1f6c4"},
    {"name": "ai4bharat/IndicBERTv2-MLM-only", "revision": "51eb711a4c4b29949298b3f46971b5e1a0fd963f"},
]
model_idx = 0

model_name, model_revision = models[model_idx]["name"], models[model_idx]["revision"]
model = mteb.get_model(model_name, revision=model_revision)

results = {}

temp_output_folder = "temp_results"
os.makedirs(temp_output_folder, exist_ok=True)
results_file = f"results/results-{model_name.split('/')[1]}.json"
if os.path.exists(results_file):
    with open(results_file) as fp:
        results = json.load(fp)

for task_type, lang_to_tasks_map in tasks.items():
    results[task_type] = results.get(task_type, {})
    for lang_pair, task_list in lang_to_tasks_map.items():
        results[task_type][lang_pair] = results[task_type].get(lang_pair, {})
        for task in task_list:
            if task.metadata.name in results[task_type][lang_pair]:
                continue
            evaluation = mteb.MTEB(tasks=[task])
            if task.metadata.name == "XQuADPlusRetrieval":
                eval_split = "validation"
            elif task.metadata.name in ["MIRACLReranking", "MIRACLRetrieval"]:
                eval_split = "dev"
            else:
                eval_split = "test"
            task_results = evaluation.run(model, output_folder=temp_output_folder, eval_splits=[eval_split],
                                          overwrite_results=True)
            results[task_type][lang_pair][task.metadata.name] = task_results[0].scores[eval_split][0]["main_score"]

            with open(results_file, "w") as fp:
                fp.write(json.dumps(results, indent=4))
