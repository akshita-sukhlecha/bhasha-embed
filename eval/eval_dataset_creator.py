import os

import datasets
import pandas as pd
import numpy as np
from indictrans import Transliterator

# git clone https://github.com/libindic/indic-trans.git
# cd indic-trans && pip install -r requirements.txt
# cd indic-trans && pip install .

trn = Transliterator(source="hin", target="eng", build_lookup=True)


def create_hin_latn_data_for_quad_format(hin_deva_data):
    hin_latn_data = []
    hin_deva_to_latn_contexts = {}
    hin_deva_to_latn_questions = {}
    for context in set(hin_deva_data["context"]):
        hin_deva_to_latn_contexts[context] = trn.transform(context)
    for question in set(hin_deva_data["question"]):
        hin_deva_to_latn_questions[question] = trn.transform(question)

    for idx, row in enumerate(hin_deva_data):
        trans_question = hin_deva_to_latn_questions[row["question"]]
        trans_context_raw = hin_deva_to_latn_contexts[row["context"]]
        trans_words = [w for w in trans_context_raw.split(" ") if w != ""]
        trans_context = " ".join(trans_words)

        # Answer text generation assumes that the number of words are the same in hin_deva and hin_latn contexts
        trans_answer_start = []
        trans_answer_text = []
        for ans_start, ans_text in zip(row["answers"]["answer_start"], row["answers"]["text"]):
            context_prior = row["context"][:ans_start]
            num_prior_words = len([w for w in context_prior.split(" ") if w != ""])
            # if ans_start is not at a new word
            if " " not in row["context"][ans_start - 1: ans_start + 1]:
                num_prior_words = max(0, num_prior_words - 1)
            num_ans_words = len([w for w in ans_text.split(" ") if w != ""])

            trans_ans_text = " ".join(trans_words[num_prior_words: num_prior_words + num_ans_words])
            trans_ans_start = len(" ".join(trans_words[:num_prior_words]))
            trans_answer_text.append(trans_ans_text)
            trans_answer_start.append(trans_ans_start + 1 if trans_ans_start else trans_ans_start)

        trans_row = {
            "context": trans_context,
            "question": trans_question,
            "answers": {
                "answer_start": np.array(trans_answer_start).astype(np.int32),
                "text": np.array(trans_answer_text).astype(object)
            },
            "id": row["id"]
        }
        hin_latn_data.append(trans_row)
    hin_latn_df = pd.DataFrame(hin_latn_data)
    return hin_latn_df


def load_mlqa_dataset():
    # fields:
    #   context : passage (duplicate across rows)
    #   question : query (few duplicates across rows)
    #   answers : { "answer_start": [], "text": [] }
    #   id : unique field for mapping across languages
    # An id may not be present across all languages. So, total ids/rows for each language is different.
    dataset = {
        "path": "facebook/mlqa",
        "revision": "397ed406c1a7902140303e7faf60fff35b58d285",
    }
    splits = ["validation", "test"]
    datapath = "./data/mlqa_plus"
    langs = {
        "ara_Arab": "mlqa.ar.ar",
        "deu_Latn": "mlqa.de.de",
        "eng_Latn": "mlqa.en.en",
        "hin_Deva": "mlqa.hi.hi",
        "spa_Latn": "mlqa.es.es",
        "vie_Latn": "mlqa.vi.vi",
        "zho_Hans": "mlqa.zh.zh"
    }

    for lang, hf_subset in langs.items():
        lang_path = f"{datapath}/{lang}/"
        os.makedirs(lang_path, exist_ok=True)
        for split in splits:
            lang_data = datasets.load_dataset(**dataset, name=hf_subset, split=split)
            # lang_data.to_csv(f"{datapath}/{lang}/{split}.csv")
            lang_data.to_parquet(f"{datapath}/{lang}/{split}.parquet")

    os.makedirs(f"{datapath}/hin_Latn", exist_ok=True)
    for split in splits:
        hin_deva_data = datasets.load_dataset(**dataset, name=langs["hin_Deva"], split=split)
        hin_latn_df = create_hin_latn_data_for_quad_format(hin_deva_data)
        hin_latn_df = hin_latn_df[["id", "context", "question", "answers"]]
        # hin_latn_df.to_csv(f"{datapath}/hin_Latn/{split}.csv", index=False)
        hin_latn_df.to_parquet(f"{datapath}/hin_Latn/{split}.parquet", index=False)


def load_xquad_dataset():
    # fields:
    #   context : passage (duplicate across rows)
    #   question : query (duplicate across rows)
    #   answers : { "answer_start": [], "text": [] }
    #   id : unique field for mapping across languages
    # Total number of ids/rows for any language = 1.19k
    dataset = {
        "path": "google/xquad",
        "revision": "51adfef1c1287aab1d2d91b5bead9bcfb9c68583"
    }
    splits = ["validation"]
    datapath = "./data/xquad_plus"

    langs = {
        "ara_Arab": "xquad.ar",
        "deu_Latn": "xquad.de",
        "ell_Grek": "xquad.el",
        "eng_Latn": "xquad.en",
        "hin_Deva": "xquad.hi",
        "spa_Latn": "xquad.es",
        "ron_Latn": "xquad.ro",
        "rus_Cyrl": "xquad.ru",
        "tha_Thai": "xquad.th",
        "tur_Latn": "xquad.tr",
        "vie_Latn": "xquad.vi",
        "zho_Hans": "xquad.zh"
    }
    for lang, hf_subset in langs.items():
        lang_path = f"{datapath}/{lang}/"
        os.makedirs(lang_path, exist_ok=True)
        for split in splits:
            lang_data = datasets.load_dataset(**dataset, name=hf_subset, split=split)
            # lang_data.to_csv(f"{datapath}/{lang}/{split}.csv")
            lang_data.to_parquet(f"{datapath}/{lang}/{split}.parquet")

    os.makedirs(f"{datapath}/hin_Latn", exist_ok=True)
    for split in splits:
        hin_deva_data = datasets.load_dataset(**dataset, name=langs["hin_Deva"], split=split)
        hin_latn_df = create_hin_latn_data_for_quad_format(hin_deva_data)
        hin_latn_df = hin_latn_df[["id", "context", "question", "answers"]]
        # hin_latn_df.to_csv(f"{datapath}/hin_Latn/{split}.csv", index=False)
        hin_latn_df.to_parquet(f"{datapath}/hin_Latn/{split}.parquet", index=False)


def load_semrel_dataset():
    # fields: sentence1, sentence2, label
    dataset = {
        "path": "SemRel/SemRel2024",
        "revision": "ef5c383d1b87eb8feccde3dfb7f95e42b1b050dd"
    }
    splits = ["test", "dev"]
    datapath = "./data/semrel_plus"

    langs = {
        "afr_Latn": "afr",
        "amh_Ethi": "amh",
        "arb_Arab": "arb",
        "arq_Arab": "arq",
        "ary_Arab": "ary",
        "eng_Latn": "eng",
        "hau_Latn": "hau",
        "hin_Deva": "hin",
        "ind_Latn": "ind",
        "kin_Latn": "kin",
        "mar_Deva": "mar",
        "pan_Guru": "pan",
        "spa_Latn": "esp",
        "tel_Telu": "tel"
    }

    for lang, hf_subset in langs.items():
        lang_path = f"{datapath}/{lang}/"
        os.makedirs(lang_path, exist_ok=True)
        for split in splits:
            lang_data = datasets.load_dataset(**dataset, name=hf_subset, split=split)
            # lang_data.to_csv(f"{datapath}/{lang}/{split}.csv")
            lang_data.to_parquet(f"{datapath}/{lang}/{split}.parquet")

    os.makedirs(f"{datapath}/hin_Latn", exist_ok=True)
    for split in splits:
        hin_deva_data = datasets.load_dataset(**dataset, name=langs["hin_Deva"], split=split)
        hin_latn_data = []
        for row in hin_deva_data:
            trans_row = {
                "sentence1": trn.transform(row["sentence1"]),
                "sentence2": trn.transform(row["sentence2"]),
                "label": row["label"]
            }
            hin_latn_data.append(trans_row)
        hin_latn_df = pd.DataFrame(hin_latn_data)
        # hin_latn_df.to_csv(f"{datapath}/hin_Latn/{split}.csv", index=False)
        hin_latn_df.to_parquet(f"{datapath}/hin_Latn/{split}.parquet", index=False)


# def load_hin_mix_data():
#     # fields: text
#     dataset = {
#         "path": "kartikagg98/HINMIX_hi-en",
#         "revision": "5735b2d0737980b03c50ee643ad8ac31da5d8531"
#     }
#     splits = ["test"]
#     datapath = "./data/hin_mix"
#
#     langs = {
#         "eng_Latn": "lcsalign-en",
#         "hin_Deva": "lcsalign-hi",
#         "hin_Deva_cm": "lcsalign-hicm",   # has code-mixed words (not sentences)
#         "hin_Latn_cm": "lcsalign-hicmrom"
#     }
#     for lang, hf_subset in langs.items():
#         split_path = f"{datapath}/{lang}/"
#         os.makedirs(split_path, exist_ok=True)
#         for split in splits:
#             lang_data = datasets.load_dataset(**dataset, name=hf_subset, split=split)
#             # lang_data.to_csv(f"{datapath}/{lang}/{split}.csv")
#             lang_data.to_parquet(f"{datapath}/{lang}/{split}.parquet")
#


if __name__ == "__main__":
    # Transliterate hindi datasets to create hin_latn datasets
    load_mlqa_dataset()
    load_xquad_dataset()
    load_semrel_dataset()
