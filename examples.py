# pip install sentence-transformers numpy

import numpy as np
from sentence_transformers import SentenceTransformer

model_name = "intfloat/multilingual-e5-base"  # "AkshitaS/Hinglish-embedding-base"
query_prefix, corpus_prefix = "query: ", "query: "
model = SentenceTransformer(model_name)


def get_similarity_scores(samples):
    for sample in samples:
        queries = [query_prefix + sample["query"]]
        corpus = [corpus_prefix + c for c in sample["corpus"]]
        query_embedding = model.encode(queries, convert_to_numpy=True, normalize_embeddings=True)[0]
        corpus_embeddings = model.encode(corpus, convert_to_numpy=True, normalize_embeddings=True)
        print("Query:", sample["query"])
        for i, x in enumerate(sample["corpus"]):
            sim_score = np.dot(query_embedding, corpus_embeddings[i])
            print("\t", np.round(sim_score, 4), ":", x)
        print("")


################################################################################################

print("CASE 1 : \n\t"
      "Here, there is an Hindi query sentence and the corpus contains : \n\t\t"
      "a. the query sentence translated to English,  \n\t\t"
      "b. a sentence in Hindi related to the query sentence,  \n\t\t"
      "c. the related sentence translated to English \n\t"
      "Ideal similarity score : a > b == c \n")

en_name, hi_name = ("Moolchand", "मूलचंद")
samples = [
    {
        "query": f"{hi_name} ने कानून की पढ़ाई की और ३० की उम्र में राजनीति से जुड़ गए",
        "corpus": [
            f"{en_name} studied law and became a politician at the age of 30.",
            f"{hi_name} का जन्म राजनीतिज्ञों के परिवार में हुआ था",
            f"{en_name} was born in a family of politicians",
        ]
    },
    {
        "query": f"{en_name} studied law and became a politician at the age of 30.",
        "corpus": [
            f"{hi_name} ने कानून की पढ़ाई की और ३० की उम्र में राजनीति से जुड़ गए",
            f"{en_name} was born in a family of politicians",
            f"{hi_name} का जन्म राजनीतिज्ञों के परिवार में हुआ था",

        ]
    }
]
get_similarity_scores(samples)

print("##########################################################################################\n")

print("CASE 2 : \n\t"
      "When both the query and corpus are in Romanized Hindi and the corpus contains: \n\t\t"
      "a. Very related sentence\n\t\t"
      "b. Slightly related sentence\n\t\t"
      "c. Unrelated sentence \n\t"
      "Ideal similarity score : a > b > c \n")

samples = [
    {
        "query": "Is baar diwali par main 15 din ke liye ghar ja rahi hoon",
        "corpus": [
            "october mein deepawali ki chhutiyan hai sabhi ke ",
            "Mere parivaar mein tyoharon pe devi puja ki parampara hai",
            "Pavan ne kanoon ki padhai ki aur 30 ki umar mein rajneeti se jud gaye",
        ]
    }
]
get_similarity_scores(samples)
