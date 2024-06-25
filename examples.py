# pip install sentence-transformers numpy

import numpy as np
from sentence_transformers import SentenceTransformer

model_id = "AkshitaS/Hinglish-embedding-base"
model = SentenceTransformer(model_id)


def get_similarity_scores(samples):
    for sample in samples:
        query_embedding = model.encode([sample["query"]], convert_to_numpy=True, normalize_embeddings=True)[0]
        corpus_embeddings = model.encode(sample["corpus"], convert_to_numpy=True, normalize_embeddings=True)
        print("Query:", sample["query"])
        for i, x in enumerate(sample["corpus"]):
            sim_score = np.dot(query_embedding, corpus_embeddings[i])
            print("\t", np.round(sim_score, 4), ":", x)
        print("")


################################################################################################
print("CASE 1 : "
      "a. When the query and corpus are in different languages (Hindi/English)"
      "b. The corpus has both English and Hindi extracts")
print("Here, there is an english query sentence and the corpus contains : "
      "1. the query sentence translated to Hindi, "
      "2. a sentence in English related to the query sentence, "
      "3. the related sentence translated to Hindi")
print("Ideally, the similarity score of corpus sentence wrt query should be : 1 > 2 == 3 ")
print("But many models give higher similarity score if the extract has the same language as the query. \n")

en_name, hi_name = ("Moolchand", "मूलचंद")
samples = [
    {
        "query": f"{en_name} studied law and became a politician at the age of 30.",
        "corpus": [
            f"{en_name} was born in a family of politicians",
            f"{hi_name} का जन्म राजनीतिज्ञों के परिवार में हुआ था",
            f"{hi_name} ने कानून की पढ़ाई की और ३० की उम्र में राजनीति से जुड़ गए",
        ]
    },
    {
        "query": f"{hi_name} ने कानून की पढ़ाई की और ३० की उम्र में राजनीति से जुड़ गए",
        "corpus": [
            f"{en_name} was born in a family of politicians",
            f"{hi_name} का जन्म राजनीतिज्ञों के परिवार में हुआ था",
            f"{en_name} studied law and became a politician at the age of 30."
        ]
    }
]

get_similarity_scores(samples)

################################################################################################

print("\n\n")
print("CASE 2 : When both the query and corpus are in Romanized Hindi")
print("Most embedding models do not understand Romanized Hindi and perform bad on such texts.\n")

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
