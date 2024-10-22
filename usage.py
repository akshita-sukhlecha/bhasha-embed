queries = [
    "प्रणव ने कानून की पढ़ाई की और ३० की उम्र में राजनीति से जुड़ गए",
    "Pranav studied law and became a politician at the age of 30.",
    "Pranav ne kanoon ki padhai kari aur 30 ki umar mein rajneeti se jud gaye"
]
documents = [
    "प्रणव ने कानून की पढ़ाई की और ३० की उम्र में राजनीति से जुड़ गए",
    "Pranav studied law and became a politician at the age of 30.",
    "Pranav ne kanoon ki padhai kari aur 30 ki umar mein rajneeti se jud gaye",
    "प्रणव का जन्म राजनीतिज्ञों के परिवार में हुआ था",
    "Pranav was born in a family of politicians",
    "Pranav ka janm rajneetigyon ke parivar mein hua tha"
]

################### Using Sentence Transformers ###################

# pip install sentence-transformers numpy

import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("AkshitaS/bhasha-embed-v0")

query_embeddings = model.encode(queries, normalize_embeddings=True)
document_embeddings = model.encode(documents, normalize_embeddings=True)

similarity_matrix = (query_embeddings @ document_embeddings.T)
print(similarity_matrix.shape)
# (3, 6)
print(np.round(similarity_matrix, 2))
#[[1.00  0.97  0.97  0.92  0.90  0.91]
# [0.97  1.00  0.96  0.90  0.91  0.91]
# [0.97  0.96  1.00  0.89  0.90  0.92]]

########################## Using Transformers ###################

## pip install transformers torch numpy

# import numpy as np
# from torch import Tensor
# import torch.nn.functional as F
# from transformers import AutoTokenizer, AutoModel
#
#
# def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
#     last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
#     return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
#
#
# model_id = "AkshitaS/bhasha-embed-v0"
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModel.from_pretrained(model_id)
#
# input_texts = queries + documents
# batch_dict = tokenizer(input_texts, padding=True, truncation=True, return_tensors='pt')
# outputs = model(**batch_dict)
# embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
#
# embeddings = F.normalize(embeddings, p=2, dim=1)
# similarity_matrix = (embeddings[:len(queries)] @ embeddings[len(queries):].T).detach().numpy()
# print(similarity_matrix.shape)
# # (3, 6)
# print(np.round(similarity_matrix, 2))
# #[[1.00  0.97  0.97  0.92  0.90  0.91]
# # [0.97  1.00  0.96  0.90  0.91  0.91]
# # [0.97  0.96  1.00  0.89  0.90  0.92]]


