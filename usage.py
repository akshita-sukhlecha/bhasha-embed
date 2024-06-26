queries = [
    "मूलचंद ने कानून की पढ़ाई की और ३० की उम्र में राजनीति से जुड़ गए",
    "Moolchand studied law and became a politician at the age of 30.",
    "Moolchand ne kanoon ki padhai kari aur 30 ki umar mein rajneeti se jud gaye"
]
documents = [
    "मूलचंद ने कानून की पढ़ाई की और ३० की उम्र में राजनीति से जुड़ गए",
    "Moolchand studied law and became a politician at the age of 30.",
    "Moolchand ne kanoon ki padhai kari aur 30 ki umar mein rajneeti se jud gaye",
    "मूलचंद का जन्म राजनीतिज्ञों के परिवार में हुआ था",
    "Moolchand was born in a family of politicians",
    "Moolchand ka janm rajneetigyon ke parivar mein hua tha"
]

################### Using Sentence Transformers ###################

# pip install sentence-transformers

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("AkshitaS/Hinglish-embedding-base")
query_embeddings = model.encode(queries, normalize_embeddings=True)
document_embeddings = model.encode(documents, normalize_embeddings=True)

scores = (query_embeddings @ document_embeddings.T)
print(scores.shape)
# (3, 6)
print(scores)

########################## Using Transformers ###################

## pip install transformers torch

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
# model_id = "AkshitaS/Hinglish-embedding-base"
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModel.from_pretrained(model_id)
# input_texts = queries + documents
# batch_dict = tokenizer(input_texts, padding=True, truncation=True, return_tensors='pt')
# outputs = model(**batch_dict)
# embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
#
# embeddings = F.normalize(embeddings, p=2, dim=1)
# scores = (embeddings[:len(queries)] @ embeddings[len(queries):].T)
# print(scores)

