import numpy as np
from sentence_transformers import SentenceTransformer

# pip install sentence-transformers numpy

model_name = "AkshitaS/bhasha-embed-v0"  # "intfloat/multilingual-e5-base"
query_prefix, corpus_prefix = "", ""  # "query: ", "query: "  or "passage: "
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

print("Example 1 : Cross-lingual alignment \n")
en_name, hi_name = ("Pranav", "प्रणव")
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

print("Example 2 : Understanding of Romanized Hindi \n")
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

print("##########################################################################################\n")

print("Example 3 : Passage retrieval from multi-lingual corpus \n")
samples = [
    {
        "query": "ताजमहल का निर्माण किसने करवाया और क्यों?",
        "corpus": [
            "ताजमहल का निर्माण मुगल सम्राट शाहजहाँ ने करवाया था। उन्होंने अपनी पत्नी मुमताज़ महल की याद में यह अद्भुत मकबरा बनवाया। यह प्रेम का प्रतीक माना जाता है और इसकी सुंदरता विश्व प्रसिद्ध है।",
            "The Taj Mahal was commissioned by Mughal Emperor Shah Jahan. He built this magnificent mausoleum in memory of his wife Mumtaz Mahal. It is considered a symbol of love and is renowned for its beauty worldwide.",
            "Tajmahal ka nirmaan mughal samraat shaahjahaan ne karwaya tha. unhonne apani patni mumtaaz mahal kee yaad mein yah adbhut makabara banavaaya. yah prem ka prateek maana jaata hai aur iski sundarta vishv prasiddh hai.",
            "ताजमहल आगरा, भारत में स्थित है और हर साल लाखों पर्यटक इसे देखने आते हैं। अधिकतर पर्यटक यहाँ अक्टूबर, नवंबर एवं फरवरी के महीनों में आते हैं। ",
            "Taj Mahal is located in Agra, India and is visited by millions of tourists every year. Most of the tourists come here in the months of October, November and February.",
            "Taj mahal aagra, bhaarat mein sthit hai aur har saal laakhon paryatak ise dekhane aate hain. adhiktar paryatak yahaan october, november aur february ke mahino mein aate hain."
        ]
    }
]
get_similarity_scores(samples)

print("##########################################################################################\n")

print("Example 4 : Understanding of Code-mixed text \n")
samples = [
    {
        "query": "ख़राब चादर",
        "corpus": [
            "नमस्ते, मैंने आपकी कंपनी से एक चादर खरीदी थी। लेकिन इसमें एक समस्या है। यह बहुत जल्दी फट गई है। मैंने इसे बस एक महीने पहले ही खरीदा था। क्या आप कृपया मुझे इसे बदलने में मदद कर सकते हैं? मुझे उसके लिए क्या प्रक्रिया करनी होगी?",
            "Hello, I purchased a bedsheet from your company. But there's a problem with it. It has torn very quickly. I bought it just a month ago. Could you please help me exchange it for a new one? What process do I need to follow to do so?",
            "Hello, मैंने आपकी company से एक bedsheet खरीदी थी। But there's a problem with it. यह बहुत जल्दी फट गई है। मैंने इसे बस एक महीने पहले ही खरीदा था। Could you please help me exchange it for a new one? मुझे उसके लिए क्या process करना होगा?",
            "Hello, maine aapki company se ek bedsheet kharidi thi. But there's a problem with it. Yeh bahut jaldi fat gayi hae. Maine ise bas ek mahine pehle hi kharida tha. Could you please help me exchange it for a new one? Mujhe iske liye kya process karna hoga?",
            "मैं एक घड़ी वापस करना चाहता हूं जो मैंने पिछले सप्ताह खरीदी थी। यह काम नहीं कर रही है, मैंने अलग-अलग बैटरियां आज़माईं।",
            "I want to return the clock I bought last week. Its not working, I tried different batteries.",
        ]
    }
]
get_similarity_scores(samples)
