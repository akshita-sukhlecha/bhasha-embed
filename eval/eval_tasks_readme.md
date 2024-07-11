### Tasks excluded from evaluation 

Some Hindi tasks have not been added to the evaluation because of one of the following reasons: 
- its dataset is very large and takes time to run.  
- it contains only train split
- its dataset is very easy or pertains to a specific domain
- encountered issue with mteb while running the task (mteb=1.12.50)  


Here are the excluded tasks along with reason : 
```python
from mteb.tasks import IndicQARetrieval, MIRACLRetrieval, WikipediaRetrievalMultilingual
from mteb.tasks import BibleNLPBitextMining, FloresBitextMining, IN22ConvBitextMining, IN22GenBitextMining, IndicGenBenchFloresBitextMining, NTREXBitextMining
from mteb.tasks import MIRACLReranking, XNLI
from mteb.tasks import MassiveIntentClassification, MassiveScenarioClassification, SentimentAnalysisHindi

tasks = {
    "retrieval": {
        "hin_Deva-hin_Deva": [
            IndicQARetrieval(hf_subsets=["hi"]),  # Dataset loading issue
            WikipediaRetrievalMultilingual(hf_subsets=["hi"]),  # Data loading Bug
            MIRACLRetrieval(hf_subsets=["hi"])  # too big
        ]
    },
    "bitext": {
        "eng_Latn-hin_Deva": [
            BibleNLPBitextMining(hf_subsets=["eng_Latn-hin_Deva"]),  # only bible
            FloresBitextMining(hf_subsets=["eng_Latn-hin_Deva"]),  # split=devtest, ?
            IN22ConvBitextMining(hf_subsets=["eng_Latn-hin_Deva"]),  # not working - issue no hf_subset
            IN22GenBitextMining(hf_subsets=["eng_Latn-hin_Deva"]),  # not working - issue no hf_subset
            IndicGenBenchFloresBitextMining(hf_subsets=["eng-hin"]),  # perf score = 1.0 for all models
            NTREXBitextMining(hf_subsets=["eng_Latn-hin_Deva"])  # not working
        ],
        "hin_Deva-eng_Latn": [
            BibleNLPBitextMining(hf_subsets=["hin_Deva-eng_Latn"]),  # only bible texts
            FloresBitextMining(hf_subsets=["hin_Deva-eng_Latn"]),
            # split=devtest, error: subset2langscripts[hf_subset] KeyError: 'default'
            IN22ConvBitextMining(hf_subsets=["hin_Deva-eng_Latn"]),  # not working - issue no hf_subset
            IN22GenBitextMining(hf_subsets=["hin_Deva-eng_Latn"]),  # not working - issue no hf_subset
            IndicGenBenchFloresBitextMining(hf_subsets=["hin-eng"]),  # perf score = 1.0 for all models
            NTREXBitextMining(hf_subsets=["hin_Deva-eng_Latn"]),  # not working
        ]
    },
    "classification": {
        "hin_Deva": [
            XNLI(hf_subsets=["hi"]),  # ?
            MassiveIntentClassification(hf_subsets=["hi"]),  # Too big
            MassiveScenarioClassification(hf_subsets=["hi"]),  # Too big
            SentimentAnalysisHindi(hf_subsets=["hin_Deva"])  # Only train subset
        ]
    },
    "ranking": {
        "hin_Deva": [
            MIRACLReranking(hf_subsets=["hi"]),
            # only dev, error in 1.12.75: RerankingEvaluator.py all_query_embs = np.asarray(...  can't convert cuda:0 device type tensor to numpy.
        ]
    }

}
```