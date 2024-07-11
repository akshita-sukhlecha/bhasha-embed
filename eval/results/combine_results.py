import json
import pandas as pd

all_results = {}
base_path = "./"
models = ["bhasha-embed-v0", "paraphrase-multilingual-mpnet-base-v2", "LaBSE", "multilingual-e5-base",
          "multilingual-e5-large", "muril-base-cased", "IndicBERTv2-MLM-only"]
for model_name in models:
    file = "results-" + model_name + ".json"
    with open(base_path + file) as fp:
        results = json.load(fp)
    all_results[model_name] = results

norm_results = []
for task_type, x in all_results['bhasha-embed-v0'].items():
    for lang_set, y in x.items():
        for task_name in y.keys():
            row = {
                'task_type': task_type,
                'lang_set': lang_set,
                'task_name': task_name
            }
            for model_name in all_results.keys():
                if task_type in all_results[model_name] and lang_set in all_results[model_name][task_type] and \
                        task_name in all_results[model_name][task_type][lang_set]:
                    row[model_name] = all_results[model_name][task_type][lang_set][task_name]
            norm_results.append(row)
pd.DataFrame(norm_results).to_csv("all_results.csv", index=False)
