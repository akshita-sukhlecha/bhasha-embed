from typing import Dict, List


def get_lang_pairs(languages) -> Dict[str, List[str]]:
    # Returns all possible language pairs
    lang_pairs = {}
    for x in languages:
        for y in languages:
            pair = f"{x}-{y}"
            lang_pairs[pair] = [x.replace("_", "-"), y.replace("_", "-")]
    return lang_pairs


def get_lang_sets(languages) -> Dict[str, List[str]]:
    # Returns all possible language triplets
    lang_sets = {}
    languages = sorted(languages)
    for idx, x in enumerate(languages):
        for y in languages[idx + 1:]:
            for z in languages:
                lang_set = f"{x}-{y}-{z}"
                lang_sets[lang_set] = [x.replace("_", "-"), y.replace("_", "-"), z.replace("_", "-")]
    for z in languages:
        all_langs = [l.replace("_", "-") for l in languages]
        all_langs = all_langs + [z.replace("_", "-")]
        lang_set = "-".join(languages) + "-" + z
        lang_sets[lang_set] = all_langs
    return lang_sets


def get_cross_lingual_retrieval_data_old(ds_dict, lang_pairs, splits,
                                         context_field="context", query_field="question", id_field="id"):
    # ds_dict : DatasetDict[<lang_pair>, DatasetDict[<split>, Dataset]]
    # Dataset : features: [<context_field>, <query_field>, <id_field>]
    queries = {lang_pair: {split: {} for split in splits} for lang_pair in lang_pairs}
    corpus = {lang_pair: {split: {} for split in splits} for lang_pair in lang_pairs}
    relevant_docs = {lang_pair: {split: {} for split in splits} for lang_pair in lang_pairs}

    for lang_pair in lang_pairs:
        langs = lang_pair.split("-")
        lang_corpus, lang_query = langs[0], langs[1]

        for split in splits:
            ds_query = ds_dict[lang_query][split]
            ds_context = ds_dict[lang_corpus][split]
            query_to_id = {query: f"Q{i}" for i, query in enumerate(set(ds_query[query_field]))}
            context_to_id = {context: f"C{i}" for i, context in enumerate(set(ds_context[context_field]))}
            row_to_context = {row[id_field]: row[context_field] for row in ds_context}

            for row in ds_query:
                if row[id_field] not in row_to_context:
                    continue
                query = row[query_field]
                query_id = query_to_id[query]
                context = row_to_context[row[id_field]]
                context_id = context_to_id[context]

                queries[lang_pair][split][query_id] = query
                corpus[lang_pair][split][context_id] = {"title": "", "text": context}
                if query_id not in relevant_docs[lang_pair][split]:
                    relevant_docs[lang_pair][split][query_id] = {}
                relevant_docs[lang_pair][split][query_id][context_id] = 1

    return queries, corpus, relevant_docs


def get_cross_lingual_retrieval_data(ds_dict, lang_sets, splits,
                                     context_field="context", query_field="question", id_field="id"):
    # ds_dict : DatasetDict[<lang>, DatasetDict[<split>, Dataset]]
    #           Dataset : features: [<context_field>, <query_field>, <id_field>]
    # lang_sets: <corpus_lang_1>-<corpus_lang_2>-<corpus_lang_3>-<query_lang>
    queries = {lang_set: {split: {} for split in splits} for lang_set in lang_sets}
    corpus = {lang_set: {split: {} for split in splits} for lang_set in lang_sets}
    relevant_docs = {lang_set: {split: {} for split in splits} for lang_set in lang_sets}

    for lang_set in lang_sets:
        langs = lang_set.split("-")
        lang_query = langs[-1]
        langs_corpus = langs[:-1] if len(langs) > 1 else langs
        langs_corpus = list(set(langs_corpus))

        for split in splits:
            ds_query = ds_dict[lang_query][split]
            query_to_id = {query: f"Q{i}" for i, query in enumerate(set(ds_query[query_field]))}
            row_to_context_ids = {row[id_field]: [] for row in ds_query}

            for idx, lang_corpus in enumerate(langs_corpus):
                ds_context = ds_dict[lang_corpus][split]
                context_to_id = {context: f"{idx}C{i}" for i, context in enumerate(set(ds_context[context_field]))}
                for row in ds_context:
                    if row[id_field] in row_to_context_ids:
                        context = row[context_field]
                        context_id = context_to_id[context]
                        row_to_context_ids[row[id_field]].append(context_id)
                        corpus[lang_set][split][context_id] = {"title": "", "text": context}

            for row in ds_query:
                context_ids = row_to_context_ids[row[id_field]]
                if not context_ids:
                    continue
                query = row[query_field]
                query_id = query_to_id[query]
                queries[lang_set][split][query_id] = query

                if query_id not in relevant_docs[lang_set][split]:
                    relevant_docs[lang_set][split][query_id] = {}
                for context_id in context_ids:
                    relevant_docs[lang_set][split][query_id][context_id] = 1

    return queries, corpus, relevant_docs
