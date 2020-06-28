import pandas as pd
from lib.evaluate.Eval import my_eval
from lib.utils import standardise_id
import re
from collections import Counter


def extract_e(string):
    string = re.sub("[\'\[\]]", "", string)
    l = string.split(', ')
    return l


def flatten_top_e(top_e):
    ents = []
    for e in top_e:
        ents.extend(extract_e(e))
    return ents


def get_top_me(me, n):
    ents = flatten_top_e(me)
    ents = Counter(ents).most_common(n)
    return ents


def top_in_row(x, top_e):
    if isinstance(top_e, str):
        top_e = [top_e]
    top_e = set(top_e)
    row_e = set(extract_e(x))
    diff = top_e.intersection(row_e)
    return bool(diff)


def top_in_df(df, top_e):
    df['got_top'] = df.main_entities.apply(lambda x: top_in_row(x, top_e))
    w_top = df[df.got_top == True]
    wo_top = df[df.got_top == False]
    return w_top, wo_top

sentences = pd.read_csv('data/basil.csv', index_col=0).fillna('')
top_e = get_top_me(sentences.main_entities.values, 5)
print(top_e)

for model, context in [('cam+', 'article'), ('cam+', 'story'), ('cam++', 'story'), ('rob', 'none')]:
    print()
    print(model, context)

    df = pd.DataFrame()
    for f in [str(el) for el in range(1, 11)]:
        subdf = pd.read_csv(f"data/dev_w_preds/dev_w_{model}_{context}_preds/{f}_dev_w_pred.csv", index_col=0)
        df = df.append(subdf)

    if model == 'rob':
        sentences.index = [standardise_id(el) for el in sentences.index]
        df.index = [standardise_id(el) for el in df.index]
        df['label'] = sentences.loc[df.index].lex_bias
        df['source'] = sentences.loc[df.index].source
        df['main_entities'] = sentences.loc[df.index].main_entities
        df['pred'] = df.preds

    entity_df = pd.DataFrame(columns=['entity', '#sent', '#biased_sent', 'w', 'prec', 'rec', 'f1'])
    for e, c in top_e:
        w_top, wo_top = top_in_df(df, e)

        nr_w_biased = len(w_top[w_top.label == 1])

        ent_mets, ent_perf = my_eval(w_top.label, w_top.pred, name=f, set_type='w')
        w_row = [e, c, nr_w_biased, 1, ent_mets['prec'], ent_mets['rec'], ent_mets['f1']]

        ent_mets, ent_perf = my_eval(wo_top.label, wo_top.pred, name=f, set_type='wo')
        wo_row = [e, c, nr_w_biased, 0, ent_mets['prec'], ent_mets['rec'], ent_mets['f1']]

        # rows = pd.DataFrame([w_row, wo_row], columns=['entity', 'frequency', 'w', 'prec', 'rec', 'f1'])
        rows = pd.DataFrame([w_row], columns=['entity', '#sent', '#biased_sent', 'w', 'prec', 'rec', 'f1'])
        entity_df = entity_df.append(rows, ignore_index=True)


    entity_df[['prec', 'rec', 'f1']] = round(entity_df[['prec', 'rec', 'f1']] * 100,2)
    entity_df = entity_df.sort_values('f1')
    print(entity_df)

    # result_values = ['$' + el + '$' for el in list(result.values)]