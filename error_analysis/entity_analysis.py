import pandas as pd
from lib.evaluate.Eval import my_eval
from lib.utils import standardise_id, collect_preds
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


def add_top_in_df(df, e):
    df['e_in_art'] = df.main_entities.apply(lambda x: top_in_row(x, e))
    # w_top = df[df.got_top == True]
    # wo_top = df[df.got_top == False]
    return df


def process_rob_labs(x):
    if isinstance(x, str):
        return x[8]
    else:
        return 0


def add_top_in_sent(df, e):
    global surface_e
    s = surface_e[e]
    df['e_in_sent'] = df.sentence.apply(lambda x: (s in x) if isinstance(x, str) else False)
    # w_top = df[df.got_top == True]
    # wo_top = df[df.got_top == False]
    return df


def add_top_e(df, e):
    df = add_top_in_df(df, e)
    df = add_top_in_sent(df, e)
    return df


pd.set_option('display.max_columns', 15)
pd.set_option('display.max_colwidth', 500)
pd.set_option('display.width', 200)

sentences = pd.read_csv('data/basil.csv', index_col=0).fillna('')
sentences.main_entities = sentences.main_entities.apply(lambda x: re.sub('Lawmakers', 'lawmakers', x))

top_e = get_top_me(sentences.main_entities.values, 3)
print(top_e)

surface_e = {}
surface_mapping = {'Republican lawmakers': 'Republican', 'Democratic lawmakers': 'Democrat', 'Republicans': 'Republican',
                   'House Democrats': 'Democrat'}
for el, c in top_e:
    if el not in ['Republican lawmakers', 'Democratic lawmakers', 'House Democrats', 'Republicans']:
        surface = el.split(' ')[-1]
    else:
        surface = surface_mapping[el]
    surface_e[el] = surface

for model, context in [('cam+', 'article'), ('cam+', 'story'), ('rob', 'none')]: # , ('cam+', 'story'), ('cam++', 'story'),
    print()
    print(model, context)

    preds_df = collect_preds(model, context, sentences=sentences)

    entity_df = pd.DataFrame(columns=['entity', '#sent', 'mention', '#sent2', '#biased_sent', 'prec', 'rec', 'f1'])
    for e, nr_sent in top_e:
        df = add_top_e(preds_df, e)

        in_art_nor_sent = df[(df.e_in_art == 0) & (df.e_in_sent == 0)]

        in_art = df[(df.e_in_art == 1)]
        in_art_only = df[(df.e_in_art == 1) & (df.e_in_sent == 0)]
        in_target_too = df[(df.e_in_art == 1) & (df.e_in_sent == 1)]

        nr_biased = len(in_art[in_art.label == 1])

        for n, gr in [('in target', in_target_too), ('in art only', in_art_only)]:
            mets, perf = my_eval(gr.label, gr.pred, name=n, set_type='w')
            w_row = [e, nr_sent, n, nr_biased, len(gr), mets['prec'], mets['rec'], mets['f1']]
            rows = pd.DataFrame([w_row], columns=['entity', '#sent', 'mention', '#sent2', '#biased_sent', 'prec', 'rec', 'f1'])
            entity_df = entity_df.append(rows, ignore_index=True)

    entity_df[['prec', 'rec', 'f1']] = round(entity_df[['prec', 'rec', 'f1']] * 100,2)
    # entity_df = entity_df.sort_values('#sent')
    for n, gr in entity_df.groupby('entity'):
        print(gr)
    # result_values = ['$' + el + '$' for el in list(result.values)]
