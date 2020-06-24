import pandas as pd
from lib.evaluate.Eval import my_eval
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

entity_df = pd.DataFrame(columns=['entity', 'frequency', 'w', 'prec', 'rec', 'f1'])
for f in [str(el) for el in range(1,11)]:
    df = pd.read_csv(f"data/dev_w_preds/{f}_dev_w_pred.csv")
    for e, c in top_e:
        w_top, wo_top = top_in_df(df, e)

        ent_mets, ent_perf = my_eval(w_top.label, w_top.dev, name=f, set_type='w')
        w_row = [e, c, 0, ent_mets['prec'], ent_mets['rec'], ent_mets['rec']]

        ent_mets, ent_perf = my_eval(wo_top.label, wo_top.dev, name=f, set_type='wo')
        wo_row = [e, c, 1, ent_mets['prec'], ent_mets['rec'], ent_mets['rec']]

        # rows = pd.DataFrame([w_row, wo_row], columns=['entity', 'frequency', 'w', 'prec', 'rec', 'f1'])
        rows = pd.DataFrame([w_row], columns=['entity', 'frequency', 'w', 'prec', 'rec', 'f1'])
        entity_df = entity_df.append(rows, ignore_index=True)

for n, gr in entity_df.groupby(['entity', 'w']):
    c = gr.frequency.values[0]
    test = gr[['prec', 'rec', 'f1']] * 100
    test = test.describe()
    test_m = test.loc['mean'].round(2).astype(str)
    test_std = test.loc['std'].round(2).astype(str)
    result = test_m + ' \pm ' + test_std

    result_values = ['$' + el + '$' for el in list(result.values)]
    reverse_label = 1 if n[1] == 0 else 0
    latex_row = [n[0], c, reverse_label] + result_values
    latex_row = [str(el) for el in latex_row]
    latex_row = ' & '.join(latex_row) + ' \\\\'
    print(latex_row)