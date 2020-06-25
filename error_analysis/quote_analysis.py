import pandas as pd
from lib.evaluate.Eval import my_eval


def got_quote(x):
    double_q = '"' in x
    return double_q

sentences = pd.read_csv('data/basil.csv', index_col=0).fillna('')
source_df = pd.DataFrame(columns=['source', 'prec', 'rec', 'f1'])
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_colwidth', 500)

for f in [str(el) for el in range(1,11)]:
    df = pd.read_csv(f"data/dev_w_preds/{f}_dev_w_pred.csv", index_col=0)
    df['sentence'] = sentences.loc[df.index].sentence
    df['quote'] = df.sentence.apply(got_quote)

    # ANALYZE BY SOURCE
    tp = df[(df.label == 1) & (df.dev == 1)]
    fp = df[(df.label == 0) & (df.dev == 1)]
    tn = df[(df.label == 0) & (df.dev == 0)]
    fn = df[(df.label == 1) & (df.dev == 0)]

    for n, el in zip(['tp', 'fp', 'tn', 'fn'], [tp, fp, tn, fn]):
        tot = len(el)
        s = sum(el.quote)
        print(n, s/tot)
        print()