import pandas as pd
from lib.evaluate.Eval import my_eval


def got_quote(x):
    double_q = '"' in str(x)
    return double_q

source_df = pd.DataFrame(columns=['source', 'prec', 'rec', 'f1'])

pd.set_option('display.max_columns', 10)
pd.set_option('display.max_colwidth', 500)

average_ratio = 0
average_ratio_in_tp = 0
for f in [str(el) for el in range(1,11)]:
    df = pd.read_csv(f"data/dev_w_preds/{f}_dev_w_pred.csv", index_col=0)

    df['quote'] = df.sentence.apply(got_quote)
    df['tp'] = (df.label == 1) & (df.pred == 1)

    df_tp = df[df.tp == True]
    df_no_tp = df[df.tp == False]

    ratio = sum(df_no_tp.quote) / len(df_no_tp)
    ratio_in_tp = sum(df_tp.quote) / len(df_tp)
    average_ratio += ratio
    average_ratio_in_tp += ratio_in_tp

print(average_ratio / 10)
print(average_ratio_in_tp / 10)
