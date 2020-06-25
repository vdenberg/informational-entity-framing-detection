import pandas as pd
from lib.evaluate.Eval import my_eval

sentences = pd.read_csv('data/basil.csv', index_col=0).fillna('')
source_df = pd.DataFrame(columns=['source', 'prec', 'rec', 'f1'])
for f in [str(el) for el in range(1,11)]:
    df = pd.read_csv(f"data/dev_w_preds/{f}_dev_w_pred.csv")

    # ANALYZE BY SOURCE
    for n, gr in df.groupby('source'):
        source_mets, source_perf = my_eval(gr.label, gr.pred, name=n, set_type='dev')

        row = [n, source_mets['prec'], source_mets['rec'], source_mets['f1']]
        rows = pd.DataFrame([row], columns=['source', 'prec', 'rec', 'f1'])

    source_df = source_df.append(rows, ignore_index=True)

for n, gr in source_df.groupby("source"):
    test = gr[['prec', 'rec', 'f1']] * 100
    test = test.describe()
    test_m = test.loc['mean'].round(2).astype(str)
    test_std = test.loc['std'].round(2).astype(str)
    result = test_m + ' \pm ' + test_std
    print(f"\nResults of {n} on test:")
    print(result)
