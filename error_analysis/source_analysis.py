import pandas as pd
from lib.evaluate.Eval import my_eval

for model in ['cam+', 'cam++']:
    print(model)
    sentences = pd.read_csv('data/basil.csv', index_col=0).fillna('')
    source_df = pd.DataFrame(columns=['source', 'prec', 'rec', 'f1'])
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.max_colwidth', 500)

    general_df = pd.DataFrame()
    for f in [str(el) for el in range(1,11)]:
        df = pd.read_csv(f"data/dev_w_{model}_preds/{f}_dev_w_pred.csv", index_col=0)
        general_df = general_df.append(df)

    general_mets, general_perf = my_eval(general_df.label, general_df.pred, name='all')
    row = ['all', general_mets['prec'], general_mets['rec'], general_mets['f1']]
    rows = pd.DataFrame([row], columns=['source', 'prec', 'rec', 'f1'])
    source_df = source_df.append(rows, ignore_index=True)

    # ANALYZE BY SOURCE
    for n, gr in general_df.groupby('source'):
        source_mets, source_perf = my_eval(gr.label, gr.pred, name=n)

        row = [n, source_mets['prec'], source_mets['rec'], source_mets['f1']]
        rows = pd.DataFrame([row], columns=['source', 'prec', 'rec', 'f1'])
        source_df = source_df.append(rows, ignore_index=True)

    source_df[['prec', 'rec', 'f1']] = round(source_df[['prec', 'rec', 'f1']] * 100, 2)
    source_df = source_df.set_index('source')
    print(source_df.loc[['fox', 'nyt', 'hpo', 'all']])
