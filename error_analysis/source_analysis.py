import pandas as pd
from lib.evaluate.Eval import my_eval

for model, context in [('cam+', 'story'), ('cam++', 'story')]:
    print()
    print(model)

    sentences = pd.read_csv('data/basil.csv', index_col=0).fillna('')
    source_df = pd.DataFrame(columns=['source', 'prec', 'rec', 'f1'])
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.max_colwidth', 500)

    general_df = pd.DataFrame()
    for f in [str(el) for el in range(1,11)]:
        df = pd.read_csv(f"data/dev_w_preds/dev_w_{model}_{context}_preds/{f}_dev_w_pred.csv", index_col=0)
        general_df = general_df.append(df)

    columns = ['source', 'size', 'size_bias', 'prec', 'rec', 'f1']

    general_mets, general_perf = my_eval(general_df.label, general_df.pred, name='all')
    biased = general_df[general_df.label == 1]

    row = ['All', len(general_df), len(biased), general_mets['prec'], general_mets['rec'], general_mets['f1']]
    rows = pd.DataFrame([row], columns=columns)
    source_df = source_df.append(rows, ignore_index=True)

    # ANALYZE BY SOURCE
    for n, gr in general_df.groupby('source'):
        source_mets, source_perf = my_eval(gr.label, gr.pred, name=n)
        biased = gr[gr.label == 1]

        row = [n, len(gr), len(biased), source_mets['prec'], source_mets['rec'], source_mets['f1']]
        rows = pd.DataFrame([row], columns=columns)
        source_df = source_df.append(rows, ignore_index=True)

    source_df[['prec', 'rec', 'f1']] = round(source_df[['prec', 'rec', 'f1']] * 100, 2)
    source_df = source_df.set_index('source')
    source_df[['size', 'size_bias']] = source_df[['size', 'size_bias']].astype(int)
    source_df = source_df.rename(index={'fox': 'FOX', 'hpo': 'HPO', 'nyt': 'NYT'})
    print(source_df.loc[['FOX', 'NYT', 'HPO', 'All']].to_latex())
