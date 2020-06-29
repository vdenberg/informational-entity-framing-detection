import pandas as pd
from lib.evaluate.Eval import my_eval
from lib.utils import standardise_id

sentences = pd.read_csv('data/basil.csv', index_col=0).fillna('')
print(sentences.columns)

pd.set_option('display.max_columns', 10)
pd.set_option('display.max_colwidth', 500)

#sentences = '58fox62', '52fox18', '47nyt19', '46fox24', '48fox19'
for model, context in [('cam+', 'article'), ('cam+', 'story'), ('cam++', 'story'), ('rob', 'none')]:
    print()
    print(model, context)

    df = pd.DataFrame()
    for f in [str(el) for el in range(1,11)]:
        subdf = pd.read_csv(f"data/dev_w_preds/dev_w_{model}_{context}_preds/{f}_dev_w_pred.csv", index_col=0)
        df = df.append(subdf)

    if model == 'rob':
        sentences.index = [standardise_id(el) for el in sentences.index]
        df.index = [standardise_id(el) for el in df.index]
        df['label'] = sentences.loc[df.index].bias
        df['source'] = [el.lower() for el in sentences.loc[df.index].source]

    columns = ['source', 'size', 'size_bias', 'prec', 'rec', 'f1']

    source_df = pd.DataFrame(columns=columns)


    general_mets, general_perf = my_eval(df.label, df.pred, name='all')
    print(general_perf)
    biased = df[df.label == 1]

    row = ['&&All', len(df), len(biased), general_mets['prec'], general_mets['rec'], general_mets['f1']]
    rows = pd.DataFrame([row], columns=columns)
    source_df = source_df.append(rows, ignore_index=True)

    # ANALYZE BY SOURCE
    for n, gr in df.groupby('source'):
        source_mets, source_perf = my_eval(gr.label, gr.pred, name=n)
        biased = gr[gr.label == 1]

        row = [n, len(gr), len(biased), source_mets['prec'], source_mets['rec'], source_mets['f1']]
        rows = pd.DataFrame([row], columns=columns)
        source_df = source_df.append(rows, ignore_index=True)

    source_df[['prec', 'rec', 'f1']] = round(source_df[['prec', 'rec', 'f1']] * 100, 2)
    source_df = source_df.set_index('source')
    source_df[['size', 'size_bias']] = source_df[['size', 'size_bias']].astype(int)
    source_df = source_df.rename(index={'fox': 'FOX', 'hpo': '&&HPO', 'nyt': '&&NYT'})

    print(source_df.loc[['FOX', '&&NYT', '&&HPO', '&&All']].to_latex())
