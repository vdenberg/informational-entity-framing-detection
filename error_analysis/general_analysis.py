import pandas as pd

sentences = pd.read_csv('data/basil.csv', index_col=0).fillna('')

general_df = pd.DataFrame()
for f in [str(el) for el in range(1,11)]:
    df = pd.read_csv(f"data/dev_w_preds/{f}_dev_w_pred.csv", index_col=0)
    general_df = general_df.append(df)
    print(df.columns)
    print(sentences.columns)

general_df = general_df.sort_index()
general_df['lex_bias'] = sentences['lex_bias'].astype(int)
general_df = general_df.rename(columns={'label': 'inf_bias'})
general_df = general_df[['story', 'source', 'sentence', 'lex_bias', 'inf_bias', 'pred']]
general_df.to_csv('error_analysis/for_katja.csv')

