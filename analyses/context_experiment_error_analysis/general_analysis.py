import pandas as pd

sentences = pd.read_csv('data/basil.csv', index_col=0).fillna('')
sentences.index = [el.lower() for el in sentences.index]

general_df = pd.DataFrame()
model = 'cam+'
for f in [str(el) for el in range(1,11)]:
    df = pd.read_csv(f"data/dev_w_{model}_preds/{f}_dev_w_pred.csv", index_col=0)
    general_df = general_df.append(df)

general_df['sentence_ids'] = sentences.loc[general_df.index, 'uniq_idx.1']
general_df['sent_idx'] = sentences.loc[general_df.index, 'sent_idx']

general_df = general_df.sort_values(by=['sentence_ids', 'sent_idx'])
general_df['lex_bias'] = sentences['lex_bias'].astype(int)
general_df = general_df.rename(columns={'label': 'inf_bias'})
general_df = general_df[['story', 'source', 'sent_idx', 'sentence', 'lex_bias', 'inf_bias', 'pred']]
general_df.to_csv('error_analysis/cam+_preds_for_katja.csv')

