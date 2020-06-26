import pandas as pd

sentences = pd.read_csv('data/basil.csv', index_col=0).fillna('')

general_df = pd.DataFrame()
for f in [str(el) for el in range(1,11)]:
    df = pd.read_csv(f"data/dev_w_preds/{f}_dev_w_pred.csv", index_col=0)
    general_df = general_df.append(df)

print(len(general_df), len(sentences))
a = set([el.lower() for el in general_df.index])
b = set([el.lower() for el in sentences.index])
print(a-b)