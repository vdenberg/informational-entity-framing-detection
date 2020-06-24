import pandas as pd

for f in [str(el) for el in range(1,11)]:
    df = pd.read_csv(f"data/dev_w_preds/{f}_dev_w_pred.csv")
    me = df.main_entities.value_counts()
    ie = df.inf_entities.value_counts()
    print(me)
    print(ie)