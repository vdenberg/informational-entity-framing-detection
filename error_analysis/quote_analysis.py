import pandas as pd
from lib.evaluate.Eval import my_eval
from lib.handle_data.ErrorAnalysis import ErrorAnalysis
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_colwidth', 500)

models2compare = [('cam+', 'article'), ('rob', 'none')]
ea = ErrorAnalysis(models2compare)
df = ea.w_preds

for model, context in models2compare: # , ('cam+', 'story'), ('cam++', 'story'),
    print()
    print(model, context)

    rate_of_quotes = sum(df.quote) /len(df)
    print(ea.N, rate_of_quotes)

    df_w_conf_mat = ea.conf_mat(df, model, context)

    for el in ['tp', 'fp', 'tn', 'fn']:
        subdf = df[df[el]]
        prop = sum(subdf.quote) / len(subdf)
        print(el, len(subdf), '\%' + str(round(prop*100,2)))
