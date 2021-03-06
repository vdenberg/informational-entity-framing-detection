import pandas as pd
from lib.evaluate.Eval import my_eval
from lib.utils import standardise_id
import re
from collections import Counter
from lib.handle_data.ErrorAnalysis import ErrorAnalysis

pd.set_option('display.max_columns', 15)
pd.set_option('display.max_colwidth', 500)
pd.set_option('display.width', 200)

ea = ErrorAnalysis(models='base_best')

# ea.inf_bias_only()
top_e = ea.get_top_e()
print(top_e)

dfs = [ea.analyze_top_e(ea.w_preds, top_e, model, context) for model, context in ea.models]
entity_df = ea.concat_comparisons(dfs, only_rec=False)
print(entity_df.to_latex())


df = ea.add_e()
dfs = [ea.compare_subsets(ea.w_preds, 'tar_in_sent', model, context) for model, context in ea.models]
entity_df = ea.concat_comparisons(dfs, only_rec=True)
print(entity_df)

top_e = ea.get_top_e(n=10)
df = ea.add_top_e(ea.w_preds, top_e)
dfs = [ea.compare_subsets(df, 'tope_in_me', model, context) for model, context in ea.models]
entity_df = ea.concat_comparisons(dfs, only_rec=True)
print(entity_df)

