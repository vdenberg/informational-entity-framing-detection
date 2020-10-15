import pandas as pd
from lib.evaluate.Eval import my_eval
from lib.handle_data.ErrorAnalysis import ErrorAnalysis
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 300)
pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.width', 2000)

ea = ErrorAnalysis('base_best')

sentlen_dfs = [ea.compare_subsets(ea.w_preds, 'len', model, context) for model, context in ea.models]
sentlen_df = ea.concat_comparisons(sentlen_dfs)
sentlen_df.index = ["0-90", "91-137", "138-192", "193-647", "All"]
print(sentlen_df.to_latex())

#ea.no_bias_only()
df = ea.w_preds

#short = df[(df.source != 'nyt') & (df.rob_22 == 1)][df.len == '0-90']
#long = df[(df.source == 'nyt') & (df.rob_22 == 0) & (df.cim_coverage == 1)][df.len == '193-647']

directed = df[(df.source == 'hpo') & (df.story == 53)][df.len == '0-90']
directed = df[((df.story == 21))]

#print(short.sample(20)[['source', 'sentence', 'bias', 'rob_22', 'cim_coverage']])
#print(long.sample(20)[['source', 'sentence', 'bias']])
print(directed[['source', 'sentence', 'bias', 'lex_bias']])
