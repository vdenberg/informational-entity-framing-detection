import pandas as pd
from lib.evaluate.Eval import my_eval
from lib.utils import standardise_id
from lib.handle_data.ErrorAnalysis import ErrorAnalysis
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_colwidth', 500)
pd.set_option('display.width', 200)

ea = ErrorAnalysis(models='base_best')

source_dfs = [ea.compare_subsets(ea.w_preds, 'source', model, context) for model, context in ea.models]
source_df = ea.concat_comparisons(source_dfs)

print(source_df.to_latex())
ea.inf_bias_only()

pol_df = ea.clean_for_pol_analysis()
pol_sample = ea.sample_sentences(pol_df, which='inf_pol')
for n, s in pol_sample:
    print(n, s)

pol_dfs = [ea.compare_subsets(pol_df, 'inf_pol', model, context) for model, context in ea.models]
pol_df = ea.concat_comparisons(pol_dfs, only_rec=True)

print(pol_df.to_latex())

dir_df = ea.clean_for_dir_analysis()
dir_dfs = [ea.compare_subsets(dir_df, 'inf_dir', model, context) for model, context in ea.models]
dir_df = ea.concat_comparisons(dir_dfs, only_rec=True)
print(dir_df.to_latex())



