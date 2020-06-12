import pandas as pd
from pprint import pprint


def clean_mean(df, grby='', set_type=''):
    mets = ['acc', 'prec', 'rec', 'f1']
    if set_type:
        tmp_df = df[df.set_type == set_type]
    else:
        tmp_df = df
    return tmp_df.groupby(grby)[mets].mean().round(2)

task = 'SC_BERT'
fp = f'reports/SCvsSSC/{task}/tables/task_results_table.csv'
orig_df = pd.read_csv(fp, index_col=False)

mets = ['acc', 'prec', 'rec', 'f1']
orig_df[mets] = orig_df[mets].round(4) * 100

# look at how the model did overall
#pd.set_option('display.max_rows',500)
#print(clean_mean(grby=['seed', 'lr', 'bs', 'fold'], set_type='dev'))

# look at learning rate and batch size
print("\nImpact of learning rate and batch size:")
df = orig_df #[orig_df.model == 'bert']
#df['fan'] = df.fold == 'fan'
print(clean_mean(df, grby=['lr', 'bs'], set_type='dev'))
# isolate best setting
#df = df[(orig_df.lr == 0.00002) & (orig_df.bs == 21)]
#df = df[(orig_df.lr == 0.00001) & (orig_df.bs == 1)]

# get baseline results
test = df[df.set_type == 'test'][['prec', 'rec', 'f1']]
test = test.describe()
test_m = test.loc['mean'].round(2).astype(str)
test_std = test.loc['std'].round(2).astype(str)
result = test_m + ' \pm ' + test_std
print("\nBaseline results on test:")
print(result)

# get embed models
# look at seed
print("\nPick best seed to use to produce embeds: ")
print(clean_mean(df, grby='seed', set_type='dev'))
df = df[(df.seed == 231)]

# get model locs of best models
dev = df[df.set_type == 'dev']
model_locs = {str(f): (loc, f1) for f, loc, f1 in zip(dev.fold, dev.model_loc, dev.f1)}
pprint(model_locs)

# compare with cnm
print("\nCheck whether embeds are being used correctly:")
cnm = orig_df[orig_df.model == 'cnm']
cnm = cnm[cnm.lr == 0.00002]
print(clean_mean(cnm, grby='set_type'))

exit(0)
print("\nCompare to CAM:")
cam = orig_df[orig_df.model == 'cam']

# check how grid search did
print(clean_mean(cam, grby=['h', 'lr', 'bs', 'set_type']))
#print(clean_mean(cam, grby=['h', 'fold', 'lr', 'bs', 'set_type']))
# pick preferred setting
cam = cam[(cam.h == 250) & (cam.bs == 32) & (cam.lr == 0.005)]

# get cam results
test = cam[cam.set_type == 'test'][['prec', 'rec', 'f1']]
test = test.describe()
test_m = test.loc['mean'].round(2).astype(str)
test_std = test.loc['std'].round(2).astype(str)
result = test_m + ' \pm ' + test_std
print("\nCAM results on test:")
print(result)

#models/checkpoints/bert_baseline/bert263_f2_warmup_bs8_ep7
#models/checkpoints/bert_baseline/bert263_f2_warmup_bs8_ep6 !!