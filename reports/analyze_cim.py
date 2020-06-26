import pandas as pd
from pprint import pprint


def clean_mean(df, grby='', set_type=''):
    mets = ['acc', 'prec', 'rec', 'f1']
    if set_type:
        tmp_df = df[df.set_type == set_type]
    else:
        tmp_df = df
    return tmp_df.groupby(grby)[mets].mean().round(2)


fp1 = f'reports/cam/cam_+*#_exp_more_epochs.csv'
fp = f'reports/cam/story/cam++_base_34_h1200_bs32_lr0.001.csv'
#fp = f'reports/cam/cross4bert_article/cam+_base_local.csv'
#fp = f'reports/cam/cam_input_exp.csv'
fps = [fp]

for fp in fps:
    pd.options.display.max_columns = 999
    pd.set_option('display.max_rows',500)

    orig_df = pd.read_csv(fp, index_col=False)
    orig_df = orig_df.fillna('')
    mets = ['acc', 'prec', 'rec', 'f1']
    orig_df[mets] = orig_df[mets].round(4) * 100
    df = orig_df

    # look at how the model did overall
    if 'input' not in df.columns:
        df['input'] = 'c4'
    #df['model'] = df.model.apply(lambda x: x.split('_')[0])

    df = df[(df.input == 'c4')]
    def_means = clean_mean(df, grby=['model', 'seed',  'fold'], set_type='test')
    print(def_means)

    # get baseline results
    for n, gr in df[df.set_type == 'test'].groupby("model"):
        test = gr[['prec', 'rec', 'f1']]
        test = test.describe()
        test_m = test.loc['mean'].round(2).astype(str)
        test_std = test.loc['std'].round(2).astype(str)
        result = test_m + ' \pm ' + test_std
        print(f"\nResults of {n} on test:")
        print(result)
    '''
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
    '''
'''
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
#models/checkpoints/bert_baseline/bert263_f2_warmup_bs8_ep6
'''