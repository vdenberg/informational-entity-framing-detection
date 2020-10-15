import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

if __name__ == '__main__':
    BASIL = '../../data/basil_w_features.csv'
    basil = pd.read_csv(BASIL, index_col=0).fillna('')

    BASIL_ERR = '../../data/basil_devtest_w_errors.csv'
    basil_err = pd.read_csv(BASIL_ERR, index_col=0).fillna('')
    basil['error'] = 'train'
    basil.loc[basil_err.index, 'error'] = basil_err.error
    basil.loc[basil.error != 'train', 'error'] = 'test bert ' + basil.loc[basil.error != 'train', 'error']

    basil['y_pred'] = 'n/a'
    basil.loc[basil.error == 'test bert tp', 'y_pred'] = 1
    basil.loc[basil.error == 'test bert fp', 'y_pred'] = 1
    basil.loc[basil.error == 'test bert fn', 'y_pred'] = 0
    basil.loc[basil.error == 'test bert tn', 'y_pred'] = 0

    basil_train = basil[basil.y_pred == 'n/a']
    print(basil_train.shape)
    basil_devtest = basil[basil.y_pred != 'n/a']
    print(basil_devtest.shape)
    y_true = basil_devtest.bias.astype(int).values
    y_pred = basil_devtest.y_pred.astype(int).values

    eval = precision_recall_fscore_support(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(eval)
    print('TN: {}, FP: {}, FN: {}, TP: {}'.format(tn, fp, fn, tp))
    print('TN: {}, FP: {}, FN: {}, TP: {}'.format(tn, fp, fn, tp))

    lexindata = pd.DataFrame(index=['all', 'all w/o inf', 'all w/ inf', 'train', 'train w/o inf', 'train w/ inf', 'test', 'test w/o inf', 'test w/ inf', 'test bert n', 'test bert p', 'test bert tn', 'test bert tp', 'test bert fn', 'test bert fp'], columns=['#sent', '#lex', '%'])

    for n, gr in basil.groupby('error'):
        lexindata.loc[n,'#sent'] = len(gr)
        lexindata.loc[n,'#lex'] = sum(gr.lex_bias)

    lexindata.loc['all', '#sent'] = len(basil)
    lexindata.loc['all', '#lex'] = sum(basil.lex_bias)

    tmp = basil[basil.bias == 0]
    lexindata.loc['all w/o inf', '#sent'] = len(tmp)
    lexindata.loc['all w/o inf', '#lex'] = sum(tmp.lex_bias)

    tmp = basil[basil.bias == 1]
    lexindata.loc['all w/ inf', '#sent'] = len(tmp)
    lexindata.loc['all w/ inf', '#lex'] = sum(tmp.lex_bias)

    tmp = basil[(basil.error == 'train') & (basil.bias == 0)]
    lexindata.loc['train w/o inf', '#sent'] = len(tmp)
    lexindata.loc['train w/o inf', '#lex'] = sum(tmp.lex_bias)

    tmp = basil[(basil.error == 'train') & (basil.bias == 1)]
    lexindata.loc['train w/ inf', '#sent'] = len(tmp)
    lexindata.loc['train w/ inf', '#lex'] = sum(tmp.lex_bias)

    tmp = basil[basil.error != 'train']
    lexindata.loc['test','#sent'] = len(tmp)
    lexindata.loc['test','#lex'] = sum(tmp.lex_bias)

    tmp = basil[(basil.error != 'train') & (basil.bias == 0)]
    lexindata.loc['test w/o inf', '#sent'] = len(tmp)
    lexindata.loc['test w/o inf', '#lex'] = sum(tmp.lex_bias)

    tmp = basil[(basil.error != 'train') & (basil.bias == 1)]
    lexindata.loc['test w/ inf', '#sent'] = len(tmp)
    lexindata.loc['test w/ inf', '#lex'] = sum(tmp.lex_bias)

    tmp = basil[(basil.y_pred == 1)]
    lexindata.loc['test bert p', '#sent'] = len(tmp)
    lexindata.loc['test bert p', '#lex'] = sum(tmp.lex_bias)

    tmp = basil[(basil.y_pred == 0)]
    lexindata.loc['test bert n', '#sent'] = len(tmp)
    lexindata.loc['test bert n', '#lex'] = sum(tmp.lex_bias)

    lexindata['%'] = lexindata['#lex']/lexindata['#sent']*100

    tmp = basil[(basil.y_pred != 'n/a')]
    tmp.y_pred = tmp.y_pred.astype(int)
    c = tmp[['lex_bias', 'y_pred']].corr()
    print(c)

    tmp = basil[basil.error.isin(['test bert fn','test bert fp'])]
    tmp.y_pred = tmp.y_pred.astype(int)
    c = tmp[['lex_bias', 'y_pred']].corr()
    print(c)

    print(lexindata)
    print(sum(basil.bias), round(sum(basil.bias)/len(basil), 2))



