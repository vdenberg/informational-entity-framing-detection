# requirements:
# !pip install --upgrade tensorflow
# !pip install ktrain

###

import ktrain
from ktrain import text
from lib.handle_data.SplitData import Split
from lib.evaluate.StandardEval import my_eval
import pandas as pd
import time
import os
import argparse
import tensorflow as tf

###

GPU = True
EPOCHS = 10
BATCH_SIZE = 16
TEST = False
SPL = 'fan'
TRAIN = True
INFER = True

device_name = tf.test.gpu_device_name()
if device_name == '/device:GPU:0':
    print('Found GPU at: {}'.format(device_name))
else:
    pass

DATA_FP = 'data/basil.csv'
MODEL_DIR = 'models/ktrain/'

basil = pd.read_csv(DATA_FP, index_col=0).fillna('')
basil['label'] = basil.bias.astype(str)
basil.index = [x.lower() for x in basil.index]
spl = Split(basil, which=SPL, tst=TEST)
folds = spl.apply_split(features=['sentence'])

t = text.Transformer('bert-base-uncased', maxlen=512, classes=['0', '1'])
model = t.get_classifier()

if TRAIN:
    for fold_i, fold in enumerate(folds):
        # for getting indices: fold['train'].index.values
        x_train, y_train = fold['train']['sentence'].to_list(), fold['train']['label'].values
        x_dev, y_dev = fold['dev']['sentence'].to_list(), fold['dev']['label'].values
        x_test, y_test = fold['dev']['sentence'].to_list(), fold['dev']['label'].values

        print(fold_i, 'preprocess')
        trn = t.preprocess_train(x_train, y_train)
        val = t.preprocess_test(x_dev, y_dev)

        print(fold_i, 'learn')
        start_time = time.time()
        learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=BATCH_SIZE)
        learner.fit_onecycle(2e-5, EPOCHS)
        print("--- %s seconds ---" % (time.time() - start_time))

        print(fold_i, 'validate')
        learner.validate(class_names=t.get_classes())

        print(fold_i, 'save predictor')
        predictor = ktrain.get_predictor(model, preproc=t)
        predictor.save(os.join.path(MODEL_DIR, f'predictor_{SPL}_fold{fold["name"]}'))

        """
        print(fold_i, 'analyze losses')
        losses = top_losses = learner.view_top_losses(n=10, preproc=t) # loss = [idx, loss, truth, pred]
        indices, losses, _, preds = zip(*losses)
        losses_df = fold['dev'].iloc[indices]
        losses_df['loss'] = losses
        losses_df['pred'] = preds

        predictor.explain(n=10, preproc=t)
        """

basil['error'] = ['not_in_dev'] * len(basil)

if INFER:
    neg_pos = []
    pos_neg = []
    pos_pos = []
    for fold_i, fold in enumerate(folds):
        x_train, y_train = fold['train']['sentence'].to_list(), fold['train']['bias'].values
        x_dev, y_dev = fold['dev']['sentence'].to_list(), fold['dev']['bias'].values
        x_test, y_test = fold['dev']['sentence'].to_list(), fold['dev']['bias'].values

        print(fold_i, 'load predictor')
        predictor = ktrain.load_predictor(PRED + '/fold{}'.format(fold_i))
        y_pred = predictor.predict(x_dev)

        # evaluate
        print(fold_i, 'evaluate')
        y_pred = [int(el) for el in y_pred]
        y_dev = [int(el) for el in y_dev]
        evalb, evalstr = evaluation(y_pred=y_pred, y_true=y_dev)
        print(fold_i, evalstr, end='\n\n')

        print(fold_i, 'sample')
        compare = pd.DataFrame(zip(y_pred, y_dev), index=fold['dev'].index.values, columns=['y_pred', 'y_dev'])
        basil.loc[fold['dev'].index.values, 'fold'] = fold_i
        t01 = compare[(compare['y_dev'] == 0) & (compare['y_pred'] == 1)]
        t11 = compare[(compare['y_dev'] == 1) & (compare['y_pred'] == 1)]
        t10 = compare[(compare['y_dev'] == 1) & (compare['y_pred'] == 0)]
        t00 = compare[(compare['y_dev'] == 0) & (compare['y_pred'] == 0)]
        basil.loc[t01.index, 'error'] = 'fp'
        basil.loc[t11.index, 'error'] = 'tp'
        basil.loc[t10.index, 'error'] = 'fn'
        basil.loc[t00.index, 'error'] = 'tn'
        # predictor.explain('')

basil_err = basil.loc[basil['error'] != 'not_in_dev']
basil_err[['error', 'fold', 'sentence', 'bias', 'lex_bias']].to_csv(DIR + '/basil_w_errors.csv')

pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 200)

print(basil_err.error.value_counts())
print(basil_err.head())


# predictor.explain('Jesus Christ is the central figure of Christianity.')
