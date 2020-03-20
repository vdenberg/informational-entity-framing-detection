# requirements:
# !pip install --upgrade tensorflow
# !pip install ktrain

###

import ktrain
from ktrain import text
from helpers import Split, evaluation
import pandas as pd
import time
import os
import argparse
import tensorflow as tf

###

GPU = True
DIR = '.'
EPOCHS = 10
BATCH_SIZE = 6
TEST = False
SPL = 'berg'
TRAIN = True
INFER = True

if GPU:
    device_name = tf.test.gpu_device_name()
    if device_name == '/device:GPU:0':
        print('Found GPU at: {}'.format(device_name))
    else:
        raise SystemError('GPU device not found')

BASIL = DIR + '/basil_w_features.csv'
if not os.path.exists(DIR + '/models'):
    os.mkdir(DIR + '/models')
if not os.path.exists(DIR + '/models/berg'):
    os.mkdir(DIR + '/models/berg')
if not os.path.exists(DIR + '/models/fan'):
    os.mkdir(DIR + '/models/fan')

assert os.path.exists(DIR + '/basil_w_features.json')
assert os.path.exists(DIR + '/split.json')
assert os.path.exists(DIR + '/test_tokens.txt')
assert os.path.exists(DIR + '/train_tokens.txt')
assert os.path.exists(DIR + '/test_tokens.txt')

PRED = DIR + '/models'
if SPL == 'fan':
    PRED += '/fan'
if SPL == 'berg':
    PRED += '/berg'

basil = pd.read_csv(BASIL, index_col=0).fillna('')
basil.bias = basil.bias.astype(str)
spl = Split(basil, which=SPL, split_loc=DIR, tst=TEST)
folds = spl.apply_split(features=['sentence'], output_as='df')

t = text.Transformer('bert-base-uncased', maxlen=512, classes=['0', '1'])
model = t.get_classifier()

if TRAIN:
    for fold_i, fold in enumerate(folds):
        # for getting indices: fold['train'].index.values
        x_train, y_train = fold['train']['sentence'].to_list(), fold['train']['bias'].values
        x_dev, y_dev = fold['dev']['sentence'].to_list(), fold['dev']['bias'].values
        x_test, y_test = fold['dev']['sentence'].to_list(), fold['dev']['bias'].values

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
        predictor.save(PRED + '/fold{}'.format(fold_i))

        #print(fold_i, 'analyze losses')
        #learner.view_top_losses(n=10, preproc=t)

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
