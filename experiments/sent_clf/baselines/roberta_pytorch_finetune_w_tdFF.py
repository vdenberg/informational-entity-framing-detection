from __future__ import absolute_import, division, print_function
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers.configuration_roberta import RobertaConfig
import pickle
from lib.classifiers.RobertaWrapper import RobertaForSequenceClassification, RobertaForSequenceClassificationWithTDFF, Inferencer, save_model, load_features
from datetime import datetime
from torch.nn import CrossEntropyLoss
import torch
import os, sys, random, argparse
import numpy as np
from lib.handle_data.PreprocessForBert import *
from lib.utils import get_torch_device
import time
from pprint import pprint
import logging

#######
# FROM:
# https://medium.com/swlh/how-twitter-users-turned-bullied-quaden-bayles-into-a-scammer-b14cb10e998a?source=post_recirc---------1------------------
#####


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, my_id, input_ids, input_mask, segment_ids, label_id):
        self.my_id = my_id
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


################
# HYPERPARAMETERS
################

parser = argparse.ArgumentParser()
# TRAINING PARAMS
parser.add_argument('-model', '--model', type=str, default='rob_base') #2,3,4
parser.add_argument('-ep', '--n_epochs', type=int, default=10) #2,3,4
parser.add_argument('-load', '--load', action='store_true', default=True)
parser.add_argument('-sampler', '--sampler', type=str, default='random')

parser.add_argument('-lr', '--lr', type=float, default=None) #5e-5, 3e-5, 2e-5
parser.add_argument('-bs', '--bs', type=int, default=None) #16, 21
parser.add_argument('-sv', '--sv', type=int, default=None) #16, 21
parser.add_argument('-fold', '--fold', type=str, default=None) #16, 21
args = parser.parse_args()

seeds = [args.sv] if args.sv else [0, 0, 0]
bss = [args.bs] if args.bs else [32, 24, 16]
lrs = [args.lr] if args.lr else [2e-5, 1e-5, 3e-5]
folds = [args.fold] if args.fold else ['1', '2', '3', '4', '5']
samplers = [args.sampler] if args.sampler else ['sequential', 'random']

# find GPU if present
model_mapping = {'rob_base': 'roberta-base',
                 'rob_dapt': 'experiments/adapt_dapt_tapt/pretrained_models/news_roberta_base',
                 'rob_tapt': 'experiments/adapt_dapt_tapt/pretrained_models/dsp_roberta_base_tapt_hyperpartisan_news_5015',
                 'rob_dapttapt': 'experiments/adapt_dapt_tapt/pretrained_models/dsp_roberta_base_dapt_news_tapt_hyperpartisan_news_5015',
                 }
ROBERTA_MODEL = model_mapping[args.model]
device, USE_CUDA = get_torch_device()
SAMPLER = args.sampler
N_EPS = args.n_epochs
LEARNING_RATE = args.lr
BATCH_SIZE = args.bs


########################
# SET HYPERPARAMETERS
########################

GRADIENT_ACCUMULATION_STEPS = 1
WARMUP_PROPORTION = 0.1
NUM_LABELS = 2
PRINT_EVERY = 100

########################
# WHERE ARE THE FILES
########################

TASK_NAME = 'SCtdFF'
FEAT_DIR = f'data/sent_clf/features_for_roberta/'
CHECKPOINT_DIR = f'models/checkpoints/{TASK_NAME}/'
REPORTS_DIR = f'reports/{TASK_NAME}'
TABLE_DIR = os.path.join(REPORTS_DIR, 'tables')
CACHE_DIR = 'models/cache/'  # This is where BERT will look for pre-trained models to load parameters from.

if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)
if not os.path.exists(REPORTS_DIR):
    os.makedirs(REPORTS_DIR)
if not os.path.exists(TABLE_DIR):
    os.makedirs(TABLE_DIR)


inferencer = Inferencer(REPORTS_DIR, logger, device, use_cuda=USE_CUDA)
table_columns = 'model,seed,bs,lr,model_loc,fold,epoch,set_type,rep_sim,loss,fn,fp,tn,tp,acc,prec,rec,f1'
main_results_table = pd.DataFrame(columns=table_columns.split(','))

if __name__ == '__main__':
    # set logger
    now = datetime.now()
    now_string = now.strftime(format=f'%b-%d-%Hh-%-M_{TASK_NAME}')
    LOG_NAME = f"{REPORTS_DIR}/{now_string}.log"
    console_hdlr = logging.StreamHandler(sys.stdout)
    file_hdlr = logging.FileHandler(filename=LOG_NAME)
    logging.basicConfig(level=logging.INFO, handlers=[console_hdlr, file_hdlr])
    logger = logging.getLogger()
    logger.info(args)

    for SAMPLER in samplers:
        for SEED in seeds:
            if SEED == 0:
                SEED_VAL = random.randint(0, 300)
            else:
                SEED_VAL = SEED

            seed_name = f"{args.model}_{SAMPLER}_{SEED_VAL}"
            random.seed(SEED_VAL)
            np.random.seed(SEED_VAL)
            torch.manual_seed(SEED_VAL)
            torch.cuda.manual_seed_all(SEED_VAL)

            for BATCH_SIZE in bss:
                bs_name = seed_name + f"_bs{BATCH_SIZE}"
                for LEARNING_RATE in lrs:
                    setting_name = bs_name + f"_lr{LEARNING_RATE}"
                    setting_results_table = pd.DataFrame(columns=table_columns.split(','))
                    for fold_name in folds:
                        fold_results_table = pd.DataFrame(columns=table_columns.split(','))
                        name = setting_name + f"_f{fold_name}"

                        best_val_res = {'model': args.model, 'seed': SEED_VAL, 'fold': fold_name, 'bs': BATCH_SIZE, 'lr': LEARNING_RATE, 'set_type': 'dev',
                                        'f1': 0, 'model_loc': ''}
                        test_res = {'model': args.model, 'seed': SEED_VAL, 'fold': fold_name, 'bs': BATCH_SIZE, 'lr': LEARNING_RATE, 'set_type': 'test'}

                        train_fp = f"data/sent_clf/features_for_roberta/{fold_name}_train_features.pkl"
                        dev_fp = f"data/sent_clf/features_for_roberta/{fold_name}_dev_features.pkl"
                        test_fp = f"data/sent_clf/features_for_roberta/{fold_name}_test_features.pkl"
                        _, train_batches, train_labels = load_features(train_fp, BATCH_SIZE, SAMPLER)
                        _, dev_batches, dev_labels = load_features(dev_fp, BATCH_SIZE, SAMPLER)
                        _, test_batches, test_labels = load_features(test_fp, BATCH_SIZE, SAMPLER)

                        logger.info(f"***** Training on Fold {fold_name} *****")
                        logger.info(f"  Details: {best_val_res}")
                        logger.info(f"  Logging to {LOG_NAME}")

                        model = RobertaForSequenceClassificationWithTDFF.from_pretrained(ROBERTA_MODEL, cache_dir=CACHE_DIR, num_labels=NUM_LABELS,
                                                                                 output_hidden_states=False, output_attentions=False)
                        model.to(device)
                        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01, eps=1e-6)  # To reproduce BertAdam specific behavior set correct_bias=False

                        n_train_batches = len(train_batches)
                        half_train_batches = int(n_train_batches / 2)
                        GRADIENT_ACCUMULATION_STEPS = 2
                        WARMUP_PROPORTION = 0.06
                        num_tr_opt_steps = n_train_batches * N_EPS / GRADIENT_ACCUMULATION_STEPS
                        num_tr_warmup_steps = int(WARMUP_PROPORTION * num_tr_opt_steps)
                        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_tr_warmup_steps, num_training_steps=num_tr_opt_steps)

                        model.train()

                        for ep in range(1, N_EPS + 1):
                            epoch_name = name + f"_ep{ep}"

                            if args.load and os.path.exists(os.path.join(CHECKPOINT_DIR, epoch_name)):
                                # this epoch for this setting has been trained before already
                                trained_model = RobertaForSequenceClassificationWithTDFF.from_pretrained(os.path.join(CHECKPOINT_DIR, epoch_name),
                                                                                                num_labels=NUM_LABELS,
                                                                                                output_hidden_states=False,
                                                                                                output_attentions=False)
                                dev_mets, dev_perf = inferencer.evaluate(trained_model, dev_batches, dev_labels,
                                                                         set_type='dev', name=epoch_name)
                            else:
                                tr_loss = 0
                                for step, batch in enumerate(train_batches):
                                    batch = tuple(t.to(device) for t in batch)

                                    model.zero_grad()
                                    outputs = model(batch[0], batch[1], labels=batch[2])
                                    #(loss), logits, probs, sequence_output = outputs
                                    (loss), logits, sequence_output = outputs

                                    loss.backward()
                                    tr_loss += loss.item()
                                    optimizer.step()
                                    scheduler.step()

                                    if step % PRINT_EVERY == 0 and step != 0:
                                        logging.info(f' Ep {ep} / {N_EPS} - {step} / {len(train_batches)} - Loss: {loss.item()}')

                                av_loss = tr_loss / len(train_batches)
                                save_model(model, CHECKPOINT_DIR, epoch_name)
                                dev_mets, dev_perf = inferencer.evaluate(model, dev_batches, dev_labels, av_loss=av_loss,
                                                                         set_type='dev', name=epoch_name)

                            # check if best
                            high_score = ''
                            if dev_mets['f1'] > best_val_res['f1']:
                                best_val_res.update(dev_mets)
                                best_val_res.update({'model_loc': os.path.join(CHECKPOINT_DIR, epoch_name)})
                                high_score = '(HIGH SCORE)'

                            logger.info(f'{epoch_name}: {dev_perf} {high_score}')

                        # load best model, save embeddings, print performance on test
                        if best_val_res['model_loc'] == '':
                            # none of the epochs performed above f1 = 0, so just use last epoch
                            best_val_res['model_loc'] = os.path.join(CHECKPOINT_DIR, epoch_name)
                        best_model = RobertaForSequenceClassificationWithTDFF.from_pretrained(best_val_res['model_loc'], num_labels=NUM_LABELS,
                                                                                   output_hidden_states=False,
                                                                                   output_attentions=False)
                        logger.info(f"***** (Embeds and) Test - Fold {fold_name} *****")
                        logger.info(f"  Details: {best_val_res}")

                        '''
                        for EMB_TYPE in ['poolbert', 'avbert']:
                            all_ids, all_batches, all_labels = load_features('data/features_for_bert/all_features.pkl', batch_size=1)
                            embs = inferencer.predict(model, all_batches, return_embeddings=True, emb_type=EMB_TYPE)
                            basil_w_BERT = pd.DataFrame(index=all_ids)
                            basil_w_BERT[EMB_TYPE] = embs
                            emb_name = f'{name}_basil_w_{EMB_TYPE}'
                            basil_w_BERT.to_csv(f'data/{emb_name}.csv')
                            logger.info(f'Written embs ({len(embs)},{len(embs[0])}) to data/{emb_name}.csv')
                        '''

                        # eval on test

                        test_mets, test_perf = inferencer.evaluate(best_model, test_batches, test_labels, set_type='test',
                                                                   name='best_model_loc')
                        logging.info(f"{test_perf}")
                        test_res.update(test_mets)

                        # store performance in table

                        fold_results_table = fold_results_table.append(best_val_res, ignore_index=True)
                        fold_results_table = fold_results_table.append(test_res, ignore_index=True)
                        setting_results_table = setting_results_table.append(fold_results_table)

                        # print result on fold

                        logging.info(
                            f'Fold {fold_name} results: \n{fold_results_table[["model", "seed", "bs", "lr", "fold", "set_type", "f1"]]}')

                        # print result of setting

                    logging.info(
                        f'Setting {setting_name} results: \n{setting_results_table[["model", "seed", "bs", "lr", "fold", "set_type", "f1"]]}')

                    # store performance of setting

                    main_results_table = main_results_table.append(setting_results_table, ignore_index=True)

                    # write performance to file

                    setting_results_table.to_csv(os.path.join(TABLE_DIR, f'{setting_name}_results_table.csv'), index=False)

                main_results_table.to_csv(os.path.join(TABLE_DIR, f'task_results_table.csv'), index=False)

'''
n_train_batches = len(train_batches)
half_train_batches = int(n_train_batches / 2)
num_tr_opt_steps = n_train_batches * NUM_TRAIN_EPOCHS  # / GRADIENT_ACCUMULATION_STEPS
num_tr_warmup_steps = int(WARMUP_PROPORTION * num_tr_opt_steps)
#scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_tr_warmup_steps, num_training_steps=num_tr_opt_steps)
'''