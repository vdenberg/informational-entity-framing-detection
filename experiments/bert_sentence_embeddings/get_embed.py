from __future__ import absolute_import, division, print_function
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
import pickle
from lib.classifiers.BertForEmbed import Inferencer, save_model
from lib.classifiers.BertWrapper import BertForSequenceClassification, BertWrapper, load_features
from datetime import datetime
from torch.nn import CrossEntropyLoss
import torch
import os, sys, random, argparse
import numpy as np
from lib.handle_data.PreprocessForBert import *
from lib.utils import get_torch_device
import time
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
parser.add_argument('-ep', '--n_epochs', type=int, default=4) #2,3,4
parser.add_argument('-lr', '--learning_rate', type=float, default=2e-5) #5e-5, 3e-5, 2e-5
parser.add_argument('-bs', '--batch_size', type=int, default=24) #16, 21
parser.add_argument('-l', '--load_from_ep', type=int, default=0) #16, 21
args = parser.parse_args()

# find GPU if present
device, USE_CUDA = get_torch_device()
BERT_MODEL = 'bert-base-cased' #bert-large-cased
TASK_NAME = 'bert_baseline'
CHECKPOINT_DIR = f'models/checkpoints/{TASK_NAME}/'
REPORTS_DIR = f'reports/{TASK_NAME}'
CACHE_DIR = 'models/cache/' # This is where BERT will look for pre-trained models to load parameters from.

N_EPS = args.n_epochs
LEARNING_RATE = args.learning_rate
LOAD_FROM_EP = args.load_from_ep
BATCH_SIZE = args.batch_size
GRADIENT_ACCUMULATION_STEPS = 1
WARMUP_PROPORTION = 0.1
NUM_LABELS = 2
PRINT_EVERY = 100

inferencer = Inferencer(REPORTS_DIR, logger, device, use_cuda=USE_CUDA)
table_columns = 'model,seed,bs,lr,model_loc,fold,epoch,set_type,loss,acc,prec,rec,f1,fn,fp,tn,tp'
main_results_table = pd.DataFrame(columns=table_columns.split(','))

if __name__ == '__main__':
    # set logger
    now = datetime.now()
    now_string = now.strftime(format=f'%b-%d-%Hh-%-M_{TASK_NAME}_get_embed')
    LOG_NAME = f"{REPORTS_DIR}/{now_string}.log"
    console_hdlr = logging.StreamHandler(sys.stdout)
    file_hdlr = logging.FileHandler(filename=LOG_NAME)
    logging.basicConfig(level=logging.INFO, handlers=[console_hdlr, file_hdlr])
    logger = logging.getLogger()
    logger.info(args)

    model_locs = {'1': ('models/checkpoints/bert_baseline/bert_182_bs16_lr2e-05_f1_ep3', 41.63),
                     '2': ('models/checkpoints/bert_baseline/bert_182_bs16_lr2e-05_f2_ep4', 40.32),
                     '3': ('models/checkpoints/bert_baseline/bert_182_bs16_lr2e-05_f3_ep2', 47.339999999999996),
                     '4': ('models/checkpoints/bert_baseline/bert_182_bs16_lr2e-05_f4_ep1', 43.24),
                     '5': ('models/checkpoints/bert_baseline/bert_182_bs16_lr2e-05_f5_ep4', 25.77),
                     '6': ('models/checkpoints/bert_baseline/bert_182_bs16_lr2e-05_f6_ep4', 23.7),
                     '7': ('models/checkpoints/bert_baseline/bert_182_bs16_lr2e-05_f7_ep4', 33.67),
                     '8': ('models/checkpoints/bert_baseline/bert_182_bs16_lr2e-05_f8_ep3', 31.819999999999997),
                     '9': ('models/checkpoints/bert_baseline/bert_182_bs16_lr2e-05_f9_ep4', 36.5)}

    for SEED in [182]:
        if SEED == 0:
            SEED_VAL = random.randint(0, 300)
        else:
            SEED_VAL = SEED

        seed_name = f"bert_{SEED_VAL}"
        random.seed(SEED_VAL)
        np.random.seed(SEED_VAL)
        torch.manual_seed(SEED_VAL)
        torch.cuda.manual_seed_all(SEED_VAL)

        for BATCH_SIZE in [16]:
            bs_name = seed_name + f"_bs{BATCH_SIZE}"
            for LEARNING_RATE in [2e-5]:
                setting_name = bs_name + f"_lr{LEARNING_RATE}"
                setting_results_table = pd.DataFrame(columns=table_columns.split(','))
                for fold_name in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']:
                    fold_results_table = pd.DataFrame(columns=table_columns.split(','))
                    name = setting_name + f"_f{fold_name}"

                    best_val_res = {'model': 'bert', 'seed': SEED_VAL, 'fold': fold_name, 'bs': BATCH_SIZE, 'lr': LEARNING_RATE, 'set_type': 'dev',
                                    'f1': 0, 'model_loc': ''}
                    test_res = {'model': 'bert', 'seed': SEED_VAL, 'fold': fold_name, 'bs': BATCH_SIZE, 'lr': LEARNING_RATE, 'set_type': 'test'}

                    train_fp = f"data/features_for_bert/folds/{fold_name}_train_features.pkl"
                    dev_fp = f"data/features_for_bert/folds/{fold_name}_dev_features.pkl"
                    test_fp = f"data/features_for_bert/folds/{fold_name}_test_features.pkl"
                    _, train_batches, train_labels = load_features(train_fp, BATCH_SIZE)
                    _, dev_batches, dev_labels = load_features(dev_fp, BATCH_SIZE)
                    _, test_batches, test_labels = load_features(test_fp, BATCH_SIZE)

                    logger.info(f"***** Training on Fold {fold_name} *****")
                    logger.info(f"  Details: {best_val_res}")
                    logger.info(f"  Logging to {LOG_NAME}")

                    # get ep from best model loc
                    best_ep_model_loc, best_ep_dev_f1 = model_locs[fold_name]
                    best_ep = best_ep_model_loc[-1]

                    for ep in [best_ep]:
                        epoch_name = name + f"_ep{best_ep}"

                        trained_model = BertForSequenceClassification.from_pretrained(os.path.join(CHECKPOINT_DIR, epoch_name),
                                                                                        num_labels=NUM_LABELS,
                                                                                        output_hidden_states=True,
                                                                                        output_attentions=True)
                        trained_model.to(device)
                        dev_mets, dev_perf = inferencer.eval(trained_model, dev_batches, dev_labels,
                                                             set_type='dev', name=epoch_name)
                        logger.info(f'{epoch_name}: {dev_perf}')

                        if round(dev_mets['f1']*100,2) != best_ep_dev_f1:
                            logger.info(f"Performance not the same: {round(dev_mets['f1']*100,2)} not same as {best_ep_dev_f1} for {epoch_name}")

                        # check if best
                        if dev_mets['f1'] > best_val_res['f1']:
                            best_val_res.update(dev_mets)
                            best_val_res.update({'model_loc': os.path.join(CHECKPOINT_DIR, epoch_name)})

                    # load best model, save embeddings, print performance on test
                    if best_val_res['model_loc'] == '':
                        # none of the epochs performed above f1 = 0, so just use last epoch
                        best_val_res['model_loc'] = os.path.join(CHECKPOINT_DIR, epoch_name)

                    #best_model = BertForSequenceClassification.from_pretrained(best_val_res['model_loc'], num_labels=NUM_LABELS,
                    #                                                           output_hidden_states=True,
                    #                                                           output_attentions=True)
                    best_model = trained_model

                    logger.info(f"***** Embeds (and Test) - Fold {fold_name} *****")
                    logger.info(f"  Details: {best_val_res}")

                    for EMB_TYPE in ['poolbert']:
                        all_ids, all_batches, all_labels = load_features('data/features_for_bert/all_features.pkl', batch_size=1)
                        basil_w_BERT = pd.DataFrame(index=all_ids)
                        embs = inferencer.predict(best_model, all_batches, return_embeddings=True, emb_type=EMB_TYPE)
                        basil_w_BERT[EMB_TYPE] = embs
                        emb_name = f'{name}_basil_w_{EMB_TYPE}'
                        basil_w_BERT.to_csv(f'data/{emb_name}.csv')
                        logger.info(f'Got embs: \n{basil_w_BERT.head()}')
                        logger.info(f'Written embs ({len(embs)},{len(embs[0])}) to data/{emb_name}.csv')


                    test_mets, test_perf = inferencer.eval(best_model, test_batches, test_labels, set_type='test', name='best_model_loc')
                    logging.info(f"{test_perf}")
                    test_res.update(test_mets)

                    fold_results_table = fold_results_table.append(best_val_res, ignore_index=True)
                    fold_results_table = fold_results_table.append(test_res, ignore_index=True)
                    logging.info(f'Fold {fold_name} results: \n{fold_results_table[["model", "seed","bs", "lr", "fold", "set_type","f1"]]}')
                    setting_results_table = setting_results_table.append(fold_results_table)

                logging.info(f'Setting {setting_name} results: \n{setting_results_table[["model", "seed","bs","lr", "fold", "set_type","f1"]]}')
                setting_results_table.to_csv(f'reports/bert_baseline/tables/{setting_name}_get_embed_results_table.csv', index=False)
                main_results_table = main_results_table.append(setting_results_table, ignore_index=True)
            main_results_table.to_csv(f'reports/bert_baseline/tables/bert_get_embed_results_table.csv', index=False)

'''
n_train_batches = len(train_batches)
half_train_batches = int(n_train_batches / 2)
num_tr_opt_steps = n_train_batches * NUM_TRAIN_EPOCHS  # / GRADIENT_ACCUMULATION_STEPS
num_tr_warmup_steps = int(WARMUP_PROPORTION * num_tr_opt_steps)
#scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_tr_warmup_steps, num_training_steps=num_tr_opt_steps)
'''