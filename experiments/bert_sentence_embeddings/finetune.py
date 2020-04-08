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

'''
def to_tensor(features):
    example_ids = [f.my_id for f in features]
    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    data = TensorDataset(input_ids, input_mask, label_ids)
    return example_ids, data, label_ids  # example_ids, input_ids, input_mask, segment_ids, label_ids
'''

################
# HYPERPARAMETERS
################

parser = argparse.ArgumentParser()
# TRAINING PARAMS
parser.add_argument('-ep', '--n_epochs', type=int, default=4) #2,3,4
parser.add_argument('-lr', '--learning_rate', type=float, default=2e-5) #5e-5, 3e-5, 2e-5
parser.add_argument('-bs', '--batch_size', type=int, default=24) #16, 21
parser.add_argument('-load', '--load_from_ep', type=int, default=0)
args = parser.parse_args()

# find GPU if present
device, USE_CUDA = get_torch_device()
BERT_MODEL = 'bert-base-cased' #bert-large-cased
TASK_NAME = 'bert_baseline'
CHECKPOINT_DIR = f'models/checkpoints/{TASK_NAME}/'
REPORTS_DIR = f'reports/{TASK_NAME}'
CACHE_DIR = 'models/cache/' # This is where BERT will look for pre-trained models to load parameters from.

NUM_TRAIN_EPOCHS = args.n_epochs
LEARNING_RATE = args.learning_rate
LOAD_FROM_EP = args.load_from_ep
BATCH_SIZE = args.batch_size
GRADIENT_ACCUMULATION_STEPS = 1
WARMUP_PROPORTION = 0.1
NUM_LABELS = 2
PRINT_EVERY = 100

inferencer = Inferencer(REPORTS_DIR, logger, device, use_cuda=USE_CUDA)
results_table = pd.DataFrame(columns=['model,seed,fold,epoch,set_type,loss,acc,prec,rec,f1,fn,fp,tn,tp'.split(',')])

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

    model_locs = {'1': 'models/checkpoints/bert_baseline/bertforembed_263_f1_ep9',
                  '2': 'models/checkpoints/bert_baseline/bertforembed_263_f2_ep6',
                  '3': 'models/checkpoints/bert_baseline/bertforembed_263_f3_ep3',
                  '4': 'models/checkpoints/bert_baseline/bertforembed_263_f4_ep4',
                  '5': 'models/checkpoints/bert_baseline/bertforembed_263_f5_ep4',
                  '6': 'models/checkpoints/bert_baseline/bertforembed_263_f6_ep8',
                  '7': 'models/checkpoints/bert_baseline/bertforembed_263_f7_ep5',
                  '8': 'models/checkpoints/bert_baseline/bertforembed_263_f8_ep9',
                  '9': 'models/checkpoints/bert_baseline/bertforembed_263_f9_ep4',
                  '10': 'models/checkpoints/bert_baseline/bertforembed_263_f10_ep3'
                  }

    for SEED in [26354, 0, 0]:
        if SEED == 0:
            SEED_VAL = random.randint(0, 300)
        else:
            SEED_VAL = SEED
        random.seed(SEED_VAL)
        np.random.seed(SEED_VAL)
        torch.manual_seed(SEED_VAL)
        torch.cuda.manual_seed_all(SEED_VAL)

        for fold_name in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']:
            name_base = f"bert_{SEED_VAL}_f{fold_name}"

            best_val_mets = {'f1': 0}
            best_val_perf = ''
            best_model_loc = ''

            train_fp = f"data/features_for_bert/folds/{fold_name}_train_features.pkl")
            dev_fp = f"data/features_for_bert/folds/{fold_name}_dev_features.pkl")
            test_fp = f"data/features_for_bert/folds/{fold_name}_test_features.pkl")
            _, train_batches, train_labels = load_features(train_fp, BATCH_SIZE)
            _, dev_batches, dev_labels = load_features(dev_fp, BATCH_SIZE)
            _, test_batches, test_labels = load_features(test_fp, BATCH_SIZE)

            logger.info(f"***** Training on Fold {fold_name} *****")
            logger.info(f"  Batch size = {BATCH_SIZE}")
            logger.info(f"  Learning rate = {LEARNING_RATE}")
            logger.info(f"  SEED = {SEED_VAL}")
            logger.info(f"  Logging to {LOG_NAME}")

            model = BertForSequenceClassification.from_pretrained(BERT_MODEL, cache_dir=CACHE_DIR, num_labels=NUM_LABELS,
                                                                  output_hidden_states=True, output_attentions=True)
            model.to(device)
            optimizer = AdamW(model.parameters(), lr=LEARNING_RATE,  eps=1e-8)  # To reproduce BertAdam specific behavior set correct_bias=False
            model.train()

            for ep in range(1, NUM_TRAIN_EPOCHS+1):
                tr_loss = 0
                for step, batch in enumerate(train_batches):
                    batch = tuple(t.to(device) for t in batch)
                    input_ids, input_mask, labels = batch

                    model.zero_grad()
                    outputs = model(input_ids, input_mask, labels=labels)
                    (loss), logits, probs, sequence_output, pooled_output = outputs

                    #loss_fct = CrossEntropyLoss()
                    #loss = loss_fct(logits.view(-1, NUM_LABELS), label_ids.view(-1))

                    if GRADIENT_ACCUMULATION_STEPS > 1:
                        loss = loss / GRADIENT_ACCUMULATION_STEPS

                    loss.backward()
                    tr_loss += loss.item()
                    optimizer.step()
                    #scheduler.step()

                    if step % PRINT_EVERY == 0 and step != 0:
                        logging.info(f' Epoch {ep} / {NUM_TRAIN_EPOCHS} - {step} / {len(train_batches)}')

                # Save after Epoch
                epoch_name = name_base + f"_ep{ep}"
                av_loss = tr_loss / len(train_batches)
                save_model(model, CHECKPOINT_DIR, epoch_name)

                dev_mets, dev_perf = inferencer.eval(model, dev_batches, dev_labels, av_loss=av_loss, set_type='dev', name=epoch_name)

                # check if best
                high_score = ''
                if dev_mets['f1'] > best_val_mets['f1']:
                    best_ep = ep
                    best_val_mets = dev_mets
                    best_val_perf = dev_perf
                    best_model_loc = os.path.join(CHECKPOINT_DIR, epoch_name)
                    high_score = '(HIGH SCORE)'

                logger.info(f'ep {ep}: {dev_perf} {high_score}')

            best_model = BertForSequenceClassification.from_pretrained(best_model_loc, num_labels=NUM_LABELS,
                                                                       output_hidden_states=True,
                                                                       output_attentions=True)

            for EMB_TYPE in ['poolbert']:
                all_ids, all_batches, all_labels = load_features('data/features_for_bert/all_features.pkl', batch_size=1)
                embeddings = inferencer.predict(model, all_batches, return_embeddings=True, emb_type=EMB_TYPE)
                logger.info(f'Finished {len(embeddings)} embeddings with shape {embeddings.shape}')
                basil_w_BERT = pd.DataFrame(index=all_ids)
                basil_w_BERT[EMB_TYPE] = embeddings
                basil_w_BERT.to_csv(f'data/{SEED_VAL}_{fold_name}_basil_w_{EMB_TYPE}.csv')
                logger.info(f'Written to data/{SEED_VAL}_{fold_name}_basil_w_{EMB_TYPE}.csv')

            # Save final model
            logger.info(f"***** Testing on Fold {fold_name} *****")
            logger.info(f"  Model = {best_model_loc}")
            logger.info(f"  Batch size = {BATCH_SIZE}")
            logger.info(f"  Learning rate = {LEARNING_RATE}")
            logger.info(f"  SEED = {SEED_VAL}")
            logger.info(f"  Logging to {LOG_NAME}")

            name = name_base + f"_fin{NUM_TRAIN_EPOCHS}"
            save_model(model, CHECKPOINT_DIR, name)
            test_mets, test_perf = inferencer.eval(best_model, test_batches, test_labels, set_type='test', name='test ' + name)
            logging.info(f"{test_perf}")

            best_val_mets.update({'seed': SEED_VAL, 'fold': fold_name, 'set_type': 'val', 'epoch': best_model_loc[-1]})
            test_mets.update({'seed': SEED_VAL, 'fold': fold_name, 'set_type': 'test'})
            results_table = results_table.append(best_val_mets, ignore_index=True)
            results_table = results_table.append(test_mets, ignore_index=True)

    results_table.to_csv('reports/bert_baseline/results_table.csv', index=False)

'''
n_train_batches = len(train_batches)
half_train_batches = int(n_train_batches / 2)
num_tr_opt_steps = n_train_batches * NUM_TRAIN_EPOCHS  # / GRADIENT_ACCUMULATION_STEPS
num_tr_warmup_steps = int(WARMUP_PROPORTION * num_tr_opt_steps)
#scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_tr_warmup_steps, num_training_steps=num_tr_opt_steps)
'''