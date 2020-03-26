from __future__ import absolute_import, division, print_function
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
import pickle
from lib.classifiers.BertForEmbed import BertForSequenceClassification, Inferencer, save_model
#from lib.classifiers.BertWrapper import BertForSequenceClassification, BertWrapper
from tqdm import trange
from datetime import datetime
from torch.nn import CrossEntropyLoss, MSELoss
import torch
from torch.utils.data import (DataLoader, SequentialSampler, RandomSampler, TensorDataset)
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


def to_tensor(features, OUTPUT_MODE):
    example_ids = [f.my_id for f in features]
    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

    if OUTPUT_MODE == "classification":
        label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif OUTPUT_MODE == "regression":
        label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    data = TensorDataset(input_ids, input_mask, segment_ids, label_ids)
    return example_ids, data, label_ids  # example_ids, input_ids, input_mask, segment_ids, label_ids

# split_input() # only needs to happen once, can be found in split_data

# find GPU if present
device, USE_CUDA = get_torch_device()

# Bert pre-trained model selected from: bert-base-uncased, bert-large-uncased, bert-base-cased, bert-large-cased,
# bert-base-multilingual-uncased, bert-base-multilingual-cased, bert-base-chinese.
BERT_MODEL = 'bert-base-cased'

# structure of project
TASK_NAME = 'bert_baseline'
DATA_DIR = 'data/features_for_bert/'
CHECKPOINT_DIR = f'models/checkpoints/{TASK_NAME}/'
REPORTS_DIR = f'reports/{TASK_NAME}'
CACHE_DIR = 'models/cache/' # This is where BERT will look for pre-trained models to load parameters from.

cache_dir = CACHE_DIR


################
# HYPERPARAMETERS
################

parser = argparse.ArgumentParser()
# TRAINING PARAMS
parser.add_argument('-ep', '--n_epochs', type=int, default=10)
parser.add_argument('-lr', '--learning_rate', type=float, default=2e-5)
parser.add_argument('-sv', '--sv', type=int, default=0)
parser.add_argument('-load', '--load_from_ep', type=int, default=0)
parser.add_argument('-f', '--fold', type=str, default='fan')
args = parser.parse_args()

NUM_TRAIN_EPOCHS = args.n_epochs
LEARNING_RATE = args.learning_rate
SEED = args.sv
LOAD_FROM_EP = args.load_from_ep

BATCH_SIZE = 24
GRADIENT_ACCUMULATION_STEPS = 1
WARMUP_PROPORTION = 0.1
NUM_LABELS = 2
PRINT_EVERY = 50

if SEED == 0:
    SEED_VAL = random.randint(0, 300)
else:
    SEED_VAL = SEED
random.seed(SEED_VAL)
np.random.seed(SEED_VAL)
#torch.random(SEED_VAL)
torch.manual_seed(SEED_VAL)
torch.cuda.manual_seed_all(SEED_VAL)

OUTPUT_MODE = 'classification'
output_mode = OUTPUT_MODE
inferencer = Inferencer(REPORTS_DIR, output_mode, logger, device, use_cuda=USE_CUDA)

# set logger
now = datetime.now()
now_string = now.strftime(format=f'%b-%d-%Hh-%-M_fold-{args.fold}_seed-{args.sv}')
LOG_NAME = f"{REPORTS_DIR}/{now_string}.log"

console_hdlr = logging.StreamHandler(sys.stdout)
file_hdlr = logging.FileHandler(filename=LOG_NAME)
logging.basicConfig(level=logging.INFO, handlers=[console_hdlr, file_hdlr])
logger = logging.getLogger()

logger.info(args)

fold_name = args.fold
if __name__ == '__main__':
    for SEED_VAL in [111]:
        for fold_name in ['6', '7', '8', '9', '10']: #'fan', '1', '2',
            if fold_name == 'orig':
                train_fp = os.path.join(DATA_DIR, "train_features.pkl")
                dev_fp = os.path.join(DATA_DIR, "dev_features.pkl")
                test_fp = os.path.join(DATA_DIR, "test_features.pkl")
                all_fp = os.path.join(DATA_DIR, "all_features.pkl")
            else:
                train_fp = os.path.join(DATA_DIR, f"folds/{fold_name}_train_features.pkl")
                dev_fp = os.path.join(DATA_DIR, f"folds/{fold_name}_dev_features.pkl")
                test_fp = os.path.join(DATA_DIR, f"folds/{fold_name}_test_features.pkl")

            with open(train_fp, "rb") as f:
                train_features = pickle.load(f)
                _, train_data, train_labels = to_tensor(train_features, OUTPUT_MODE)

            with open(dev_fp, "rb") as f:
                dev_features = pickle.load(f)
                _, dev_data, dev_labels = to_tensor(dev_features, OUTPUT_MODE)

            with open(test_fp, "rb") as f:
                test_features = pickle.load(f)
                _, test_data, test_labels = to_tensor(test_features, OUTPUT_MODE)

            logger.info(f"***** Training on Fold {fold_name} *****")
            logger.info(f"  Batch size = {BATCH_SIZE}")
            logger.info(f"  Learning rate = {LEARNING_RATE}")
            logger.info(f"  SEED = {SEED_VAL}")
            logger.info(f"  Logging to {LOG_NAME}")

            num_train_optimization_steps = int(
                len(train_features) / BATCH_SIZE) * NUM_TRAIN_EPOCHS  # / GRADIENT_ACCUMULATION_STEPS
            num_train_warmup_steps = int(WARMUP_PROPORTION * num_train_optimization_steps)

            if LOAD_FROM_EP:
                name = f'epoch{LOAD_FROM_EP}'
                load_dir = os.path.join(CHECKPOINT_DIR, name)
                logger.info(f'Loading model {load_dir}')
                model = BertForSequenceClassification.from_pretrained(load_dir, num_labels=NUM_LABELS,
                                                                      output_hidden_states=True, output_attentions=True)
                logger.info(f'Loaded model {load_dir}')
                inferencer.eval(model, dev_data, dev_labels, name=f'epoch{LOAD_FROM_EP}')
            else:
                load_dir = CACHE_DIR
                model = BertForSequenceClassification.from_pretrained(BERT_MODEL, cache_dir=load_dir, num_labels=NUM_LABELS,
                                                                      output_hidden_states=True, output_attentions=True)

            model.to(device)

            # optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, correct_bias=False) #, eps=1e-8)  # To reproduce BertAdam specific behavior set correct_bias=False
            optimizer = AdamW(model.parameters(), lr=LEARNING_RATE,
                              eps=1e-8)  # To reproduce BertAdam specific behavior set correct_bias=False
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_train_warmup_steps,
                                                        num_training_steps=num_train_optimization_steps)  # PyTorch scheduler

            global_step = 0
            nb_tr_steps = 0
            tr_loss = 0

            train_sampler = RandomSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

            model.train()

            t0 = time.time()
            best_val_mets = {'f1': 0}
            for ep in trange(1, int(NUM_TRAIN_EPOCHS+1), desc="Epoch"):
                if LOAD_FROM_EP: ep += LOAD_FROM_EP
                tr_loss = 0
                nb_tr_examples, nb_tr_steps = 0, 0
                for step, batch in enumerate(train_dataloader):
                    batch = tuple(t.to(device) for t in batch)
                    input_ids, input_mask, segment_ids, label_ids = batch
                    # print(label_ids)

                    model.zero_grad()
                    #if step % PRINT_EVERY == 0 and step != 0:
                    #    logger.info(input_ids) # [[  101,  4254,  1989,  ...,     0,     0,     0], [  101,   146,   787,  ...,     0,     0,     0],
                    #    logger.info(input_mask) #[[1, 1, 1,  ..., 0, 0, 0],
                    outputs = model(input_ids, input_mask, labels=label_ids)
                    (loss), logits, probs, sequence_output, pooled_output = outputs
                    loss = outputs[0]

                    if OUTPUT_MODE == "classification":
                        loss_fct = CrossEntropyLoss()
                        loss = loss_fct(logits.view(-1, NUM_LABELS), label_ids.view(-1))
                    elif OUTPUT_MODE == "regression":
                        loss_fct = MSELoss()
                        loss = loss_fct(logits.view(-1), label_ids.view(-1))

                    if GRADIENT_ACCUMULATION_STEPS > 1:
                        loss = loss / GRADIENT_ACCUMULATION_STEPS

                    loss.backward()

                    tr_loss += loss.item()
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1

                    # if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    optimizer.step()
                    scheduler.step()
                    global_step += 1

                    if step % PRINT_EVERY == 0 and step != 0:
                        # Calculate elapsed time in minutes.
                        elapsed = time.time() - t0
                        # Report progress.
                        # logging.info(f' Epoch {ep} / {NUM_TRAIN_EPOCHS} - {step} / {len(train_dataloader)} - Loss: {loss.item()}')

                # Save after Epoch
                epoch_name = f'bert_fold{fold_name}_ep{ep}'
                av_loss = tr_loss / len(train_dataloader)
                save_model(model, CHECKPOINT_DIR, epoch_name)
                dev_mets = inferencer.eval(model, dev_data, dev_labels, av_loss=av_loss, set_type='dev', name=epoch_name)
                if dev_mets['f1'] > best_val_mets['f1']:
                    best_val_mets = dev_mets

            # Save final model
            name = f'bert_fold{fold_name}_finepof{NUM_TRAIN_EPOCHS}'
            save_model(model, CHECKPOINT_DIR, name)
            test_mets = inferencer.eval(model, test_data, test_labels, set_type='test', name='test ' + name)
            logger.info(f"  Logged to {LOG_NAME}")

            results_df = pd.read_csv('reports/bert_baseline/results_table.csv', index_col=False)
            best_val_mets['seed'] = SEED_VAL
            best_val_mets['fold'] = fold_name
            best_val_mets['set_type'] = 'val'
            test_mets['seed'] = SEED_VAL
            test_mets['fold'] = fold_name
            test_mets['set_type'] = 'test'
            results_df = results_df.append(best_val_mets, ignore_index=True)
            results_df = results_df.append(test_mets, ignore_index=True)
            results_df.to_csv(f'reports/bert_baseline/results_table_{fold_name}_{SEED_VAL}.csv', index=False)
            results_df.to_csv('reports/bert_baseline/results_table.csv', index=False)
