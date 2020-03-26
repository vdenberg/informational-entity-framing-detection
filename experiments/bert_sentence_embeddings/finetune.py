from __future__ import absolute_import, division, print_function
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
import pickle
from lib.classifiers.BertForEmbed import BertForSequenceClassification, Inferencer, save_model
#from lib.classifiers.BertWrapper import BertForSequenceClassification, BertWrapper
from tqdm import trange
from torch.nn import CrossEntropyLoss, MSELoss
import torch
from torch.utils.data import (DataLoader, SequentialSampler, RandomSampler, TensorDataset)
import os, sys, random, argparse
import numpy as np
from lib.handle_data.PreprocessForBert import *
from lib.utils import get_torch_device
import time
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

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
TASK_NAME = 'bert_for_embed'
DATA_DIR = 'data/features_for_bert/'
CHECKPOINT_DIR = f'models/checkpoints/{TASK_NAME}/'
REPORTS_DIR = f'reports/{TASK_NAME}_evaluation_report/'
CACHE_DIR = 'models/cache/' # This is where BERT will look for pre-trained models to load parameters from.

cache_dir = CACHE_DIR
if os.path.exists(REPORTS_DIR) and os.listdir(REPORTS_DIR):
    REPORTS_DIR += f'/report_{len(os.listdir(REPORTS_DIR))}'
    os.makedirs(REPORTS_DIR)
if not os.path.exists(REPORTS_DIR):
    os.makedirs(REPORTS_DIR)
    REPORTS_DIR += f'/report_{len(os.listdir(REPORTS_DIR))}'
    os.makedirs(REPORTS_DIR)

################
# HYPERPARAMETERS
################

parser = argparse.ArgumentParser()
# TRAINING PARAMS
parser.add_argument('-ep', '--n_epochs', type=int, default=10)
parser.add_argument('-lr', '--learning_rate', type=float, default=2e-5)
parser.add_argument('-sv', '--sv', type=int, default=0)
parser.add_argument('-load', '--load_from_ep', type=int, default=0)
parser.add_argument('-f', '--fold_name', type=str, default='fan')
args = parser.parse_args()

NUM_TRAIN_EPOCHS = args.n_epochs
LEARNING_RATE = args.learning_rate
LOAD_FROM_EP = args.load_from_ep

BATCH_SIZE = 24
GRADIENT_ACCUMULATION_STEPS = 1
WARMUP_PROPORTION = 0.1
OUTPUT_MODE = 'classification'
NUM_LABELS = 2
PRINT_EVERY = 50

SEED_VAL = args.sv
FOLD_NAME = args.fold_name
fold_name = FOLD_NAME
random.seed(SEED_VAL)
np.random.seed(SEED_VAL)
torch.manual_seed(SEED_VAL)
torch.cuda.manual_seed_all(SEED_VAL)

output_mode = OUTPUT_MODE
inferencer = Inferencer(REPORTS_DIR, OUTPUT_MODE, logger, device, use_cuda=USE_CUDA)

FOLDS = False
if __name__ == '__main__':

    #for foldname in ['fan']: #, '0', '1', '2']:
    train_fp = os.path.join(DATA_DIR, f"folds/{fold_name}_train_features.pkl")
    dev_fp = os.path.join(DATA_DIR,  f"folds/{fold_name}_dev_features.pkl")
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
    logger.info(f"  Fold = {fold_name}")
    logger.info(f"  SEED = {SEED_VAL}")

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

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

    model.train()
    t0 = time.time()
    for ep in range(int(NUM_TRAIN_EPOCHS)):
        patience = 3
        prev_val_f1 = 0

        if LOAD_FROM_EP: ep += LOAD_FROM_EP
        tr_loss = 0
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            # print(label_ids)

            model.zero_grad()
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

            # if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            optimizer.step()
            scheduler.step()

        # Save after Epoch
        epoch_name = f'fold{fold_name}_sv{SEED_VAL}_ep{ep}'
        av_loss = tr_loss / len(train_dataloader)
        save_model(model, CHECKPOINT_DIR, epoch_name)
        val_f1 = inferencer.eval(model, dev_data, dev_labels, av_loss=av_loss, set_type='dev', name=epoch_name)

        if val_f1 > prev_val_f1:
            patience = 3
        else:
            patience -= 1
        if (not patience > 0) and (val_f1 > 0.20):
            logger.info("Stopping training")
            break
        prev_val_f1 = val_f1

    # Save final model
    name = f'fold{fold_name}_sv{SEED_VAL}_fin_ep_of_{NUM_TRAIN_EPOCHS}'
    save_model(model, 'models/', name)
    inferencer.eval(model, test_data, test_labels, set_type='test', name='val ' + name)

