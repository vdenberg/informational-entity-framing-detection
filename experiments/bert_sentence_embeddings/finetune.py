from __future__ import absolute_import, division, print_function
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
import pickle
from lib.classifiers.BertForEmbed import Inferencer, save_model, BertForSequenceClassification
#from lib.classifiers.BertWrapper import BertWrapper, BertForSequenceClassification
from lib.evaluate.StandardEval import my_eval
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


# =====================================================================================
#                    PARAMETERS
# =====================================================================================

# Read arguments from command line

class OldFinetuner:
    def __init__(self, logger, n_epochs=5, lr=2e-5, seed=6, load_from_ep=0):
        self.device, self.USE_CUDA = get_torch_device()
        self.BERT_MODEL = 'bert-base-cased'
        self.TASK_NAME = 'bert_for_embed'
        self.DATA_DIR = 'data/features_for_bert/'
        self.CHECKPOINT_DIR = f'models/checkpoints/bert_for_embed/'
        self.REPORTS_DIR = f'reports/bert_for_embed_evaluation_report/'
        self.CACHE_DIR = 'models/cache/' # This is where BERT will look for pre-trained models to load parameters from.

        cache_dir = self.CACHE_DIR
        if os.path.exists(self.REPORTS_DIR) and os.listdir(self.REPORTS_DIR):
            self.REPORTS_DIR += f'/report_{len(os.listdir(self.REPORTS_DIR))}'
            os.makedirs(self.REPORTS_DIR)
        if not os.path.exists(self.REPORTS_DIR):
            os.makedirs(self.REPORTS_DIR)
            self.REPORTS_DIR += f'/report_{len(os.listdir(self.REPORTS_DIR))}'
            os.makedirs(self.REPORTS_DIR)

        self.NUM_TRAIN_EPOCHS = n_epochs
        self.LEARNING_RATE = lr
        self.SEED = seed
        self.LOAD_FROM_EP = load_from_ep

        self.BATCH_SIZE = 24
        self.GRADIENT_ACCUMULATION_STEPS = 1
        self.WARMUP_PROPORTION = 0.1
        self.NUM_LABELS = 2
        self.PRINT_EVERY = 50
        self.trained_model = None

        if self.SEED == 0:
            SEED_VAL = random.randint(0, 300)
        else:
            SEED_VAL = self.SEED
        random.seed(SEED_VAL)
        np.random.seed(SEED_VAL)
        torch.manual_seed(SEED_VAL)
        torch.cuda.manual_seed_all(SEED_VAL)
        self.SEED_VAL = SEED_VAL

        self.OUTPUT_MODE = 'classification'

        self.logger = logger
        self.inferencer = Inferencer(self.REPORTS_DIR, self.OUTPUT_MODE, logger, self.device, use_cuda=self.USE_CUDA)

    def train(self, train_data, train_labels, dev_data, dev_labels, n_train_batches, name):

        logger.info(f"***** Training on Fold {'fan'} *****")
        logger.info(f"  Batch size = {self.BATCH_SIZE}")
        logger.info(f"  Learning rate = {self.LEARNING_RATE}")
        logger.info(f"  SEED = {self.SEED_VAL}")

        #bertwrapper = BertWrapper(self.CHECKPOINT_DIR, self.NUM_TRAIN_EPOCHS, len(train_features) / self.BATCH_SIZE, self.LOAD_FROM_EP)

        if self.LOAD_FROM_EP:
            name = f'epoch{self.LOAD_FROM_EP}'
            load_dir = os.path.join(self.CHECKPOINT_DIR, name)
            logger.info(f'Loading model {load_dir}')
            model = BertForSequenceClassification.from_pretrained(load_dir, num_labels=self.NUM_LABELS,
                                                                  output_hidden_states=True, output_attentions=True)
            logger.info(f'Loaded model {load_dir}')
            self.inferencer.eval(model, dev_data, dev_labels, name=f'epoch{self.LOAD_FROM_EP}')
        else:
            load_dir = self.CACHE_DIR
            model = BertForSequenceClassification.from_pretrained(self.BERT_MODEL, cache_dir=load_dir, num_labels=self.NUM_LABELS,
                                                                  output_hidden_states=True, output_attentions=True)

        model.to(self.device)

        # optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, correct_bias=False) #, eps=1e-8)  # To reproduce BertAdam specific behavior set correct_bias=False
        num_train_optimization_steps = n_train_batches * self.NUM_TRAIN_EPOCHS  # / GRADIENT_ACCUMULATION_STEPS
        num_train_warmup_steps = int(self.WARMUP_PROPORTION * num_train_optimization_steps)

        optimizer = AdamW(model.parameters(), lr=self.LEARNING_RATE,
                          eps=1e-8)  # To reproduce BertAdam specific behavior set correct_bias=False
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_train_warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)  # PyTorch scheduler

        global_step = 0
        nb_tr_steps = 0
        tr_loss = 0

        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.BATCH_SIZE)

        model.train()

        t0 = time.time()
        for ep in trange(int(self.NUM_TRAIN_EPOCHS), desc="Epoch"):
            if self.LOAD_FROM_EP: ep += self.LOAD_FROM_EP
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                # print(label_ids)

                model.zero_grad()
                outputs = model(input_ids, input_mask, labels=label_ids)
                (loss), logits, probs, sequence_output, pooled_output = outputs
                loss = outputs[0]

                if self.OUTPUT_MODE == "classification":
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.NUM_LABELS), label_ids.view(-1))
                elif self.OUTPUT_MODE == "regression":
                    loss_fct = MSELoss()
                    loss = loss_fct(logits.view(-1), label_ids.view(-1))

                if self.GRADIENT_ACCUMULATION_STEPS > 1:
                    loss = loss / self.GRADIENT_ACCUMULATION_STEPS

                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                # if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                scheduler.step()
                global_step += 1

                if step % self.PRINT_EVERY == 0 and step != 0:
                    # Calculate elapsed time in minutes.
                    elapsed = time.time() - t0
                    # Report progress.
                    # logging.info(f' Epoch {ep} / {NUM_TRAIN_EPOCHS} - {step} / {len(train_dataloader)} - Loss: {loss.item()}')

            # Save after Epoch
            ep_name = f'epoch{ep}'
            av_loss = tr_loss / len(train_dataloader)

            dev_preds = self.inferencer.predict(model, dev_data)
            dev_mets, dev_perf = my_eval(dev_labels.numpy(), dev_preds, av_loss=av_loss, set_type='dev', name=ep_name)
            self.logger.info(f" > BERT on {name} Epoch {ep} (took {elapsed}): {dev_perf}")
            #bertwrapper.model = model
            #bertwrapper.save_model('models/', final_name)

        self.trained_model = model
        return model

    def test(self, test_data, test_labels, name):
        save_model(self.trained_model, self.CHECKPOINT_DIR, name)
        preds = self.inferencer.predict(self.trained_model, test_data)
        test_mets, test_perf = my_eval(test_labels.numpy(), preds, set_type='test', name=name)
        self.logger.info(f' FINISHED training {name} (took {self.train_time})')
        self.logger.info(f' {test_perf}')

    def fan(self):
        with open(self.DATA_DIR + "folds/fan_test_features.pkl", "rb") as f:
        #with open(self.DATA_DIR + "test_features.pkl", "rb") as f:
            test_features = pickle.load(f)
            test_ids, test_data, test_labels = to_tensor(test_features, self.OUTPUT_MODE)

        #with open(self.DATA_DIR + "train_features.pkl", "rb") as f:
        with open(self.DATA_DIR + "folds/fan_train_features.pkl", "rb") as f:
            train_features = pickle.load(f)
            train_ids, train_data, train_labels = to_tensor(train_features, self.OUTPUT_MODE)

        #with open(self.DATA_DIR + "dev_features.pkl", "rb") as f:
        with open(self.DATA_DIR + "folds/fan_dev_features.pkl", "rb") as f:
            dev_features = pickle.load(f)
            dev_ids, dev_data, dev_labels = to_tensor(dev_features, self.OUTPUT_MODE)

        n_train_batches = int(len(train_features) / self.BATCH_SIZE)
        self.train(train_data, train_labels, dev_data, dev_labels, n_train_batches=n_train_batches, name='fan')
        self.test(test_data, test_labels, name='fan')

    def berg(self):
        for fold_name in range(0,10):
            train_fp = os.path.join(self.DATA_DIR, 'folds', f"{fold_name}_train_features.pkl")
            dev_fp = os.path.join(self.DATA_DIR, 'folds', f"{fold_name}_dev_features.pkl")
            test_fp = os.path.join(self.DATA_DIR, 'folds', f"{fold_name}_test_features.pkl")

            with open(test_fp, "rb") as f:
                # with open(self.DATA_DIR + "test_features.pkl", "rb") as f:
                test_features = pickle.load(f)
                test_ids, test_data, test_labels = to_tensor(test_features, self.OUTPUT_MODE)

            # with open(self.DATA_DIR + "train_features.pkl", "rb") as f:
            with open(train_fp, "rb") as f:
                train_features = pickle.load(f)
                train_ids, train_data, train_labels = to_tensor(train_features, self.OUTPUT_MODE)

            # with open(self.DATA_DIR + "dev_features.pkl", "rb") as f:
            with open(dev_fp, "rb") as f:
                dev_features = pickle.load(f)
                dev_ids, dev_data, dev_labels = to_tensor(dev_features, self.OUTPUT_MODE)

            n_train_batches = int(len(train_features) / self.BATCH_SIZE)
            self.train(train_data, train_labels, dev_data, dev_labels, n_train_batches=n_train_batches, name=f'fold{fold_name}')
            self.test(test_data, test_labels, name=f'fold{fold_name}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-sv', '--sv', type=int, default=111)
    parser.add_argument('-lr', '--lr', type=float, default=2e-5)
    parser.add_argument('-ep', '--ep', type=int, default=5)
    args = parser.parse_args()

    # set logger
    now = datetime.now()
    now_string = now.strftime(format='%b-%d-%Hh-%-M')
    REPORTS_DIR = 'reports/bertbaseline/'
    LOG_NAME = f"{REPORTS_DIR}/{now_string}.log"

    console_hdlr = logging.StreamHandler(sys.stdout)
    file_hdlr = logging.FileHandler(filename=LOG_NAME)
    logging.basicConfig(level=logging.INFO, handlers=[console_hdlr, file_hdlr])
    logger = logging.getLogger()

    logger.info(f"Start Logging to {LOG_NAME}")
    logger.info(args)
    for seed in [111, 6, 263, 124, 1123]:
        ft = OldFinetuner(logger=logger, n_epochs=args.ep, lr=args.lr, seed=seed, load_from_ep=0)
        ft.fan()
        ft.berg()
