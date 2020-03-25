from __future__ import absolute_import, division, print_function
import os
import torch
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from torch.nn import CrossEntropyLoss, MSELoss
import pickle
from lib.classifiers.BertForEmbed import BertForSequenceClassification, Inferencer
from lib.handle_data.PreprocessForBert import InputFeatures
from lib.utils import get_torch_device
from tqdm import trange
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def to_tensor(features, OUTPUT_MODE='classification'):
    example_ids = [f.my_id for f in features]
    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

    label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

    data = TensorDataset(input_ids, input_mask, segment_ids, label_ids)
    return data, label_ids # example_ids, input_ids, input_mask, segment_ids, label_ids


def save_model(model_to_save, model_dir):
    model_to_save = model_to_save.module if hasattr(model_to_save, 'module') else model_to_save  # Only save the model it-self

    config_name = f"config.json"
    weights_name = f"pytorch_model.bin"
    output_model_file = os.path.join(model_dir, weights_name)
    output_config_file = os.path.join(model_dir, config_name)

    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)


#######
# FROM:
# https://medium.com/swlh/how-twitter-users-turned-bullied-quaden-bayles-into-a-scammer-b14cb10e998a?source=post_recirc---------1------------------
#####

# split_input() # only needs to happen once, can be found in split_data

# find GPU if present
device, USE_CUDA =  get_torch_device() #torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Bert pre-trained model selected from: bert-base-uncased, bert-large-uncased, bert-base-cased, bert-large-cased,
# bert-base-multilingual-uncased, bert-base-multilingual-cased, bert-base-chinese.
BERT_MODEL = 'bert-base-cased'

# structure of project
DATA_DIR = 'data/features_for_bert/'
CHECKPOINT_DIR = f'checkpoints/bert_for_embed/'
REPORTS_DIR = f'reports/bert_for_embed_evaluation_report/'
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

BATCH_SIZE = 24 #24
LEARNING_RATE = 2e-3 #2e-5
NUM_TRAIN_EPOCHS = 3
RANDOM_SEED = 124
GRADIENT_ACCUMULATION_STEPS = 1
WARMUP_PROPORTION = 0.1
OUTPUT_MODE = 'classification'
NUM_LABELS = 2

output_mode = OUTPUT_MODE

with open(DATA_DIR + "train_features.pkl", "rb") as f:
    train_features = pickle.load(f)
    train_data, train_labels = to_tensor(train_features)

with open(DATA_DIR + "dev_features.pkl", "rb") as f:
    dev_features = pickle.load(f)
    dev_data, dev_labels = to_tensor(dev_features)

num_train_optimization_steps = int(len(train_features) / BATCH_SIZE / GRADIENT_ACCUMULATION_STEPS) * NUM_TRAIN_EPOCHS
num_train_warmup_steps = int(WARMUP_PROPORTION * num_train_optimization_steps)

inferencer = Inferencer(REPORTS_DIR, OUTPUT_MODE, logger, device, USE_CUDA)

if __name__ == '__main__':
    model = BertForSequenceClassification.from_pretrained(BERT_MODEL, cache_dir=CACHE_DIR, num_labels=NUM_LABELS)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE,
                      correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_train_warmup_steps,
                                                num_training_steps=num_train_optimization_steps)  # PyTorch scheduler

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_features))
    logger.info("  Batch size = %d", BATCH_SIZE)
    logger.info("  Num steps = %d", num_train_optimization_steps)

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

    model.train()
    for ep in trange(int(NUM_TRAIN_EPOCHS), desc="Epoch"):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            (loss), logits, pooled_output = model(input_ids, segment_ids, input_mask, labels=label_ids)
            #print(outputs)
            #hidden_states = outputs[1]
            #av_hidden_states = hidden_states.mean()

            if OUTPUT_MODE == "classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, NUM_LABELS), label_ids.view(-1))
            elif OUTPUT_MODE == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), label_ids.view(-1))

            if GRADIENT_ACCUMULATION_STEPS > 1:
                loss = loss / GRADIENT_ACCUMULATION_STEPS

            loss.backward()
            print('\rEpoch {} / {} - Step {} / {} - Loss: {}'.format(ep, NUM_TRAIN_EPOCHS, step, len(train_dataloader), loss), end='')

            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

        # Save after Epoch
        intermediate_name = f'epoch{ep}'
        inferencer.eval(model, dev_data, dev_labels, name=intermediate_name)
        save_model(model, CHECKPOINT_DIR, intermediate_name)

    # Save final model
    final_name = f'finetuned'
    inferencer.eval(model, dev_data, dev_labels, name=final_name)
    save_model(model, CHECKPOINT_DIR, final_name)