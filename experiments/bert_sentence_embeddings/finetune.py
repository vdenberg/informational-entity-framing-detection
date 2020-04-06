from __future__ import absolute_import, division, print_function
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
import pickle
from lib.classifiers.BertForEmbed import BertForSequenceClassification, Inferencer
from lib.classifiers.BertWrapper import BertForSequenceClassification, BertWrapper, InputFeatures
from datetime import datetime
from torch.nn import CrossEntropyLoss
import torch
import random, argparse
import numpy as np
from lib.handle_data.PreprocessForBert import *
from lib.utils import get_torch_device
import logging
from lib.utils import to_batches, to_tensors
from lib.evaluate.Eval import eval
import time

#######
# FROM:
# https://medium.com/swlh/how-twitter-users-turned-bullied-quaden-bayles-into-a-scammer-b14cb10e998a?source=post_recirc---------1------------------
#####

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
parser.add_argument('-ep', '--n_epochs', type=int, default=10)
parser.add_argument('-inf', '--print_every', type=int, default=100)
parser.add_argument('-lr', '--learning_rate', type=float, default=2e-5)
parser.add_argument('-sv', '--sv', type=int, default=263)
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
PRINT_EVERY = args.print_every

if SEED == 0:
    SEED_VAL = random.randint(0, 300)
else:
    SEED_VAL = SEED
random.seed(SEED_VAL)
np.random.seed(SEED_VAL)
torch.manual_seed(SEED_VAL)
torch.cuda.manual_seed_all(SEED_VAL)

OUTPUT_MODE = 'classification'
output_mode = OUTPUT_MODE
inferencer = Inferencer(REPORTS_DIR, output_mode, logger, device, use_cuda=USE_CUDA)

'''
Set up for BERT baseline:
5 seeds
10 folds

for SEED_VAL in [111, 263, 6, 124, 1001]:
    for fold_name in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']: #'fan', 

'''

'''
Set up for embeddings production:
-- experiment with Cyclic LR -> not clearly better
-- experiment with batchsize -> looks good
seeds: 111 and 263
folds: 1, 2, 6

for SEED_VAL in [111, 263]:
    for fold_name in ['1', '2', '6']: 
        for schedule in ['cyclic', 'warmup']:
            for BATCH_SIZE in [8, 16, 24]:
                name_base = f"bert{SEED_VAL}_fold{fold_name}_sch{schedule}_bs{batchsize}":
                

'''

best_val_mets = {'f1': 0}
best_val_perf = ''
best_model_loc = ''

# set logger
now = datetime.now()
now_string = now.strftime(format=f'%b-%d-%Hh-%-M_{"finding good dev model"}')
LOG_NAME = f"{REPORTS_DIR}/{now_string}.log"

console_hdlr = logging.StreamHandler(sys.stdout)
file_hdlr = logging.FileHandler(filename=LOG_NAME)
logging.basicConfig(level=logging.INFO, handlers=[console_hdlr, file_hdlr])
logger = logging.getLogger()

logger.info(args)

with open("data/features_for_bert/folds/all_features.pkl", "rb") as f:
    all_ids, all_data, all_labels = to_tensors(features=pickle.load(f))
all_batches = to_batches(all_data, batch_size=1)

best_model_loc = {'1': 'models/checkpoints/bert_baseline/bertforembed_263_f1_ep9',
                  '2': 'models/checkpoints/bert_baseline/bertforembed_263_f2_ep6',
                  '3': 'models/checkpoints/bert_baseline/bertforembed_263_f3_ep3',
                  '4': 'models/checkpoints/bert_baseline/bertforembed_263_f4_ep4',
                  '5': 'models/checkpoints/bert_baseline/bertforembed_263_f5_ep4',
                  '6': 'models/checkpoints/bert_baseline/bertforembed_263_f6_ep8',
                  '7': 'models/checkpoints/bert_baseline/bertforembed_263_f7_ep5',
                  '8': 'models/checkpoints/bert_baseline/bertforembed_263_f8_ep9',
                  '9': 'models/checkpoints/bert_baseline/bertforembed_263_f9_ep4',
                  '10': 'models/checkpoints/bert_baseline/bertforembed_263_f10_ep3'}

for fold_name in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']:
    name_base = f"bert_{SEED_VAL}_f{fold_name}"

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
        _, train_data, train_labels = to_tensors(features=train_features)

    with open(dev_fp, "rb") as f:
        dev_features = pickle.load(f)
        _, dev_data, dev_labels = to_tensors(features=dev_features)

    with open(test_fp, "rb") as f:
        test_features = pickle.load(f)
        _, test_data, test_labels = to_tensors(features=test_features)

    logger.info(f"***** Training on Fold {fold_name} *****")
    logger.info(f"  Batch size = {BATCH_SIZE}")
    logger.info(f"  Learning rate = {LEARNING_RATE}")
    logger.info(f"  SEED = {SEED_VAL}")
    logger.info(f"  Logging to {LOG_NAME}")

    n_train_batches = int(len(train_features) / BATCH_SIZE)
    num_train_optimization_steps = n_train_batches * NUM_TRAIN_EPOCHS  # / GRADIENT_ACCUMULATION_STEPS #half_train_batches = int(n_train_batches / 2)
    num_train_warmup_steps = int(WARMUP_PROPORTION * num_train_optimization_steps)
    wrapper = BertWrapper(cp_dir=CHECKPOINT_DIR, n_eps=NUM_TRAIN_EPOCHS, n_train_batches=n_train_batches,
                              bert_lr=LEARNING_RATE, seed_val=SEED_VAL)

    load_dir = CACHE_DIR
    model = BertForSequenceClassification.from_pretrained(BERT_MODEL, cache_dir=load_dir, num_labels=NUM_LABELS,
                                                          output_hidden_states=True, output_attentions=True)
    model2 = BertForSequenceClassification.from_pretrained(BERT_MODEL, cache_dir=load_dir, num_labels=NUM_LABELS,
                                                          output_hidden_states=True, output_attentions=True)

    model.to(device)
    model2.to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE,
                      eps=1e-8)  # To reproduce BertAdam specific behavior set correct_bias=False
    optimizer2 = AdamW(model.parameters(), lr=LEARNING_RATE,
                      eps=1e-8)  # To reproduce BertAdam specific behavior set correct_bias=False
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_train_warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)  # PyTorch scheduler #CyclicLR(optimizer, base_lr=LEARNING_RATE, step_size_up=half_train_batches,
                               # cycle_momentum=False, max_lr=LEARNING_RATE*2)
    scheduler2 = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_train_warmup_steps,
                                                num_training_steps=num_train_optimization_steps)  # PyTorch scheduler #CyclicLR(optimizer, base_lr=LEARNING_RATE, step_size_up=half_train_batches,
    # cycle_momentum=False, max_lr=LEARNING_RATE*2)

    train_batches = to_batches(train_data, BATCH_SIZE)
    dev_batches = to_batches(dev_data, BATCH_SIZE)
    test_batches = to_batches(test_data, BATCH_SIZE)

    model.train()
    model2.train()

    t0 = time.time()
    for ep in range(1, NUM_TRAIN_EPOCHS+1):
        tr_loss = 0
        tr_loss2 = 0
        for step, batch in enumerate(train_batches):
            # style 1
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, label_ids = batch
            model.zero_grad()
            outputs = model(input_ids, input_mask, labels=label_ids)
            (loss), logits, probs, sequence_output, pooled_output = outputs
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, NUM_LABELS), label_ids.view(-1))
            tr_loss += loss.item
            optimizer.step()
            loss.backward()
            scheduler.step()

            # copy of style 1
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, label_ids = batch
            model2.zero_grad()
            outputs2 = model(input_ids, input_mask, labels=label_ids)
            (loss), logits2, probs, sequence_output, pooled_output = outputs
            loss_fct = CrossEntropyLoss()
            loss2 = loss_fct(logits2.view(-1, NUM_LABELS), label_ids.view(-1))
            loss2.backward()
            tr_loss2 += loss2.item
            optimizer2.step()
            scheduler2.step()

            #style 2
            tr_loss3 = wrapper.train_on_batch(batch)

            if step % PRINT_EVERY == 0 and step != 0:
                # Calculate elapsed time in minutes.
                elapsed = time.time() - t0
                logging.info(f' Epoch {ep}/{NUM_TRAIN_EPOCHS} - {step}/{len(train_batches)} - Tr Loss: {loss.item()} & {loss2.item()}')

        # Save after Epoch
        epoch_name = name_base + f"_ep{ep}"
        av_tr_loss = tr_loss / len(train_batches)
        av_tr_loss2 = tr_loss2 / len(train_batches)

        dev_mets1, dev_perf1 = inferencer.eval(model, dev_batches, dev_labels, av_loss=av_tr_loss, set_type='dev', name=epoch_name)
        dev_mets2, dev_perf2 = inferencer.eval(model2, dev_batches, dev_labels, av_loss=av_tr_loss2, set_type='dev', name=epoch_name)
        dev_preds, dev_loss = wrapper.predict(dev_batches)
        dev_mets, dev_perf = eval(dev_labels, dev_preds, set_type='dev', av_loss=tr_loss3, name=epoch_name)
        wrapper.save_model(epoch_name)

        # check if best
        high_score = ''
        if dev_mets['f1'] > best_val_mets['f1']:
            best_ep = ep
            best_val_mets = dev_mets
            best_val_perf = dev_perf
            best_model_loc = os.path.join(CHECKPOINT_DIR, epoch_name)
            high_score = '(HIGH SCORE)'

        logger.info(f'outside of function: ep {ep}: {dev_perf1} ')
        logger.info(f'outside of function 2: ep {ep}: {dev_perf2}')
        logger.info(f'inside a function: ep {ep}: {dev_perf} ') #{high_score}

    for EMB_TYPE in ['poolbert']:
        best_model_loc = best_model_loc[fold_name]
        embeddings = wrapper.get_embeddings(all_batches, model_path=best_model_loc, emb_type=EMB_TYPE)
        logger.info(f'Finished {len(embeddings)} embeddings with shape ({len(embeddings)}, {len(embeddings[0])})')
        #embeddings = inferencer.predict(model, all_batches, return_embeddings=True, emb_type=EMB_TYPE)
        basil_w_BERT = pd.DataFrame(index=all_ids)
        basil_w_BERT[EMB_TYPE] = embeddings
        basil_w_BERT.to_csv(f'data/{fold_name}_basil_w_{EMB_TYPE}.csv')
        logger.info(f'Written to data/{fold_name}_basil_w_{EMB_TYPE}.csv')

    BASELINE = False
    if BASELINE:
        # Save final model
        logger.info(f"***** Testing on Fold {fold_name} *****")
        logger.info(f"  Model = {best_model_loc}")
        logger.info(f"  Batch size = {BATCH_SIZE}")
        logger.info(f"  Learning rate = {LEARNING_RATE}")
        logger.info(f"  SEED = {SEED_VAL}")
        logger.info(f"  Logging to {LOG_NAME}")

        name =  name_base + f"_fin{NUM_TRAIN_EPOCHS}"
        #save_model(model, CHECKPOINT_DIR, name)
        test_mets, test_perf = inferencer.eval(best_model, test_batches, test_labels, set_type='test', name='test ' + name)
        logging.info(f"{test_perf}")

        results_df = pd.read_csv('reports/bert_baseline/new_results_table.csv', index_col=False)
        best_val_mets['seed'] = SEED_VAL
        best_val_mets['fold'] = fold_name
        best_val_mets['epoch'] = best_model_loc[fold_name][-1] #
        best_val_mets['set_type'] = 'val'
        test_mets['seed'] = SEED_VAL
        test_mets['fold'] = fold_name
        test_mets['set_type'] = 'test'
        results_df = results_df.append(best_val_mets, ignore_index=True)
        results_df = results_df.append(test_mets, ignore_index=True)
        #results_df.to_csv(f'reports/bert_baseline/results_table_{fold_name}_{SEED_VAL}.csv', index=False)
        results_df.to_csv('reports/bert_baseline/new_results_table.csv', index=False)

logger.info(f"Best model overall: {best_model_loc}: {best_val_perf}")
