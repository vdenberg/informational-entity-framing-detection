# -*- coding: utf-8 -*-
"""
Based on: NLP From Scratch: Translation with a Sequence to Sequence Network and Attention
*******************************************************************************
**Author**: `Sean Robertson <https://github.com/spro/practical-pytorch>`_
"""

from __future__ import unicode_literals, print_function, division
from io import open
import random, os, sys
import torch
import numpy as np
import pandas as pd
from lib.classifiers.ContextAwareClassifier import ContextAwareClassifier, ContextAwareModel
from lib.handle_data.SplitData import Split
from lib.utils import get_torch_device
from torch.utils.data import (DataLoader, SequentialSampler, RandomSampler, TensorDataset)
import logging
from datetime import datetime
import argparse


######################################################################
# Loading data files
# ==================
#
# The data for this project is the BASIL dataset, stored with USE embeddings in the data directory of this project.
# We will be representing each article as a list of indexes to USE embeddings that we will story in a matrix.
#
# We'll need a unique index per word to use as the inputs and targets of
# the networks later. To keep track of all this we will use a helper class
# called ``Lang`` which has word → index (``word2index``) and index → word
# (``index2word``) dictionaries, as well as a count of each word
# ``word2count`` to use to later replace rare words.
#

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {'PAD': 0, "EOS": 1}
        self.word2count = {'PAD': 0, "EOS": 0}
        self.index2word = {0: 'PAD', 1: "EOS"}
        self.n_words = 2 # Count EOS
        self.max_len = 1  # remember EOS token

    def addSentence(self, sentence):
        sentence = sentence.split(' ')
        if len(sentence) + 1 > self.max_len:
            self.max_len = len(sentence) + 1  # remember EOS
        for word in sentence:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def read_documents(fp, max_seq_length):
    global NO_PADDING

    max_seq_length += 1 # for EOS token

    # Read the file and split into lines
    logger.info("Reading lines...")
    lines = open(fp, encoding='utf-8').read().strip().split('\n')

    # Split every line into triples and pad
    pairs = [[s.lower() for s in lin.split('\t')] for lin in lines]

    triples = []
    for p in pairs:
        id, article, label, index = p

        if not NO_PADDING:
            art_list = article.split(' ')
            padding = ['PAD'] * (max_seq_length + 1 - len(art_list))
            art_list += padding
            article = " ".join(art_list)
        label_i = int(label)
        idx = int(index)

        triples.append([article, label_i, idx])

    # make Lang instances
    input_lang = Lang('basil')

    # populate vocabulary
    for t in triples:
        input_lang.addSentence(t[0])

    return input_lang, triples


def make_weight_matrix(input_lang, embed_fp, EMB_DIM):
    basil = pd.read_csv(embed_fp, index_col=0).fillna('')
    basil = basil.rename(columns={'USE':'embeddings', 'sbert_pre':'embeddings', 'avbert':'embeddings'})
    sentence_embeddings = {i.lower(): np.array(u.strip('[]').split(', ')) for i, u in zip(basil.index, basil.embeddings)}

    matrix_len = input_lang.n_words
    weights_matrix = np.zeros((matrix_len, EMB_DIM))

    for word, index in input_lang.word2index.items(): # word here is a sentence id like 91fox27
        if (word == 'PAD') or (word == 'EOS') or (word == '11fox23'):
            pass
        else:
            embedding = sentence_embeddings[word]
            weights_matrix[index] = embedding

    #logging.info(f'Found embeddings for all but {words_not_found}')
    return weights_matrix

# =====================================================================================
#                    PARAMETERS
# =====================================================================================
# Read arguments from command line

parser = argparse.ArgumentParser()
# PRINT/SAVE PARAMS
parser.add_argument('-inf', '--step_info_every', type=int, default=1000)
parser.add_argument('-cp', '--save_epoch_cp_every', type=int, default=1)

# TRAINING PARAMS
parser.add_argument('-spl', '--split_type', type=str, default='fan')
parser.add_argument('-context', '--context_type', type=str, default='article')
parser.add_argument('-eval', '--eval', action='store_true', default=False)
parser.add_argument('-start', '--start_epoch', type=int, default=0)
parser.add_argument('-ep', '--epochs', type=int, default=500)

# OPTIMIZING PARAMS
parser.add_argument('-bs', '--batch_size', type=int, default=24)
parser.add_argument('-lr', '--learning_rate', type=float, default=2e-4)
parser.add_argument('-g', '--gamma', type=float, default=.95)

# NEURAL NETWORK DIMS
parser.add_argument('-maxlen', '--max_length', type=int, default=76)
parser.add_argument('-emb', '--embedding_type', type=str, default='avbert')
parser.add_argument('-hid', '--hidden_size', type=int, default=32)

# OTHER NN PARAMS
parser.add_argument('-sv', '--seed_val', type=int, default=124)
parser.add_argument('-nopad', '--no_padding', action='store_true', default=False)

args = parser.parse_args()

# set to variables for readability
PRINT_STEP_EVERY = args.step_info_every  # steps
SAVE_EPOCH_EVERY = args.save_epoch_cp_every  # epochs

SPLIT_TYPE = args.split_type
CONTEXT_TYPE = args.context_type
EVAL, TRAIN = args.eval, not args.eval
START_EPOCH = args.start_epoch
N_EPOCHS = args.epochs

BATCH_SIZE = args.batch_size
LR = args.learning_rate
GAMMA = args.gamma

MAX_LEN = args.max_length
EMB_TYPE = args.embedding_type
HIDDEN = args.hidden_size

SEED_VAL = args.seed_val
NO_PADDING = args.no_padding

# =====================================================================================
#                    SEED
# =====================================================================================

random.seed(SEED_VAL)
np.random.seed(SEED_VAL)
torch.manual_seed(SEED_VAL)
torch.cuda.manual_seed_all(SEED_VAL)

# =====================================================================================
#                    DIRECTORIES
# =====================================================================================

DATA_FP = f'data/cam_input/{CONTEXT_TYPE}/basil.tsv'
CHECKPOINT_DIR = f'models/checkpoints/cam/{EMB_TYPE}/{SPLIT_TYPE}'
BEST_CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, 'best')
REPORTS_DIR = f'reports/cam/{EMB_TYPE}'

if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)
if not os.path.exists(BEST_CHECKPOINT_DIR):
    os.makedirs(BEST_CHECKPOINT_DIR)
if not os.path.exists(REPORTS_DIR):
    os.makedirs(REPORTS_DIR)

# =====================================================================================
#                    LOGGER
# =====================================================================================

now = datetime.now()
now_string = now.strftime(format='%b-%d-%Hh-%-M')
LOG_NAME = f"{REPORTS_DIR}/{now_string}.log"

console_hdlr = logging.StreamHandler(sys.stdout)
file_hdlr = logging.FileHandler(filename=LOG_NAME)
logging.basicConfig(level=logging.INFO, handlers=[console_hdlr, file_hdlr])
logger = logging.getLogger()

logger.info(f"Start Logging to {LOG_NAME}")
logger.info(args)

# =====================================================================================
#                    DATA & EMBEDDINGS
# =====================================================================================

input_lang, triples = read_documents(DATA_FP, MAX_LEN)
logger.info("***** Loading data *****")
logger.info(f"Read {len(triples)} sentence triples")
logger.info(f"Embedding type: {EMB_TYPE}")
logger.info(f"Max len: {input_lang.max_len}")
logger.info(f"Example: {random.choice(triples)}")

# split data
triple_dict = {trip[0].split(' ')[trip[-1]]: trip for trip in triples}
spl = Split(triple_dict, which=SPLIT_TYPE, tst=False)
folds = spl.apply_split(features=['sentence'], input_as='pytorch', output_as='pytorch')
NR_FOLDS = len(folds)

# get embeddings
if EMB_TYPE == 'USE':
    EMB_DIM = 512
    embed_fp = 'data/basil_w_USE.csv'
elif EMB_TYPE == 'sbert':
    EMB_DIM = 768
    embed_fp = 'data/basil_w_sbert_pre.csv'
elif EMB_TYPE == 'avbert':
    EMB_DIM = 768
    embed_fp = 'data/basil_w_avBERT.csv'
WEIGHTS_MATRIX = make_weight_matrix(input_lang, embed_fp, EMB_DIM)

#######################################################################################
# =====================================================================================
#                    TRAINING
# =====================================================================================

logger.info("***** Starting training *****")
logger.info(f"Mode: {'train' if not EVAL else 'eval'}")
logger.info(f"Context: {CONTEXT_TYPE}")
logger.info(f"Num epochs: {N_EPOCHS}")
logger.info(f"Starting from: {START_EPOCH}")
logger.info(f"Batch size: {BATCH_SIZE}")
logger.info(f"Starting LR: {LR}")

# loop through folds for cross validation, note that if split_type is 'fan' there is only 1 fold
cross_val_df = pd.DataFrame(index=list(range(NR_FOLDS)) + ['mean'], columns=['Acc', 'Prec', 'Rec', 'F1'])
for fold_i, fold in enumerate(folds):

    # initialise classifier with development data of this fold and all parameters
    # loading from a checkpoint will happen automatically if you passed a starting checkpoint/epoch
    cl = ContextAwareClassifier(input_lang, fold['dev'], fold['test'], start_epoch=START_EPOCH,
                                logger=logger, cp_dir=CHECKPOINT_DIR,
                                weights_matrix=WEIGHTS_MATRIX, emb_dim=EMB_DIM, hidden_size=HIDDEN,
                                batch_size=BATCH_SIZE, learning_rate=LR,
                                step_size=1, gamma=GAMMA)

    # set model into eval mode, or train model
    if EVAL:
        cl.model.eval()
        for parameter in cl.model.parameters():
            parameter.requires_grad = False
    elif TRAIN:
        cl.model.train()
        cl.train_epochs(fold, num_epochs=N_EPOCHS, print_step_every=PRINT_STEP_EVERY, save_epoch_every=SAVE_EPOCH_EVERY)

    # evaluate on test data of this fold
    metrics, metrics_df, metrics_string = cl.evaluate(fold['test'], which='all')
    logger.info(metrics_string)
    cross_val_df.loc[fold_i] = metrics_df[['acc', 'pred', 'rec', 'f1']]

# average across folds
cross_val_df.loc['mean'] = cross_val_df.mean()
logger.info(cross_val_df)
