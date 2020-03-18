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


def readArticles(fp, max_seq_length=95):
    global NO_PADDING
    logger.info("Reading lines...")
    # Read the file and split into lines
    lines = open(fp, encoding='utf-8').read().strip().split('\n')

    # Split every line into triples and normalize
    pairs = [[s.lower() for s in lin.split('\t')] for lin in lines]

    # EB reshape to fit my task
    triples = []
    for p in pairs:
        id, article, label, index = p

        if not NO_PADDING:
            art_list = article.split(' ')
            padding = ['PAD'] * (77 - len(art_list))
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
        if (word == 'PAD') or (word == 'EOS'):
            pass
        else:
            embedding = sentence_embeddings[word]
            weights_matrix[index] = embedding

    #logging.info(f'Found embeddings for all but {words_not_found}')
    return weights_matrix


def load_model_from_checkpoint(cp_dir, spl, fold_i, cp):
    if spl == 'fan':
        cpfn = 'cp_fan_{}.pth'.format(cp)
    elif spl == 'berg':
        cpfn = 'cp_{}_{}.pth'.format(fold_i, cp)
    cpfp = os.path.join(cp_dir, cpfn)
    logger.info('Loading model from', cpfp)
    start_checkpoint = torch.load(cpfp)
    model = start_checkpoint['model']
    model.load_state_dict(start_checkpoint['state_dict'])
    return model

# =====================================================================================
#                    GET PARAMETERS
# =====================================================================================
# Read arguments from command line

parser = argparse.ArgumentParser()
# NEURAL NETWORK PARAMS
parser.add_argument('-emb', '--embedding_type', type=str, default='USE')
parser.add_argument('-spl', '--split_type', type=str, default='fan')
parser.add_argument('-start', '--start_checkpoint', type=int, default=0)
parser.add_argument('-ep', '--epochs', type=int, default=10)
parser.add_argument('-bs', '--batch_size', type=int, default=16)
parser.add_argument('-inf', '--step_info_every', type=int, default=1000)
parser.add_argument('-cp', '--save_epoch_cp_every', type=int, default=1)
parser.add_argument('-lr', '--learning_rate', type=float, default=2e-3)
parser.add_argument('-hid', '--hidden_size', type=float, default=32)
parser.add_argument('-sv', '--seed_val', type=int, default=124)
parser.add_argument('-mx', '--max_len', type=int, default=96)
parser.add_argument('-eval', '--eval', action='store_true', default=False)
parser.add_argument('-nopad', '--no_padding', action='store_true', default=False)
args = parser.parse_args()

NO_PADDING = args.no_padding
N_EPOCHS = args.epochs
HIDDEN = args.hidden_size
BATCH_SIZE = args.batch_size
LR = args.learning_rate
START_CHECKPOINT = args.start_checkpoint
SEED_VAL = args.seed_val
EVAL = args.eval
mode = 'train' if not EVAL else 'eval'
PRINT_STEP_EVERY = args.step_info_every  # steps
SAVE_EPOCH_EVERY = args.save_epoch_cp_every  # epochs
EMB_TYPE = args.embedding_type
SPL = args.split_type

DATA_FP = 'data/cam_input/basil.tsv'
CHECKPOINT_DIR = f'models/checkpoints/cam/{EMB_TYPE}'
REPORTS_DIR = f'reports/cam/{EMB_TYPE}'

if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)
if not os.path.exists(REPORTS_DIR):
    os.makedirs(REPORTS_DIR)

# =====================================================================================
#                    LOGGING INFO ...
# =====================================================================================


if not os.path.exists(REPORTS_DIR):
    os.makedirs(REPORTS_DIR)

now = datetime.now()
now_string = now.strftime(format='%b-%d-%Hh')
log_name = f"{REPORTS_DIR}/{now_string}.log"

console_hdlr = logging.StreamHandler(sys.stdout)
file_hdlr = logging.FileHandler(filename=log_name)
logging.basicConfig(level=logging.INFO, handlers=[console_hdlr, file_hdlr])
logger = logging.getLogger()
logger.info(f"Start Logging to {log_name}")
logger.info(args)

######################################################################
# Get device
device, USE_CUDA = get_torch_device()
random.seed(SEED_VAL)
np.random.seed(SEED_VAL)
torch.manual_seed(SEED_VAL)
torch.cuda.manual_seed_all(SEED_VAL)


######################################################################
# Loading data & weights matrix
# ==================
#

input_lang, triples = readArticles(DATA_FP)
logger.info("***** Loading data *****")
logger.info(f"Read {len(triples)} sentence triples")
logger.info(f"Embedding type: {EMB_TYPE}")
logger.info(f"Max len: {input_lang.max_len}")
logger.info(f"Example: {random.choice(triples)}")

# split data
triple_dict = {trip[0].split(' ')[trip[-1]]: trip for trip in triples}
spl = Split(triple_dict, which=SPL, tst=False)
folds = spl.apply_split(features=['sentence'], input_as='pytorch', output_as='pytorch')

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
weights_matrix = make_weight_matrix(input_lang, embed_fp, EMB_DIM)

######################################################################
# Training
# =======================
#

logger.info("***** Starting training *****")
logger.info(f"Mode: {mode}")
logger.info(f"Num epochs: {N_EPOCHS}")
logger.info(f"Starting from: {START_CHECKPOINT}")
logger.info(f"Batch size: {BATCH_SIZE}")
logger.info(f"Starting LR: {LR}")

eval_df = pd.DataFrame(index=list(range(len(folds))) + ['mean'], columns=['Acc', 'Prec', 'Rec', 'F1'])
for fold_i, fold in enumerate(folds):

    if START_CHECKPOINT > 0: model = load_model_from_checkpoint(CHECKPOINT_DIR, SPL, fold_i, START_CHECKPOINT)
    else: model = ContextAwareModel(input_size=EMB_DIM, hidden_size=HIDDEN, weights_matrix=weights_matrix, device=device)

    model.to(device)
    if USE_CUDA: model.cuda()

    if EVAL:
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.eval()
    else:
        model.train()

    cl = ContextAwareClassifier(model, input_lang, fold['dev'], logger=logger, device=device,
                                batch_size=BATCH_SIZE, cp_dir=CHECKPOINT_DIR, learning_rate=LR,
                                start_checkpoint=START_CHECKPOINT, step_size=1, gamma=0.95)

    if not EVAL:
        cl.train_epochs(fold, num_epochs=N_EPOCHS, print_step_every=PRINT_STEP_EVERY, save_epoch_every=SAVE_EPOCH_EVERY)

    metrics, metrics_df, metrics_string = cl.evaluate(fold['test'], which='all')
    logger.info(metrics_string)
    eval_df.loc[fold_i] = metrics_df[['acc', 'pred', 'rec', 'f1']]

eval_df.loc['mean'] = eval_df.mean()
logger.info(eval_df)
