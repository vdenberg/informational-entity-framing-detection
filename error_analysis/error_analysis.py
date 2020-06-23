import argparse, os, sys, logging, re
from datetime import datetime
import random

import torch
import numpy as np
import pandas as pd

from lib.classifiers.ContextAwareClassifier import ContextAwareClassifier
import pickle, time

from lib.classifiers.Classifier import Classifier
from lib.handle_data.SplitData import Split
from lib.utils import get_torch_device, standardise_id, to_batches, to_tensors
from lib.evaluate.Eval import my_eval


def make_weight_matrix(embed_df, EMB_DIM):
    # clean embedding string
    embed_df = embed_df.fillna(0).replace({'\n', ' '})
    sentence_embeddings = {}
    for index, emb in zip(embed_df.index, embed_df.embeddings):
        if emb != 0:
            # emb = re.sub('\s+', ' ', emb)
            # emb = emb[6:-17]
            emb = re.sub('[\(\[\]\)]', '', emb)
            emb = emb.split(', ')
            emb = np.array(emb, dtype=float)
        sentence_embeddings[index.lower()] = emb

    matrix_len = len(embed_df) + 2  # 1 for EOD token and 1 for padding token
    weights_matrix = np.zeros((matrix_len, EMB_DIM))

    sent_id_map = {sent_id.lower(): sent_num_id + 1 for sent_num_id, sent_id in enumerate(embed_df.index.values)}
    for sent_id, index in sent_id_map.items():  # word here is a sentence id like 91fox27
        if sent_id == '11fox23':
            pass
        else:
            embedding = sentence_embeddings[sent_id]
            weights_matrix[index] = embedding

    return weights_matrix


def get_weights_matrix(data, emb_fp, emb_dim=None):
    data_w_emb = pd.read_csv(emb_fp, index_col=0).fillna('')
    data_w_emb = data_w_emb.rename(
        columns={'USE': 'embeddings', 'sbert_pre': 'embeddings', 'avbert': 'embeddings', 'poolbert': 'embeddings',
                 'unpoolbert': 'embeddings', 'crossbert': 'embeddings', 'cross4bert': 'embeddings'})
    data_w_emb.index = [standardise_id(el) for el in data_w_emb.index]
    data.index = [standardise_id(el) for el in data.index]
    #tmp = set(data.index) - set(data_w_emb.index)
    data.loc[data_w_emb.index, 'embeddings'] = data_w_emb['embeddings']
    # transform into matrix
    wm = make_weight_matrix(data, emb_dim)
    return wm


# =====================================================================================
#                    PARAMETERS
# =====================================================================================

# Read arguments from command line

parser = argparse.ArgumentParser()
# PRINT/SAVE PARAMS
parser.add_argument('-inf', '--step_info_every', type=int, default=250)
parser.add_argument('-cp', '--save_epoch_cp_every', type=int, default=50)

# DATA PARAMS
parser.add_argument('-spl', '--split_type', help='Options: fan|berg|both', type=str, default='berg')
parser.add_argument('-subset', '--subset_of_data', type=float, help='Section of data to experiment on', default=1.0)
parser.add_argument('-pp', '--preprocess', action='store_true', default=False, help='Whether to proprocess again')

# EMBEDDING PARAMS
# parser.add_argument('-ft_emb', '--finetune_embeddings', action='store_true', default=False,
# help='Whether to finetune pretrained BERT embs')
parser.add_argument('-emb', '--embedding_type', type=str, help='Options: avbert|sbert|poolbert|use|crossbert',
                    default='cross4bert')

# TRAINING PARAMS
parser.add_argument('-context', '--context_type', type=str, help='Options: article|story', default='article')
parser.add_argument('-cam_type', '--cam_type', type=str, help='Options: cam|cam+|cam++|cam+*|cam+#', default='cam')

parser.add_argument('-mode', '--mode', type=str, help='Options: train|eval|debug', default='train')
parser.add_argument('-start', '--start_epoch', type=int, default=0)
parser.add_argument('-ep', '--epochs', type=int, default=150)  # 75
parser.add_argument('-pat', '--patience', type=int, default=5)  # 15

# OPTIMIZING PARAMS
parser.add_argument('-bs', '--batch_size', type=int, default=32)
parser.add_argument('-wu', '--warmup_proportion', type=float, default=0.1)
parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
parser.add_argument('-g', '--gamma', type=float, default=.95)

# NEURAL NETWORK DIMS
parser.add_argument('-hid', '--hidden_size', type=int, default=600)
parser.add_argument('-lay', '--bilstm_layers', type=int, default=2)

# OTHER NN PARAMS
parser.add_argument('-sampler', '--sampler', type=str, default='sequential')
parser.add_argument('-sv', '--seed_val', type=int, default=34)
parser.add_argument('-nopad', '--no_padding', action='store_true', default=False)
parser.add_argument('-bm', '--bert_model', type=str, default='bert-base-cased')
# GRADIENT_ACCUMULATION_STEPS = 1

args = parser.parse_args()

# set to variables for readability
PRINT_STEP_EVERY = args.step_info_every  # steps
SAVE_EPOCH_EVERY = args.save_epoch_cp_every  # epochs

MODE = args.mode
TRAIN = True if args.mode != 'eval' else False
EVAL = True if args.mode == 'eval' else False
DEBUG = True if args.mode == 'debug' else False

SPLIT_TYPE = args.split_type
CONTEXT_TYPE = args.context_type
SUBSET = args.subset_of_data
PREPROCESS = args.preprocess
# if DEBUG:
#    SUBSET = 0.5

START_EPOCH = args.start_epoch
N_EPOCHS = args.epochs
if DEBUG:
    N_EPOCHS = 20

BATCH_SIZE = args.batch_size
WARMUP_PROPORTION = args.warmup_proportion
LR = args.learning_rate
GAMMA = args.gamma
PATIENCE = args.patience

MAX_DOC_LEN = 76 if CONTEXT_TYPE == 'article' else 158
EMB_TYPE = args.embedding_type
EMB_DIM = 512 if EMB_TYPE == 'use' else 768

CAM_TYPE = args.cam_type
# if DEBUG:
#    CN = True

HIDDEN = args.hidden_size if CAM_TYPE == 'cam' else args.hidden_size * 2
BILSTM_LAYERS = args.bilstm_layers
# if DEBUG:
#    HIDDEN = 2
#    BILSTM_LAYERS = 2

SEED_VAL = args.seed_val
BERT_MODEL = args.bert_model
NUM_LABELS = 2
SAMPLER = args.sampler

# set seed
# random.seed(SEED_VAL)
# np.random.seed(SEED_VAL)
# torch.manual_seed(SEED_VAL)
# torch.cuda.manual_seed_all(SEED_VAL)

# set directories
DATA_DIR = f'data/sent_clf/cam_input/{CONTEXT_TYPE}'
CHECKPOINT_DIR = f'models/checkpoints/cam/{EMB_TYPE}/{SPLIT_TYPE}/{CONTEXT_TYPE}/subset{SUBSET}'
BEST_CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, 'best')
REPORTS_DIR = f'reports/cam/{EMB_TYPE}/{SPLIT_TYPE}/{CONTEXT_TYPE}/subset{SUBSET}'
FIG_DIR = f'figures/cam/{EMB_TYPE}/{SPLIT_TYPE}/{CONTEXT_TYPE}/subset{SUBSET}'
CACHE_DIR = 'models/cache/'  # This is where BERT will look for pre-trained models to load parameters from.
DATA_FP = os.path.join(DATA_DIR, 'cam_basil.tsv')
TABLE_DIR = f"reports/cam/tables/{EMB_TYPE}_{CONTEXT_TYPE}"

if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)
if not os.path.exists(BEST_CHECKPOINT_DIR):
    os.makedirs(BEST_CHECKPOINT_DIR)

# set device
device, USE_CUDA = get_torch_device()
if not USE_CUDA:
    pass #exit(0)

# set logger
now = datetime.now()
now_string = now.strftime(format='%b-%d-%Hh-%-M')
LOG_NAME = f"{REPORTS_DIR}/{now_string}.log"

console_hdlr = logging.StreamHandler(sys.stdout)
file_hdlr = logging.FileHandler(filename=LOG_NAME)
logging.basicConfig(level=logging.INFO, handlers=[console_hdlr, file_hdlr])
logger = logging.getLogger()

logger.info("============ STARTING =============")
logger.info(args)
logger.info(f" Log file: {LOG_NAME}")
logger.info(f" Good luck!")

model_dir = 'models/checkpoints/cam/berg/cross4bert/berg/article/subset1.0/'

logger.info("============ LOADING DATA =============")
logger.info(f" Context: {CONTEXT_TYPE}")
logger.info(f" Split type: {SPLIT_TYPE}")
logger.info(f" Max doc len: {MAX_DOC_LEN}")

data = pd.read_json(DATA_FP)
data.index = data.sentence_ids.values

spl = Split(data, which=SPLIT_TYPE, subset=SUBSET)
folds = spl.apply_split(features=['story', 'source', 'id_num', 'context_doc_num', 'token_ids', 'token_mask', 'position', 'quartile', 'src_num'])
if DEBUG:
    folds = [folds[0], folds[1]]
NR_FOLDS = len(folds)

folds = [folds[4]]


for fold in folds:
    train_batches = to_batches(to_tensors(split=fold['train'], device=device), batch_size=BATCH_SIZE, sampler=SAMPLER)
    dev_batches = to_batches(to_tensors(split=fold['dev'], device=device), batch_size=BATCH_SIZE, sampler=SAMPLER)
    test_batches = to_batches(to_tensors(split=fold['test'], device=device), batch_size=BATCH_SIZE, sampler=SAMPLER)

    fold['train_batches'] = train_batches
    fold['dev_batches'] = dev_batches
    fold['test_batches'] = test_batches


logger.info("============ LOAD EMBEDDINGS =============")
logger.info(f" Embedding type: {EMB_TYPE}")


for fold in folds:
    # read embeddings file
    if EMB_TYPE not in ['use', 'sbert']:
        # embed_fp = f"data/bert_231_bs16_lr2e-05_f{fold['name']}_basil_w_{EMB_TYPE}.csv"
        # embed_fp = f"data/rob_base_sequential_34_bs16_lr1e-05_f{fold['name']}_basil_w_{EMB_TYPE}"
        # embed_fp = f"data/rob_base_sequential_34_bs16_lr1e-05_f{fold['name']}_basil_w_{EMB_TYPE}"
        embed_fp = f"data/rob_tapt_sequential_34_bs16_lr1e-05_f{fold['name']}_basil_w_{EMB_TYPE}"
        weights_matrix = get_weights_matrix(data, embed_fp, emb_dim=EMB_DIM)
        logger.info(f" --> Loaded from {embed_fp}, shape: {weights_matrix.shape}")
    fold['weights_matrix'] = weights_matrix

for fold in folds:
    model_name = f"cam+_68_h1200_bs32_lr0.001_f{fold['name']}"
    model_fp = os.path.join(CHECKPOINT_DIR, model_name)

    cam = ContextAwareClassifier(start_epoch=START_EPOCH, cp_dir=CHECKPOINT_DIR, tr_labs=fold['train'].label,
                                 weights_mat=fold['weights_matrix'], emb_dim=EMB_DIM, hid_size=HIDDEN,
                                 layers=BILSTM_LAYERS,
                                 b_size=BATCH_SIZE, lr=LR, step=1, gamma=GAMMA, cam_type=CAM_TYPE)

    cam_cl = Classifier(model=cam, logger=logger, fig_dir=FIG_DIR, name=fold['name'], patience=PATIENCE, n_eps=N_EPOCHS,
                        printing=PRINT_STEP_EVERY, load_from_ep=None)

    test_preds = cam_cl.produce_preds(fold, model_name=model_name)

    test_df = fold['test']
    test_df['pred'] = test_preds

    # source
    for n, gr in test_df.groupby('source'):
        labels = gr.label
        preds = gr.pred
        source_mets, source_perf = my_eval(labels, preds, name=n, set_type='test')


    # frequent entity

    # lexical cues
