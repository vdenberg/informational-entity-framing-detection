import argparse, os, sys, logging, re
from datetime import datetime
import random

import torch
import numpy as np
import pandas as pd
from transformers import BertTokenizer
from lib.handle_data.SplitData import Split

from lib.classifiers.ContextAwareClassifier import ContextAwareClassifier
from lib.classifiers.BertWrapper import BertWrapper

from lib.classifiers.BertForEmbed import BertForSequenceClassification, Inferencer, to_tensor, InputFeatures
from torch.utils.data import (DataLoader, SequentialSampler, TensorDataset)
from lib.evaluate.Eval import eval
import pickle, time
from torch.nn import CrossEntropyLoss, BCELoss
from torch.nn import CrossEntropyLoss, Embedding, Dropout, Linear, Sigmoid, LSTM

from lib.classifiers.Classifier import Classifier
from lib.utils import get_torch_device, to_tensors, to_batches
#from experiments.bert_sentence_embeddings.finetune import OldFinetuner, InputFeatures


class Processor():
    def __init__(self, sentence_ids, max_doc_length):
        self.sent_id_map = {str_i.lower(): i+1 for i, str_i in enumerate(sentence_ids)}
        #self.id_map_reverse = {i: my_id for i, my_id in enumerate(data_ids)}
        self.EOD_index = len(self.sent_id_map)
        self.max_doc_length = max_doc_length + 1 # add 1 for EOD_index
        self.max_sent_length = None # set after processing
        self.PAD_index = 0

    def to_numeric_documents(self, documents):
        numeric_context_docs = []
        for doc in documents:
            doc = doc.split(' ')
            # to indexes
            doc = [self.sent_id_map[sent.lower()] for sent in doc]
            # with EOS token
            doc += [self.EOD_index]
            # padded
            padding = [self.PAD_index] * (self.max_doc_length - len(doc))
            doc += padding
            numeric_context_docs.append(doc)
        return numeric_context_docs

    def to_numeric_sentences(self, sentence_ids):
        with open("data/features_for_bert/folds/all_features.pkl", "rb") as f:
            features = pickle.load(f)
        feat_dict = {f.my_id.lower(): f for f in features}
        token_ids = [feat_dict[i].input_ids for i in sentence_ids]
        token_mask = [feat_dict[i].input_mask for i in sentence_ids]
        self.max_sent_length = len(token_ids[0])
        '''
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

        all_tokens = [tokenizer.tokenize(sent) for sent in sentences]
        all_tokens = [["[CLS]"] + tokens + ["[SEP]"] for tokens in all_tokens]
        max_sent_length = max([len(t) for t in all_tokens])
        self.max_sent_length = max_sent_length

        token_ids = []
        token_mask = []
        tok_seg_ids = []

        for tokens in all_tokens:
            segment_ids = [0] * len(tokens)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            padding = [0] * (max_sent_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            token_ids.append(input_ids)
            token_mask.append(input_mask)
            tok_seg_ids.append(segment_ids)
        '''
        return token_ids, token_mask


def make_weight_matrix(embed_df, EMB_DIM):
    # clean embedding string
    embed_df = embed_df.fillna(0).replace({'\n', ' '})
    sentence_embeddings = {}
    for index, emb in zip(embed_df.index, embed_df.embeddings):
        if emb != 0:
            emb = re.sub('\s+', ' ', emb)
            emb = emb[6:-17]
            emb = re.sub('[\(\[\]\)]', '', emb)
            emb = emb.split(', ')
            emb = np.array(emb, dtype=float)
        sentence_embeddings[index.lower()] = emb

    matrix_len = len(embed_df) + 2  # 1 for EOD token and 1 for padding token
    weights_matrix = np.zeros((matrix_len, EMB_DIM))

    sent_id_map = {sent_id.lower(): sent_num_id+1 for sent_num_id, sent_id in enumerate(embed_df.index.values)}
    for sent_id, index in sent_id_map.items():  # word here is a sentence id like 91fox27
        if sent_id == '11fox23':
            pass
        else:
            embedding = sentence_embeddings[sent_id]
            weights_matrix[index] = embedding

    return weights_matrix


# =====================================================================================
#                    PARAMETERS
# =====================================================================================

# Read arguments from command line

parser = argparse.ArgumentParser()
# PRINT/SAVE PARAMS
parser.add_argument('-inf', '--step_info_every', type=int, default=100)
parser.add_argument('-cp', '--save_epoch_cp_every', type=int, default=50)

# DATA PARAMS
parser.add_argument('-spl', '--split_type', help='Options: fan|berg|both',type=str, default='berg')
parser.add_argument('-subset', '--subset_of_data', type=float, help='Section of data to experiment on', default=1.0)
parser.add_argument('-pp', '--preprocess', action='store_true', default=False, help='Whether to proprocess again')

# EMBEDDING PARAMS
parser.add_argument('-emb', '--embedding_type', type=str, help='Options: avbert|sbert|poolbert|use', default='poolbert')
parser.add_argument('-ft_emb', '--finetune_embeddings', action='store_true', default=False,
                    help='Whether to finetune pretrained BERT embs')

# TRAINING PARAMS
parser.add_argument('-context', '--context_type', type=str, help='Options: article|story', default='article')
parser.add_argument('-mode', '--mode', type=str, help='Options: train|eval|debug', default='train')
parser.add_argument('-start', '--start_epoch', type=int, default=0)
parser.add_argument('-ep', '--epochs', type=int, default=50)
parser.add_argument('-pat', '--patience', type=int, default=5)
parser.add_argument('-cn', '--context_naive', action='store_true', help='Turn off bidirectional lstm', default=True)

# OPTIMIZING PARAMS
parser.add_argument('-bs', '--batch_size', type=int, default=24)
parser.add_argument('-wu', '--warmup_proportion', type=float, default=0.1)
parser.add_argument('-lr', '--learning_rate', type=float, default=2e-4)
parser.add_argument('-bert_lr', '--bert_learning_rate', type=float, default=2e-5)
parser.add_argument('-g', '--gamma', type=float, default=.95)

# NEURAL NETWORK DIMS
parser.add_argument('-hid', '--hidden_size', type=int, default=50)
parser.add_argument('-lay', '--bilstm_layers', type=int, default=4)

# OTHER NN PARAMS
parser.add_argument('-sv', '--seed_val', type=int, default=263)
parser.add_argument('-nopad', '--no_padding', action='store_true', default=False)
parser.add_argument('-bm', '--bert_model', type=str, default='bert-base-cased')
#GRADIENT_ACCUMULATION_STEPS = 1

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
if DEBUG:
    SUBSET = 0.5

START_EPOCH = args.start_epoch
N_EPOCHS = args.epochs
if DEBUG:
    N_EPOCHS = 20

BATCH_SIZE = args.batch_size
WARMUP_PROPORTION = args.warmup_proportion
LR = args.learning_rate
BERT_LR = args.bert_learning_rate
GAMMA = args.gamma
PATIENCE = args.patience

MAX_DOC_LEN = 76 if CONTEXT_TYPE == 'article' else 158
EMB_TYPE = args.embedding_type
FT_EMB = args.finetune_embeddings
EMB_DIM = 512 if EMB_TYPE == 'use' else 768

CN = args.context_naive
if DEBUG:
    CN = True

HIDDEN = args.hidden_size
BILSTM_LAYERS = args.bilstm_layers
if DEBUG:
    HIDDEN = 2
    BILSTM_LAYERS = 2

SEED_VAL = args.seed_val
BERT_MODEL = args.bert_model
NUM_LABELS = 2

# set seed
random.seed(SEED_VAL)
np.random.seed(SEED_VAL)
torch.manual_seed(SEED_VAL)
torch.cuda.manual_seed_all(SEED_VAL)

# set directories
DATA_DIR = f'data/cam_input/{CONTEXT_TYPE}'
CHECKPOINT_DIR = f'models/checkpoints/cam/{EMB_TYPE}/{SPLIT_TYPE}/{CONTEXT_TYPE}/subset{SUBSET}'
BEST_CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, 'best')
REPORTS_DIR = f'reports/cam/{EMB_TYPE}/{SPLIT_TYPE}/{CONTEXT_TYPE}/subset{SUBSET}'
FIG_DIR = f'figures/cam/{EMB_TYPE}/{SPLIT_TYPE}/{CONTEXT_TYPE}/subset{SUBSET}'
CACHE_DIR = 'models/cache/' # This is where BERT will look for pre-trained models to load parameters from.
DATA_FP = os.path.join(DATA_DIR, 'cam_basil.tsv')

if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)
if not os.path.exists(BEST_CHECKPOINT_DIR):
    os.makedirs(BEST_CHECKPOINT_DIR)
if not os.path.exists(REPORTS_DIR):
    os.makedirs(REPORTS_DIR)
if not os.path.exists(FIG_DIR):
    os.makedirs(FIG_DIR)

# set device
device, USE_CUDA = get_torch_device()

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

# =====================================================================================
#                    PREPROCESS DATA
# =====================================================================================

PREPROCESS = True
if PREPROCESS:
    logger.info("============ PREPROCESS DATA =============")
    logger.info(f" Writing to: {DATA_FP}")
    logger.info(f" Max doc len: {MAX_DOC_LEN}")

    sentences = pd.read_csv('data/basil.csv', index_col=0).fillna('')
    raw_data_fp = os.path.join(DATA_DIR, 'merged_basil.tsv')
    raw_data = pd.read_csv(raw_data_fp, sep='\t',
                           names=['sentence_ids', 'context_document', 'label', 'position'],
                           dtype={'sentence_ids': str, 'tokens': str, 'label': int, 'position': int}, index_col=False)
    raw_data = raw_data.set_index('sentence_ids', drop=False)

    raw_data['source'] = sentences['source']
    raw_data['story'] = sentences['story']
    raw_data['sentence'] = sentences['sentence']

    processor = Processor(sentence_ids=raw_data.sentence_ids.values, max_doc_length=MAX_DOC_LEN)
    raw_data['id_num'] = [processor.sent_id_map[i] for i in raw_data.sentence_ids.values]
    raw_data['context_doc_num'] = processor.to_numeric_documents(raw_data.context_document.values)
    token_ids, token_mask = processor.to_numeric_sentences(raw_data.sentence_ids)
    raw_data['token_ids'], raw_data['token_mask'] = token_ids, token_mask
    raw_data.to_json(DATA_FP)
    logger.info(f" Max sent len: {processor.max_sent_length}")

# =====================================================================================
#                    LOAD DATA
# =====================================================================================

logger.info("============ LOADING DATA =============")
logger.info(f" Context: {CONTEXT_TYPE}")
logger.info(f" Split type: {SPLIT_TYPE}")
logger.info(f" Max doc len: {MAX_DOC_LEN}")

data = pd.read_json(DATA_FP)
data.index = data.sentence_ids.values

spl = Split(data, which=SPLIT_TYPE, subset=SUBSET)
folds = spl.apply_split(features=['story', 'source', 'id_num', 'context_doc_num', 'token_ids', 'token_mask', 'position'])
if DEBUG:
    folds = [folds[0], folds[1]]
NR_FOLDS = len(folds)
folds = [folds[1]]

logger.info(f" --> Read {len(data)} data points")
#ogger.info(f" --> Example: {data.sample(n=1).context_doc_num.values}")
logger.info(f" --> Fold sizes: {[f['sizes'] for f in folds]}")
logger.info(f" --> Columns: {list(data.columns)}")

# =====================================================================================
#                    BATCH DATA
# =====================================================================================

for fold in folds:
    train_batches = to_batches(to_tensors(fold['train'], device=device), batch_size=BATCH_SIZE)
    dev_batches = to_batches(to_tensors(fold['dev'], device=device), batch_size=BATCH_SIZE)
    test_batches = to_batches(to_tensors(fold['test'], device=device), batch_size=BATCH_SIZE)

    fold['train_batches'] = train_batches
    fold['dev_batches'] = dev_batches
    fold['test_batches'] = test_batches

# =====================================================================================
#                    LOAD EMBEDDINGS
# =====================================================================================

logger.info("============ LOAD EMBEDDINGS =============")
logger.info(f" Embedding type: {EMB_TYPE}")

'''
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

with open(f"data/features_for_bert/folds/all_features.pkl", "rb") as f:
    all_ids, all_data, all_labels = to_tensors(pickle.load(f), device)
    bert_all_batches = to_batches(all_data, 1)
        # bert model
    bert_model = BertForSequenceClassification.from_pretrained(model_locs[fold['name']],
                                                               num_labels=2, output_hidden_states=True,
                                                               output_attentions=True)
    
'''
for fold in folds:
    # read embeddings file
    embed_fp = f"data/{fold['name']}_basil_w_{EMB_TYPE}.csv"
    data_w_embeds = pd.read_csv(embed_fp, index_col=0).fillna('')
    data_w_embeds = data_w_embeds.rename(
        columns={'USE': 'embeddings', 'sbert_pre': 'embeddings', 'avbert': 'embeddings', 'poolbert': 'embeddings'})
    data_w_embeds.index = [el.lower() for el in data_w_embeds.index]
    data.loc[data_w_embeds.index, 'embeddings'] = data_w_embeds['embeddings']

    # transform into matrix
    weights_matrix = make_weight_matrix(data, EMB_DIM)

    logger.info(f" --> Loaded from {embed_fp}, shape: {weights_matrix.shape}")
    fold['weights_matrix'] = weights_matrix


# =====================================================================================
#                    GET EMBEDDINGS
# =====================================================================================
'''
logger.info(f"Get embeddings")
    # load bert features
    with open(f"data/features_for_bert/folds/all_features.pkl", "rb") as f:
        all_ids, all_data, all_labels = to_tensor(pickle.load(f), device)
        bert_all_batches = to_batches(all_data, 1)

    # bert model
    bert_model = BertForSequenceClassification.from_pretrained('models/checkpoints/bert_baseline/good_dev_model',
                                                               num_labels=2, output_hidden_states=False,
                                                               output_attentions=False)
    bert_model.to(device)
    bert_model.eval()

    # get embeddings
    bert_embeddings = []
    for bert_batch in bert_all_batches:
        input_ids, input_mask, segment_ids, label_ids = bert_batch
        with torch.no_grad():
            bert_outputs = bert_model(input_ids, segment_ids, input_mask, labels=None)
            logits, probs, sequence_output, pooled_output = bert_outputs
            print(probs, len(probs))
            print(probs.shape)
            emb_output = list(pooled_output.detach().cpu().numpy())
        bert_embeddings.extend(emb_output)
    embed_df = pd.DataFrame(index=all_ids)
    embed_df['embeddings'] = bert_embeddings

    logger.info(data_w_embeds['embeddings'].head(3))
    logger.info(embed_df['embeddings'].head(3))
    exit(0)
'''


# =====================================================================================
#                    CONTEXT AWARE MODEL
# =====================================================================================

logger.info("============ TRAINING CAM =============")
logger.info(f" Num epochs: {N_EPOCHS}")
logger.info(f" Starting from: {START_EPOCH}")
logger.info(f" Nr layers: {BILSTM_LAYERS}")
logger.info(f" Batch size: {BATCH_SIZE}")
logger.info(f" Starting LR: {LR}")
logger.info(f" Seed: {SEED_VAL}")
logger.info(f" Patience: {PATIENCE}")
logger.info(f" Mode: {'train' if not EVAL else 'eval'}")
logger.info(f" Context-naive: {CN}")
logger.info(f" Use cuda: {USE_CUDA}")

#cam_table = pd.read_csv('reports/cam/results_table.csv')
new_cam_table = pd.DataFrame(columns=['seed,fold,epoch,set_type,loss,acc,prec,rec,f1,fn,fp,tn,tp'.split(',')])
for fold in folds:
    logger.info(f"------------ CAM ON FOLD {fold['name']} ------------")
    name_base = f"s{SEED_VAL}_f{fold['name']}_{'cyc'}_bs{BATCH_SIZE}"

    cnm = ContextAwareClassifier(start_epoch=START_EPOCH, cp_dir=CHECKPOINT_DIR, tr_labs=fold['train'].label,
                                 weights_mat=fold['weights_matrix'], emb_dim=EMB_DIM, hid_size=HIDDEN, layers=BILSTM_LAYERS,
                                 b_size=BATCH_SIZE, lr=LR, step=1, gamma=GAMMA, context_naive=CN)

    cam_cl = Classifier(model=cnm, logger=logger, fig_dir=FIG_DIR, name=name_base, patience=PATIENCE, n_eps=N_EPOCHS,
                        printing=PRINT_STEP_EVERY, load_from_ep=None)



    # train
    best_val_mets, test_mets = cam_cl.train_on_fold(fold)
    best_val_mets['seed'], test_mets['seed'] = SEED_VAL, SEED_VAL
    best_val_mets['fold'], test_mets['fold'] = fold["name"], fold["name"]

    new_cam_table = new_cam_table.append(best_val_mets, ignore_index=True)
    new_cam_table = new_cam_table.append(test_mets, ignore_index=True)
    logger.info(f"\n{new_cam_table}")

    #cam_table = cam_table.append(new_cam_table, ignore_index=True)
    #cam_table.to_csv('reports/cam/results_table.csv', index=False)

    logger.info(f"Logged to: {LOG_NAME}.")
    #logger.info(f"Results in: {'reports/cam/results_table.csv'}.")
