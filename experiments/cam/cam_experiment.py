import argparse, os, sys, logging
from datetime import datetime
import torch
import numpy as np
import pandas as pd
from transformers import BertTokenizer
from lib.handle_data.SplitData import Split

from lib.classifiers.ContextAwareClassifier import ContextAwareClassifier
from lib.classifiers.BertWrapper import BertWrapper
from lib.classifiers.Classifier import Classifier
from lib.utils import get_torch_device, to_tensors, to_batches


class Processor():
    def __init__(self, sentence_ids, max_doc_length):
        self.sent_id_map = {my_id.lower(): i for i, my_id in enumerate(sentence_ids)}
        #self.id_map_reverse = {i: my_id for i, my_id in enumerate(data_ids)}
        self.EOD_index = len(self.sent_id_map)
        self.PAD_index = self.EOD_index + 1
        self.max_doc_length = max_doc_length + 1

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

    def to_numeric_sentences(self, sentences):
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

        all_tokens = [tokenizer.tokenize(sent) for sent in sentences]
        max_sent_length = max([len(t) for t in all_tokens])

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

        return token_ids, token_mask, tok_seg_ids


def make_weight_matrix(data, embed_df, EMB_DIM):

    sentence_embeddings = {i.lower(): np.array(u.strip('[]').split(', ')) for i, u in
                           zip(embed_df.index, embed_df.embeddings)}

    matrix_len = len(data) + 2  # 1 for EOD token and 1 for padding token
    weights_matrix = np.zeros((matrix_len, EMB_DIM))

    sent_id_map = {sent_id: sent_num_id for sent_num_id, sent_id in enumerate(embed_df.index.values)}
    for sent_id, index in sent_id_map.items():  # word here is a sentence id like 91fox27
        if sent_id == '11fox23':
            pass
        else:
            embedding = sentence_embeddings[sent_id]
            weights_matrix[index] = embedding

    return weights_matrix


def save_bert_model(model_to_save, model_dir, identifier):
    ''' Save finetuned (finished or intermediate) model to a checkpoint. '''
    output_dir = os.path.join(model_dir, identifier)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
    output_config_file = os.path.join(output_dir, "config.json")

    model_to_save = model_to_save.module if hasattr(model_to_save, 'module') else model_to_save  # Only save the model it-self
    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)


# =====================================================================================
#                    PARAMETERS
# =====================================================================================

# Read arguments from command line

parser = argparse.ArgumentParser()
# PRINT/SAVE PARAMS
parser.add_argument('-inf', '--step_info_every', type=int, default=50)
parser.add_argument('-cp', '--save_epoch_cp_every', type=int, default=50)

# TRAINING PARAMS
parser.add_argument('-spl', '--split_type', help='Options: fan|berg|both',type=str, default='both')
parser.add_argument('-emb', '--embedding_type', type=str, help='Options: avbert|sbert|poolbert|use', default='use')
parser.add_argument('-ft_emb', '--finetune_embeddings', action='store_true', default=True,
                    help='Whether to finetune pretrained BERT embs')
parser.add_argument('-subset', '--subset_of_data', type=float, help='Section of data to experiment on', default=1.0)
parser.add_argument('-context', '--context_type', type=str, help='Options: article|story', default='article')
parser.add_argument('-mode', '--mode', type=str, help='Options: train|eval|debug', default='debug')
parser.add_argument('-start', '--start_epoch', type=int, default=0)
parser.add_argument('-ep', '--epochs', type=int, default=100)
parser.add_argument('-pat', '--patience', type=int, default=5)
#parser.add_argument('-cn', '--context_naive', action='store_true', help='Turn off bidirectional lstm', default=False)

# OPTIMIZING PARAMS
parser.add_argument('-bs', '--batch_size', type=int, default=16)
parser.add_argument('-wu', '--warmup_proportion', type=float, default=0.1)
parser.add_argument('-lr', '--learning_rate', type=float, default=2e-5)
parser.add_argument('-bert_lr', '--bert_learning_rate', type=float, default=1e-5)
parser.add_argument('-g', '--gamma', type=float, default=.95)

# NEURAL NETWORK DIMS
parser.add_argument('-hid', '--hidden_size', type=int, default=50)
parser.add_argument('-lay', '--bilstm_layers', type=int, default=4)

# OTHER NN PARAMS
parser.add_argument('-sv', '--seed_val', type=int, default=124)
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
if DEBUG:
    SUBSET = 0.5

START_EPOCH = args.start_epoch
N_EPOCHS = args.epochs
if DEBUG:
    N_EPOCHS = 3

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

HIDDEN = args.hidden_size
BILSTM_LAYERS = args.bilstm_layers
if DEBUG:
    HIDDEN = 2
    BILSTM_LAYERS = 2

SEED_VAL = args.seed_val
BERT_MODEL = args.bert_model
NUM_LABELS = 2

# set seed
#random.seed(SEED_VAL)
#np.random.seed(SEED_VAL)
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
EMBED_FP = f'data/basil_w_{EMB_TYPE}.csv'

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

logger.info(f"Start Logging to {LOG_NAME}")
logger.info(args)


# =====================================================================================
#                    PREPROCESS DATA
# =====================================================================================

#if not os.path.exists(DATA_FP):

logger.info("============ PREPROCESS DATA =============")
logger.info(f" Writing to: {DATA_FP}")
logger.info(f" Max len: {MAX_DOC_LEN}")

string_data_fp = os.path.join(DATA_DIR, 'merged_basil.tsv')
sentences = pd.read_csv('data/basil.csv', index_col=0).fillna('')['sentence'].values
string_data = pd.read_csv(string_data_fp, sep='\t',
                          names=['sentence_ids', 'context_document', 'label', 'position'],
                          dtype={'sentence_ids': str, 'tokens': str, 'label': int, 'position': int})
processor = Processor(sentence_ids=string_data.sentence_ids.values, max_doc_length=MAX_DOC_LEN)
string_data['sentence'] = sentences
string_data['context_doc_num'] = processor.to_numeric_documents(string_data.context_document.values)
token_ids, token_mask, tok_seg_ids = processor.to_numeric_sentences(string_data.sentence.values)
string_data['token_ids'], string_data['token_mask'], string_data['tok_seg_ids'] = token_ids, token_mask, tok_seg_ids

string_data.to_json(DATA_FP)

# =====================================================================================
#                    LOAD DATA
# =====================================================================================

logger.info("============ LOADING DATA =============")
logger.info(f" Context: {CONTEXT_TYPE}")
logger.info(f" Split type: {SPLIT_TYPE}")
logger.info(f" Max len: {MAX_DOC_LEN}")

data = pd.read_json(DATA_FP)
data.index = data.sentence_ids.values

# split data
spl = Split(data, which=SPLIT_TYPE, subset=SUBSET)

folds = spl.apply_split(features=['context_doc_num', 'token_ids', 'token_mask', 'tok_seg_ids', 'position'])
if DEBUG:
    folds = [folds[0], folds[1]]
NR_FOLDS = len(folds)

# batch data
for fold_i, fold in enumerate(folds):
    train_batches = to_batches(to_tensors(fold['train'], device), batch_size=BATCH_SIZE)
    dev_batches = to_batches(to_tensors(fold['dev'], device), batch_size=BATCH_SIZE)
    test_batches = to_batches(to_tensors(fold['test'], device), batch_size=BATCH_SIZE)

    fold['train_batches'] = train_batches
    fold['dev_batches'] = dev_batches
    fold['test_batches'] = test_batches

logger.info(f" --> Read {len(data)} data points")
logger.info(f" --> Example: {data.sample(n=1).context_doc_num.values}")
logger.info(f" --> Nr folds: {NR_FOLDS}")
logger.info(f" --> Fold sizes: {[f['sizes'] for f in folds]}")
logger.info(f" --> Columns: {list(data.columns)}")


# =====================================================================================
#                    FINETUNE EMBEDDINGS
# =====================================================================================

if FT_EMB:

    logger.info("============ FINETUNE EMBEDDINGS =============")
    logger.info(f" Embedding type: {EMB_TYPE}")
    logger.info(f" Finetuning LR: {BERT_LR}")
    logger.info(f" Num epochs (same as overall): {N_EPOCHS}")
    logger.info(f" Finetuning batch size (same as overall): {BATCH_SIZE}")
    logger.info(f" Mode: {MODE}")

    finetune_f1s = pd.DataFrame(index=list(range(NR_FOLDS)) + ['mean'], columns=['Acc', 'Prec', 'Rec', 'F1'])

    for fold in folds:
        bert = BertWrapper(BERT_MODEL, cache_dir=CACHE_DIR, cp_dir=CHECKPOINT_DIR, num_labels=NUM_LABELS,
                           bert_lr=BERT_LR, warmup_proportion=WARMUP_PROPORTION, n_epochs=N_EPOCHS,
                           n_train_batches=len(fold['train_batches']))

        logger.info(f' Finetuning on fold {fold["name"]} (sizes: {fold["sizes"]})')
        cl = Classifier(logger=logger, model=bert, n_epochs=N_EPOCHS, patience=PATIENCE,
                        fig_dir=FIG_DIR, model_name='bert', print_every=PRINT_STEP_EVERY)
        cl.train_on_fold(fold)
        finetune_f1s.loc[fold['name']] = cl.test_perf

    finetune_f1s.loc['mean'] = finetune_f1s.mean()
    logger.info(f'Finetuning results:\n{finetune_f1s}')

    all_batches = to_batches(to_tensors(data, device), batch_size=BATCH_SIZE)
    data['embeddings'] = bert.get_embeddings(all_batches, emb_type=EMB_TYPE)
    data_w_embeds = data
    data_w_embeds.to_csv(EMBED_FP)

# =====================================================================================
#                    LOAD EMBEDDINGS
# =====================================================================================

logger.info("============ LOAD EMBEDDINGS =============")
logger.info(f" Embedding type: {EMB_TYPE}")

# read embeddings file
data_w_embeds = pd.read_csv(EMBED_FP, index_col=0).fillna('')
data_w_embeds = data_w_embeds.rename(
    columns={'USE': 'embeddings', 'sbert_pre': 'embeddings', 'avbert': 'embeddings', 'poolbert': 'embeddings'})
data_w_embeds.index = [el.lower() for el in data_w_embeds.index]

# transform into matrix
WEIGHTS_MATRIX = make_weight_matrix(data, data_w_embeds, EMB_DIM)

logger.info(f" --> Weight matrix shape: {WEIGHTS_MATRIX.shape}")


# =====================================================================================
#                    CONTEXT AWARE MODEL
# =====================================================================================

logger.info("============ TRAINING =============")
logger.info(f" Num epochs: {N_EPOCHS}")
logger.info(f" Starting from: {START_EPOCH}")
logger.info(f" Nr layers: {BILSTM_LAYERS}")
logger.info(f" Batch size: {BATCH_SIZE}")
logger.info(f" Starting LR: {LR}")
logger.info(f" Mode: {'train' if not EVAL else 'eval'}")

cam_f1s = pd.DataFrame(index=list(range(NR_FOLDS)) + ['mean'], columns=['Acc', 'Prec', 'Rec', 'F1'])
for fold in folds:
    cam = ContextAwareClassifier(start_epoch=START_EPOCH, cp_dir=CHECKPOINT_DIR,
                                 train_labels=fold['train'].label.values, weights_matrix=WEIGHTS_MATRIX,
                                 emb_dim=EMB_DIM, hidden_size=HIDDEN, bilstm_layers=BILSTM_LAYERS,
                                 batch_size=BATCH_SIZE, learning_rate=LR, step_size=1, gamma=GAMMA)

    logger.info(f' CAM Training on fold {fold["name"]} (sizes: {fold["sizes"]})')
    cl = Classifier(logger=logger, model=cam, n_epochs=N_EPOCHS, patience=PATIENCE, fig_dir=FIG_DIR, model_name='cam',
                    print_every=PRINT_STEP_EVERY)
    cl.train_on_fold(fold)
    cam_f1s.loc[fold['name']] = cl.test_perf

# final results of cross validation
cam_f1s.loc['mean'] = cam_f1s.mean()
logger.info(f' CAM Results:\n{cam_f1s}')
