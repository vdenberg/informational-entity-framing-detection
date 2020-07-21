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
            #emb = re.sub('\s+', ' ', emb)
            #emb = emb[6:-17]
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
parser.add_argument('-inf', '--step_info_every', type=int, default=250)
parser.add_argument('-cp', '--save_epoch_cp_every', type=int, default=50)

# DATA PARAMS
parser.add_argument('-spl', '--split_type', help='Options: fan|berg|both',type=str, default='berg')
parser.add_argument('-subset', '--subset_of_data', type=float, help='Section of data to experiment on', default=1.0)
parser.add_argument('-pp', '--preprocess', action='store_true', default=False, help='Whether to proprocess again')

# EMBEDDING PARAMS
# parser.add_argument('-ft_emb', '--finetune_embeddings', action='store_true', default=False,
# help='Whether to finetune pretrained BERT embs')
parser.add_argument('-emb', '--embedding_type', type=str, help='Options: avbert|sbert|poolbert|use|crossbert', default='cross4bert')

# TRAINING PARAMS
parser.add_argument('-lex', '--lex', action='store_true', default=False, help='lex')
parser.add_argument('-context', '--context_type', type=str, help='Options: article|story', default='article')
parser.add_argument('-cam_type', '--cam_type', type=str, help='Options: cam|cam+|cam++|cam+*|cam+#', default='cam')
parser.add_argument('-base', '--base', type=str, help='Options: base|tapt', default='base')

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
#GRADIENT_ACCUMULATION_STEPS = 1

args = parser.parse_args()

# set to variables for readability
PRINT_STEP_EVERY = args.step_info_every  # steps
SAVE_EPOCH_EVERY = args.save_epoch_cp_every  # epochs

MODE = args.mode
TRAIN = True if args.mode != 'eval' else False
EVAL = True if args.mode == 'eval' else False
DEBUG = True if args.mode == 'debug' else False

LEX = args.lex
SPLIT_TYPE = args.split_type
CONTEXT_TYPE = args.context_type
SUBSET = args.subset_of_data
PREPROCESS = args.preprocess
#if DEBUG:
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
BASE = args.base
#if DEBUG:
#    CN = True

HIDDEN = args.hidden_size if CAM_TYPE == 'cam' else args.hidden_size * 2
BILSTM_LAYERS = args.bilstm_layers
#if DEBUG:
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
CACHE_DIR = 'models/cache/' # This is where BERT will look for pre-trained models to load parameters from.
DATA_FP = os.path.join(DATA_DIR, 'cam_basil.tsv')
TABLE_DIR = f"reports/cam/tables/{EMB_TYPE}_{CONTEXT_TYPE}"

if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)
if not os.path.exists(BEST_CHECKPOINT_DIR):
    os.makedirs(BEST_CHECKPOINT_DIR)
if not os.path.exists(REPORTS_DIR):
    os.makedirs(REPORTS_DIR)
if not os.path.exists(FIG_DIR):
    os.makedirs(FIG_DIR)
if not os.path.exists(TABLE_DIR):
    os.makedirs(TABLE_DIR)

# set device
device, USE_CUDA = get_torch_device()
if not USE_CUDA:
    exit(0)

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

if PREPROCESS:
    logger.info("============ PREPROCESS DATA =============")
    logger.info(f" Writing to: {DATA_FP}")
    logger.info(f" Max doc len: {MAX_DOC_LEN}")

    sentences = pd.read_csv('data/basil.csv', index_col=0).fillna('')
    sentences.index = [el.lower() for el in sentences.index]
    sentences.source = [el.lower() for el in sentences.source]

    raw_data_fp = os.path.join(DATA_DIR, 'merged_basil.tsv')
    raw_data = pd.read_csv(raw_data_fp, sep='\t',
                           names=['sentence_ids', 'context_document', 'label', 'position'],
                           dtype={'sentence_ids': str, 'tokens': str, 'label': int, 'position': int}, index_col=False)
    raw_data = raw_data.set_index('sentence_ids', drop=False)

    raw_data['source'] = sentences['source']
    raw_data['src_num'] = raw_data.source.apply(lambda x: {'fox': 0, 'nyt': 1, 'hpo': 2}[x])
    raw_data['story'] = sentences['story']
    raw_data['sentence'] = sentences['sentence']

    if LEX:
        raw_data['label'] = sentences['lex_bias']
        print('label is lex bias')

    raw_data['doc_len'] = raw_data.context_document.apply(lambda x: len(x.split(' ')))

    quartiles = []
    for position, doc_len in zip(raw_data.position, raw_data.doc_len):
        relative_pos = position / doc_len
        if relative_pos < .25:
            q = 0
        elif relative_pos < .5:
            q = 1
        elif relative_pos < .75:
            q = 2
        else:
            q = 3
        quartiles.append(q)

    raw_data['quartile'] = quartiles

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

'''
pos_cases = data[data.label == 1]
pos_cases = pd.concat([pos_cases]*5)
print(len(data))
print(len(pos_cases))
data = pd.concat(data, pos_cases)
print(len(data))
print(len(data))
'''
spl = Split(data, which=SPLIT_TYPE, subset=SUBSET)
folds = spl.apply_split(features=['story', 'source', 'id_num', 'context_doc_num', 'token_ids', 'token_mask', 'position', 'quartile', 'src_num'])
if DEBUG:
    folds = [folds[0]] #, folds[1]
NR_FOLDS = len(folds)

# folds = [folds[4]]

logger.info(f" --> Read {len(data)} data points")
#ogger.info(f" --> Example: {data.sample(n=1).context_doc_num.values}")
logger.info(f" --> Fold sizes: {[f['sizes'] for f in folds]}")
logger.info(f" --> Columns: {list(data.columns)}")

# =====================================================================================
#                    BATCH DATA
# =====================================================================================

for fold in folds:
    '''
    train_fp = os.path.join('data/sent_clf/features_for_bert', f"folds/{fold['name']}_train_features.pkl")
    dev_fp = os.path.join('data/sent_clf/features_for_bert', f"folds/{fold['name']}_dev_features.pkl")
    test_fp = os.path.join('data/sent_clf/features_for_bert', f"folds/{fold['name']}_test_features.pkl")

    with open(train_fp, "rb") as f:
        train_features = pickle.load(f)

    with open(dev_fp, "rb") as f:
        dev_features = pickle.load(f)

    with open(test_fp, "rb") as f:
        test_features = pickle.load(f)
    '''

    #train_batches = to_batches(to_tensors(features=train_features, device=device), batch_size=BATCH_SIZE)
    # dev_batches = to_batches(to_tensors(features=dev_features, device=device), batch_size=BATCH_SIZE)
    # test_batches = to_batches(to_tensors(features=test_features, device=device), batch_size=BATCH_SIZE)

    train_batches = to_batches(to_tensors(split=fold['train'], device=device), batch_size=BATCH_SIZE, sampler=SAMPLER)
    dev_batches = to_batches(to_tensors(split=fold['dev'], device=device), batch_size=BATCH_SIZE, sampler=SAMPLER)
    test_batches = to_batches(to_tensors(split=fold['test'], device=device), batch_size=BATCH_SIZE, sampler=SAMPLER)

    fold['train_batches'] = train_batches
    fold['dev_batches'] = dev_batches
    fold['test_batches'] = test_batches

# =====================================================================================
#                    LOAD EMBEDDINGS
# =====================================================================================

logger.info("============ LOAD EMBEDDINGS =============")
logger.info(f" Embedding type: {EMB_TYPE}")


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


if EMB_TYPE in ['use', 'sbert']:
    embed_fp = f"data/sent_clf/embeddings/basil_w_{EMB_TYPE}.csv"
    weights_matrix = get_weights_matrix(data, embed_fp, emb_dim=EMB_DIM)
    logger.info(f" --> Loaded from {embed_fp}, shape: {weights_matrix.shape}")

for fold in folds:
    # read embeddings file
    if EMB_TYPE not in ['use', 'sbert']:
        # embed_fp = f"data/bert_231_bs16_lr2e-05_f{fold['name']}_basil_w_{EMB_TYPE}.csv"
        # embed_fp = f"data/rob_base_sequential_34_bs16_lr1e-05_f{fold['name']}_basil_w_{EMB_TYPE}"
        # embed_fp = f"data/rob_base_sequential_34_bs16_lr1e-05_f{fold['name']}_basil_w_{EMB_TYPE}"
        embed_fp = f"data/rob_{BASE}_sequential_34_bs16_lr1e-05_f{fold['name']}_basil_w_{EMB_TYPE}"
        weights_matrix = get_weights_matrix(data, embed_fp, emb_dim=EMB_DIM)
        logger.info(f" --> Loaded from {embed_fp}, shape: {weights_matrix.shape}")
    fold['weights_matrix'] = weights_matrix

# =====================================================================================
#                    CONTEXT AWARE MODEL
# =====================================================================================

logger.info("============ TRAINING CAM =============")
logger.info(f" Num epochs: {N_EPOCHS}")
logger.info(f" Starting from: {START_EPOCH}")
logger.info(f" Patience: {PATIENCE}")
logger.info(f" Mode: {'train' if not EVAL else 'eval'}")
logger.info(f" CAM type: {CAM_TYPE}")
logger.info(f" Emb type: {EMB_TYPE}")
logger.info(f" Use cuda: {USE_CUDA}")
logger.info(f" Nr layers: {BILSTM_LAYERS}")

table_columns = 'model,seed,bs,lr,model_loc,fold,epoch,set_type,loss,acc,prec,rec,f1,fn,fp,tn,tp,h'
main_results_table = pd.DataFrame(columns=table_columns.split(','))

base_name = CAM_TYPE
if LEX:
    base_name += "_lex"

hiddens = [HIDDEN]
batch_sizes = [BATCH_SIZE]
learning_rates = [LR] #, 0.001, 0.002]
seeds = [SEED_VAL] #SEED_VAL, SEED_VAL*2, SEED_VAL*3, 34 68 102 136 170

for HIDDEN in hiddens:
    h_name = f"_h{HIDDEN}"
    for BATCH_SIZE in batch_sizes:
        bs_name = f"_bs{BATCH_SIZE}"
        for LR in learning_rates:
            lr_name = f"_lr{LR}"
            for SEED in seeds:
                if SEED == 0:
                    SEED_VAL = random.randint(0, 300)
                else:
                    SEED_VAL = SEED

                random.seed(SEED_VAL)
                np.random.seed(SEED_VAL)
                torch.manual_seed(SEED_VAL)
                torch.cuda.manual_seed_all(SEED_VAL)

                setting_name = base_name + f"_{SEED_VAL}" + h_name + bs_name + lr_name
                setting_table_fp = f'{TABLE_DIR}/{setting_name}.csv'
                logger.info(f' Setting table in: {setting_table_fp}.')
                FORCE = False
                if os.path.exists(setting_table_fp) and not FORCE:
                    logger.info(f'Setting {setting_name} done already.')
                    setting_results_table = pd.read_csv(setting_table_fp, index_col=None)

                else:
                    setting_results_table = pd.DataFrame(columns=table_columns.split(','))
                    for fold in folds:
                        logger.info(f"--------------- CAM ON FOLD {fold['name']} ---------------")
                        logger.info(f" Hidden layer size: {HIDDEN}")
                        logger.info(f" Batch size: {BATCH_SIZE}")
                        logger.info(f" Starting LR: {LR}")
                        logger.info(f" Seed: {SEED_VAL}")
                        logger.info(f" Nr batches: {len(fold['train_batches'])}")
                        logger.info(f" Logging to: {LOG_NAME}.")
                        fold_name = setting_name + f"_f{fold['name']}"
                        fold_table_fp = f'{TABLE_DIR}/{fold_name}.csv'

                        FORCE = False
                        if os.path.exists(fold_table_fp) and not FORCE:
                            logger.info(f'Fold {fold_name} done already.')
                            fold_results_table = pd.read_csv(fold_table_fp, index_col=None)
                        else:
                            fold_results_table = pd.DataFrame(columns=table_columns.split(','))

                            val_results = {'model': base_name, 'fold': fold["name"], 'seed': SEED_VAL, 'bs': BATCH_SIZE, 'lr': LR, 'h': HIDDEN, 'set_type': 'dev'}
                            test_results = {'model': base_name, 'fold': fold["name"], 'seed': SEED_VAL, 'bs': BATCH_SIZE, 'lr': LR, 'h': HIDDEN, 'set_type': 'test'}

                            cam = ContextAwareClassifier(start_epoch=START_EPOCH, cp_dir=CHECKPOINT_DIR, tr_labs=fold['train'].label,
                                                         weights_mat=fold['weights_matrix'], emb_dim=EMB_DIM, hid_size=HIDDEN, layers=BILSTM_LAYERS,
                                                         b_size=BATCH_SIZE, lr=LR, step=1, gamma=GAMMA, cam_type=CAM_TYPE)

                            cam_cl = Classifier(model=cam, logger=logger, fig_dir=FIG_DIR, name=fold_name, patience=PATIENCE, n_eps=N_EPOCHS,
                                                printing=PRINT_STEP_EVERY, load_from_ep=None)

                            best_val_mets, test_mets = cam_cl.train_on_fold(fold)
                            val_results.update(best_val_mets)
                            val_results.update({'model_loc': cam_cl.best_model_loc})
                            if test_mets:
                                test_results.update(test_mets)

                            fold_results_table = fold_results_table.append(val_results, ignore_index=True)
                            fold_results_table = fold_results_table.append(test_results, ignore_index=True)
                            fold_results_table.to_csv(fold_table_fp, index=False)
                            logging.info(f'Fold {fold["name"]} results: \n{fold_results_table[["model", "seed", "bs", "lr", "fold", "set_type", "f1"]]}')
                        setting_results_table = setting_results_table.append(fold_results_table)

                    logging.info(f'Setting {setting_name} results: \n{setting_results_table[["model", "seed", "bs", "lr", "fold", "set_type", "f1"]]}')
                    setting_results_table.to_csv(setting_table_fp, index=False)
                    main_results_table = main_results_table.append(setting_results_table, ignore_index=True)

            main_results_table.to_csv(f'{TABLE_DIR}/{base_name}_main_results_table_1.csv', index=False)
            logger.info(f"Logged to: {LOG_NAME}.")