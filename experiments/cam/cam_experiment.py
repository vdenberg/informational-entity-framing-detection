import argparse, os, sys, logging, re
from datetime import datetime
import random

import torch
import numpy as np
import pandas as pd
from transformers import BertTokenizer
from lib.handle_data.SplitData import Split

from lib.classifiers.ContextAwareClassifier import ContextAwareClassifier
from lib.classifiers.BertWrapper import BertForSequenceClassification, load_features
from lib.classifiers.BertForEmbed import Inferencer, InputFeatures

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
parser.add_argument('-cn', '--context_naive', action='store_true', help='Turn off bidirectional lstm', default=False)

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

logger.info(f" --> Read {len(data)} data points")
#ogger.info(f" --> Example: {data.sample(n=1).context_doc_num.values}")
logger.info(f" --> Fold sizes: {[f['sizes'] for f in folds]}")
logger.info(f" --> Columns: {list(data.columns)}")

# =====================================================================================
#                    BATCH DATA
# =====================================================================================

for fold in folds:
    '''
    train_fp = os.path.join('data/features_for_bert', f"folds/{fold['name']}_train_features.pkl")
    dev_fp = os.path.join('data/features_for_bert', f"folds/{fold['name']}_dev_features.pkl")
    test_fp = os.path.join('data/features_for_bert', f"folds/{fold['name']}_test_features.pkl")

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

    train_batches = to_batches(to_tensors(split=fold['train'], device=device), batch_size=BATCH_SIZE)
    dev_batches = to_batches(to_tensors(split=fold['dev'], device=device), batch_size=BATCH_SIZE)
    test_batches = to_batches(to_tensors(split=fold['test'], device=device), batch_size=BATCH_SIZE)

    fold['train_batches'] = train_batches
    fold['dev_batches'] = dev_batches
    fold['test_batches'] = test_batches



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
#                    LOAD EMBEDDINGS
# =====================================================================================

logger.info("============ LOAD EMBEDDINGS =============")
logger.info(f" Embedding type: {EMB_TYPE}")

'''
models = {1: ('models/checkpoints/bert_baseline/bert_26354_bs16_lr2e-05_f1_ep1', 42.449999999999996), 2: ('models/checkpoints/bert_baseline/bert_26354_bs16_lr2e-05_f2_ep4', 37.88), 3: ('models/checkpoints/bert_baseline/bert_26354_bs16_lr2e-05_f3_ep2', 45.97), 4: ('models/checkpoints/bert_baseline/bert_26354_bs16_lr2e-05_f4_ep1', 37.59), 5: ('models/checkpoints/bert_baseline/bert_26354_bs16_lr2e-05_f5_ep3', 34.410000000000004), 6: ('models/checkpoints/bert_baseline/bert_26354_bs16_lr2e-05_f6_ep4', 26.029999999999998), 7: ('models/checkpoints/bert_baseline/bert_26354_bs16_lr2e-05_f7_ep3', 32.629999999999995), 8: ('models/checkpoints/bert_baseline/bert_26354_bs16_lr2e-05_f8_ep4', 26.97), 9: ('models/checkpoints/bert_baseline/bert_26354_bs16_lr2e-05_f9_ep4', 37.169999999999995), 10: ('models/checkpoints/bert_baseline/bert_26354_bs16_lr2e-05_f10_ep3', 32.23)}
all_ids, all_batches, all_labels = load_features('data/features_for_bert/all_features.pkl', batch_size=1)
                        
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
    embed_fp = f"data/bert_231_bs16_lr2e-05_f{fold['name']}_basil_w_{EMB_TYPE}.csv"
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

table_columns = 'model,seed,bs,lr,model_loc,fold,epoch,set_type,loss,acc,prec,rec,f1,fn,fp,tn,tp'
main_results_table = pd.DataFrame(columns=table_columns.split(','))
for SEED in [231]:
    if SEED == 0:
        SEED_VAL = random.randint(0, 300)
    else:
        SEED_VAL = SEED

    seed_name = f"cnm_{SEED_VAL}"
    random.seed(SEED_VAL)
    np.random.seed(SEED_VAL)
    torch.manual_seed(SEED_VAL)
    torch.cuda.manual_seed_all(SEED_VAL)

    for BATCH_SIZE in [BATCH_SIZE]:
        bs_name = seed_name + f"_bs{BATCH_SIZE}"
        for LR in [LR]:
            setting_name = bs_name + f"_lr{LR}"
            setting_results_table = pd.DataFrame(columns=table_columns.split(','))

            for fold in folds:
                fold_results_table = pd.DataFrame(columns=table_columns.split(','))

                val_results = {'model': 'cnm', 'fold': fold["name"], 'seed': SEED_VAL, 'bs': BATCH_SIZE, 'lr': LR, 'set_type': 'dev'}
                test_results = {'model': 'cnm', 'fold': fold["name"], 'seed': SEED_VAL, 'bs': BATCH_SIZE, 'lr': LR, 'set_type': 'test'}

                logger.info(f"--------------- FOLD {fold['name']} ---------------")
                name_base = f"s{SEED_VAL}_f{fold['name']}_{'wm'}_bs{BATCH_SIZE}"
                logger.info(f" Nr batches: {len(fold['train_batches'])}")

                testme = True
                if testme:
                    logger.info(f"------------ CNM ON FOLD {fold['name']} ------------")
                    name_base = 'bert_' + name_base
                    cnm = ContextAwareClassifier(start_epoch=START_EPOCH, cp_dir=CHECKPOINT_DIR, tr_labs=fold['train'].label,
                                                 weights_mat=fold['weights_matrix'], emb_dim=EMB_DIM, hid_size=HIDDEN, layers=BILSTM_LAYERS,
                                                b_size=BATCH_SIZE, lr=LR, step=1, gamma=GAMMA, context_naive=True, super_naive=True)

                    cnm_cl = Classifier(model=cnm, logger=logger, fig_dir=FIG_DIR, name=name_base, patience=PATIENCE, n_eps=N_EPOCHS,
                                        printing=PRINT_STEP_EVERY, load_from_ep=None)

                    best_val_mets, test_mets = cnm_cl.train_on_fold(fold)
                    val_results[0].update(best_val_mets)
                    val_results.update({'model_loc': cnm_cl.best_model_loc})
                    test_results[0].update(test_mets)

                if not CN:
                    logger.info(f"------------ CAM ON FOLD {fold['name']} ------------")
                    name_base = 'cam_' + name_base

                    cam = ContextAwareClassifier(start_epoch=START_EPOCH, cp_dir=CHECKPOINT_DIR, tr_labs=fold['train'].label,
                                                 weights_mat=fold['weights_matrix'], emb_dim=EMB_DIM, hid_size=HIDDEN,
                                                 layers=BILSTM_LAYERS, n_eps=N_EPOCHS,
                                                 b_size=BATCH_SIZE, lr=LR, step=1, gamma=GAMMA,
                                                 context_naive=False, super_naive=True)

                    cam_cl = Classifier(model=cam, logger=logger, fig_dir=FIG_DIR, name=name_base, patience=PATIENCE, n_eps=N_EPOCHS,
                                        printing=PRINT_STEP_EVERY, load_from_ep=None)

                    best_val_mets, test_mets, _ = cam_cl.train_on_fold(fold)
                    val_results[1].update(best_val_mets)
                    test_results[1].update(test_mets)

                fold_results_table = fold_results_table.append(val_results, ignore_index=True)
                fold_results_table = fold_results_table.append(test_results, ignore_index=True)
                logging.info(f'Fold {fold["name"]} results: \n{fold_results_table[["model", "seed", "bs", "lr", "fold", "set_type", "f1"]]}')
                setting_results_table = setting_results_table.append(fold_results_table)
                logger.info(f"Logged to: {LOG_NAME}.")

            logging.info(f'Setting {setting_name} results: \n{setting_results_table[["model", "seed", "bs", "lr", "fold", "set_type", "f1"]]}')
            setting_results_table.to_csv(f'reports/cam/tables/{setting_name}_results_table.csv', index=False)
            main_results_table = main_results_table.append(setting_results_table, ignore_index=True)
        main_results_table.to_csv(f'reports/cam/tables/main_results_table.csv', index=False)
        logger.info(f"Logged to: {LOG_NAME}.")