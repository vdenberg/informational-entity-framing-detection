import argparse, random, os, sys, datetime, logging
import torch
import numpy as np
import pandas as pd
from transformers import BertTokenizer

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
parser.add_argument('-emb', '--embedding_type', type=str, help='Options: avbert|sbert|poolbert|use|finetune', default='avbert')
parser.add_argument('-ca', '--context_naive', action='store_true', help='Turn off bidirectional lstm', default=False)
parser.add_argument('-context', '--context_type', type=str, help='Options: article|story', default='article')
parser.add_argument('-eval', '--eval', action='store_true', default=False)
parser.add_argument('-start', '--start_epoch', type=int, default=0)
parser.add_argument('-ep', '--epochs', type=int, default=1000)

# OPTIMIZING PARAMS
parser.add_argument('-bs', '--batch_size', type=int, default=24)
parser.add_argument('-wu', '--warmup_proportion', type=float, default=0.1)
parser.add_argument('-lr', '--learning_rate', type=float, default=2e-4)
parser.add_argument('-g', '--gamma', type=float, default=.95)

# NEURAL NETWORK DIMS
parser.add_argument('-hid', '--hidden_size', type=int, default=50)

# OTHER NN PARAMS
parser.add_argument('-sv', '--seed_val', type=int, default=124)
parser.add_argument('-nopad', '--no_padding', action='store_true', default=False)
#GRADIENT_ACCUMULATION_STEPS = 1

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

MAX_DOC_LEN = 76 if CONTEXT_TYPE == 'article' else 158
MAX_SENT_LEN = 95
EMB_TYPE = args.embedding_type
EMB_DIM = 512 if EMB_TYPE == 'use' else 768
HIDDEN = args.hidden_size

SEED_VAL = args.seed_val

# set seed
random.seed(SEED_VAL)
np.random.seed(SEED_VAL)
torch.manual_seed(SEED_VAL)
torch.cuda.manual_seed_all(SEED_VAL)

# set directories
DATA_DIR = f'data/cam_input/{CONTEXT_TYPE}'
CHECKPOINT_DIR = f'models/checkpoints/cam/{EMB_TYPE}/{SPLIT_TYPE}/{CONTEXT_TYPE}'
BEST_CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, 'best')
REPORTS_DIR = f'reports/cam/{EMB_TYPE}/{SPLIT_TYPE}/{CONTEXT_TYPE}'

if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)
if not os.path.exists(BEST_CHECKPOINT_DIR):
    os.makedirs(BEST_CHECKPOINT_DIR)
if not os.path.exists(REPORTS_DIR):
    os.makedirs(REPORTS_DIR)

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
#                    DATA & EMBEDDINGS
# =====================================================================================

logger.info("***** Loading data *****")
logger.info(f"Embedding type: {EMB_TYPE}")
logger.info(f"Context: {CONTEXT_TYPE}")
logger.info(f"Split type: {SPLIT_TYPE}")
logger.info(f"Max len: {MAX_DOC_LEN}")

DATA_FP = os.path.join(DATA_DIR, 'cam_basil.tsv')
EMBED_FP = f'data/basil_w_{EMB_TYPE}.csv'

class Processor():
    def __init__(self, sentence_ids, max_doc_length, max_sent_length):
        self.sent_id_map = {my_id: i for i, my_id in enumerate(sentence_ids)}
        #self.id_map_reverse = {i: my_id for i, my_id in enumerate(data_ids)}
        self.EOD_index = len(self.sent_id_map) + 1
        self.PAD_index = self.EOD_index + 1
        self.max_doc_length = max_doc_length + 1
        self.max_sent_length = max_sent_length

    def to_numeric_documents(self, documents):
        numeric_context_docs = []
        for doc in documents:
            # to indexes
            doc = [self.sent_id_map[sent] for sent in doc]
            # with EOS token
            doc += [self.EOD_index]
            # padded
            padding = [self.PAD_index] * (self.max_doc_length - len(doc))
            doc += padding
            numeric_context_docs.append(doc)
        return numeric_context_docs

    def to_numeric_sentences(self, sentences, from_preprocessed=True):
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
        numeric_sentences = []
        for sent in sentences:
            tokens = tokenizer.tokenize(sent)
            segment_ids = [0] * len(tokens)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            padding = [0] * (self.max_sent_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding
            numeric_sent = (input_ids, input_mask, segment_ids)
            numeric_sentences.append(numeric_sent)
        return numeric_sentences


if not os.path.exists(DATA_FP):
    string_data_fp = os.path.join(DATA_DIR, 'merged_basil.tsv')
    string_data = pd.read_csv(string_data_fp,
                              columns=['sentence_ids', 'context_document', 'sentence', 'label', 'index'],
                              dtype={'my_id': str, 'sentences': str, 'tokens': str, 'label': int, 'index': int})
    processor = Processor(sentence_ids=string_data.sentence_ids.values, max_doc_length=MAX_DOC_LEN, max_sent_length=MAX_SENT_LEN)
    string_data['context_doc_num'] = processor.to_numeric_documents(string_data.context_document.values)
    string_data['sentence_num'] = processor.to_numeric_sentences(string_data.sentence.values)
    string_data.to_json(DATA_FP)
else:
    data = pd.read_json(DATA_FP)

logger.info(f"Done: Read {len(data)} sentence triples")
logger.info(f"Done: Example: {data.sample(n=1).context_doc_num}")
