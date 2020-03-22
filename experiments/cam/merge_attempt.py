import argparse, random, os, sys, logging, time
from datetime import datetime
import torch
import numpy as np
import pandas as pd
from transformers import BertTokenizer
from lib.utils import get_torch_device
from lib.handle_data.SplitData import Split
from lib.classifiers.MergeAttempt import BertForSequenceClassification
from torch.utils.data import (DataLoader, SequentialSampler, RandomSampler, TensorDataset)
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from torch.optim import lr_scheduler
from lib.evaluate.StandardEval import my_eval


class Processor():
    def __init__(self, sentence_ids, max_doc_length, max_sent_length):
        self.sent_id_map = {my_id.lower(): i for i, my_id in enumerate(sentence_ids)}
        #self.id_map_reverse = {i: my_id for i, my_id in enumerate(data_ids)}
        self.EOD_index = len(self.sent_id_map) + 1
        self.PAD_index = self.EOD_index + 1
        self.max_doc_length = max_doc_length + 1
        self.max_sent_length = max_sent_length

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


def make_weight_matrix(data, embed_fp, EMB_DIM):
    embeddings = pd.read_csv(embed_fp, index_col=0).fillna('')
    embeddings = embeddings.rename(
        columns={'USE': 'embeddings', 'sbert_pre': 'embeddings', 'avbert': 'embeddings', 'poolbert': 'embeddings'})
    sentence_embeddings = {i.lower(): np.array(u.strip('[]').split(', ')) for i, u in
                           zip(embeddings.index, embeddings.embeddings)}

    matrix_len = len(data) + 2  # 1 for EOD token and 1 for padding token
    weights_matrix = np.zeros((matrix_len, EMB_DIM))

    sent_id_map = {sent_id: sent_num_id for sent_num_id, sent_id in enumerate(embeddings.index.values)}
    for sent_id, index in sent_id_map.items():  # word here is a sentence id like 91fox27
        if sent_id == '11fox23':
            pass
        else:
            embedding = sentence_embeddings[sent_id]
            weights_matrix[index] = embedding

    return weights_matrix


def to_tensors(split, device):
    # to arrays if needed
    contexts = np.array([list(el) for el in split.context_doc_num.values])
    token_ids, token_mask, tok_seg_ids = [np.array(i) for i in zip(*[tuple(el) for el in split.sentence_num.values])]

    # to tensors
    labels = torch.tensor(split.label.to_numpy(), dtype=torch.long, device=device)
    positions = torch.tensor(split.position.to_numpy(), dtype=torch.long, device=device)
    token_ids = torch.tensor(token_ids, dtype=torch.long, device=device)
    token_mask = torch.tensor(token_mask, dtype=torch.long, device=device)
    tok_seg_ids = torch.tensor(tok_seg_ids, dtype=torch.long, device=device)
    contexts = torch.tensor(contexts, dtype=torch.long, device=device)

    # to dataset
    tensors = TensorDataset(token_ids, token_mask, tok_seg_ids, contexts, labels, positions)

    return tensors


def to_batches(tensors, batch_size):
    sampler = RandomSampler(tensors)
    loader = DataLoader(tensors, sampler=sampler, batch_size=batch_size)
    return loader


def save_bert_model(model_to_save, model_dir, identifier):
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
parser.add_argument('-inf', '--step_info_every', type=int, default=1000)
parser.add_argument('-cp', '--save_epoch_cp_every', type=int, default=1)

# TRAINING PARAMS
parser.add_argument('-spl', '--split_type', type=str, default='berg')
parser.add_argument('-emb', '--embedding_type', type=str, help='Options: avbert|sbert|poolbert|use|finetune', default='finetune')
parser.add_argument('-cn', '--context_naive', action='store_true', help='Turn off bidirectional lstm', default=False)
parser.add_argument('-context', '--context_type', type=str, help='Options: article|story', default='article')
parser.add_argument('-eval', '--eval', action='store_true', default=False)
parser.add_argument('-start', '--start_epoch', type=int, default=0)
parser.add_argument('-ep', '--epochs', type=int, default=1000)

# OPTIMIZING PARAMS
parser.add_argument('-bs', '--batch_size', type=int, default=16)
parser.add_argument('-wu', '--warmup_proportion', type=float, default=0.1)
parser.add_argument('-lr', '--learning_rate', type=float, default=2e-5)
parser.add_argument('-bert_lr', '--bert_learning_rate', type=float, default=1e-5)
parser.add_argument('-g', '--gamma', type=float, default=.95)

# NEURAL NETWORK DIMS
parser.add_argument('-hid', '--hidden_size', type=int, default=50)

# OTHER NN PARAMS
parser.add_argument('-sv', '--seed_val', type=int, default=124)
parser.add_argument('-nopad', '--no_padding', action='store_true', default=False)
parser.add_argument('-bm', '--bert_model', type=str, default='bert-base-cased')
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
WARMUP_PROPORTION = args.warmup_proportion
LR = args.learning_rate
BERT_LR = args.bert_learning_rate
GAMMA = args.gamma

MAX_DOC_LEN = 76 if CONTEXT_TYPE == 'article' else 158
MAX_SENT_LEN = 95
EMB_TYPE = args.embedding_type
EMB_DIM = 512 if EMB_TYPE == 'use' else 768
HIDDEN = args.hidden_size

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
CHECKPOINT_DIR = f'models/checkpoints/cam/{EMB_TYPE}/{SPLIT_TYPE}/{CONTEXT_TYPE}'
BEST_CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, 'best')
REPORTS_DIR = f'reports/cam/{EMB_TYPE}/{SPLIT_TYPE}/{CONTEXT_TYPE}'
CACHE_DIR = 'models/cache/' # This is where BERT will look for pre-trained models to load parameters from.

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

logger.info("============ LOADING DATA =============")
logger.info(f"Embedding type: {EMB_TYPE}")
logger.info(f"Context: {CONTEXT_TYPE}")
logger.info(f"Split type: {SPLIT_TYPE}")
logger.info(f"Max len: {MAX_DOC_LEN}")

DATA_FP = os.path.join(DATA_DIR, 'cam_basil.tsv')
EMBED_FP = f'data/basil_w_{EMB_TYPE}.csv'

#if not os.path.exists(DATA_FP):
string_data_fp = os.path.join(DATA_DIR, 'merged_basil.tsv')
sentences = pd.read_csv('data/basil.csv', index_col=0).fillna('')['sentence'].values
string_data = pd.read_csv(string_data_fp, sep='\t',
                          names=['sentence_ids', 'context_document', 'label', 'position'],
                          dtype={'sentence_ids': str, 'tokens': str, 'label': int, 'position': int})
processor = Processor(sentence_ids=string_data.sentence_ids.values, max_doc_length=MAX_DOC_LEN, max_sent_length=MAX_SENT_LEN)
string_data['sentence'] = sentences
string_data['context_doc_num'] = processor.to_numeric_documents(string_data.context_document.values)
string_data['sentence_num'] = processor.to_numeric_sentences(string_data.sentence.values)
string_data.to_json(DATA_FP)

data = pd.read_json(DATA_FP)
data.index = data.sentence_ids.values

# split data
spl = Split(data, which=SPLIT_TYPE, tst=False, permutation=True)
folds = spl.apply_split(features=['context_doc_num', 'sentence_num', 'position'], input_as='df', output_as='df')
NR_FOLDS = len(folds)

# get embeddings
WEIGHTS_MATRIX = make_weight_matrix(data, EMBED_FP, EMB_DIM)

logger.info(f"--> Read {len(data)} data points")
logger.info(f"--> Example: {data.cut_and_mix_into_ten_folds(n=1).context_doc_num.values}")
logger.info(f"--> Nr folds: {NR_FOLDS}")
logger.info(f"--> Fold sizes: {[f['sizes'] for f in folds]}")
logger.info(f"--> Weight matrix shape: {WEIGHTS_MATRIX.shape}")
logger.info(f"--> Columns: {data.columns}")

# get device
device, USE_CUDA = get_torch_device()
# =====================================================================================
#                    CLASSIFIERS
# =====================================================================================

logger.info("============ TRAINING =============")
logger.info(f"Num epochs: {N_EPOCHS}")
logger.info(f"Starting from: {START_EPOCH}")
logger.info(f"Batch size: {BATCH_SIZE}")
logger.info(f"Starting LR: {LR}")
logger.info(f"Mode: {'train' if not EVAL else 'eval'}")

for fold_i, fold in enumerate(folds):
    logger.info(f'... Fold {fold_i}')
    logger.info(f'#train = {len(fold["train"])}, #dev = {len(fold["dev"])}, #test = {len(fold["test"])}')

    train_batches = to_batches(to_tensors(fold['train'], device), batch_size=BATCH_SIZE)
    dev_batches = to_batches(to_tensors(fold['dev'], device), batch_size=BATCH_SIZE)
    test_batches = to_batches(to_tensors(fold['test'], device), batch_size=BATCH_SIZE)

    #inferencer = Inferencer(REPORTS_DIR, logger, device, use_cuda=USE_CUDA)

    bert_model = BertForSequenceClassification.from_pretrained(BERT_MODEL, cache_dir=CACHE_DIR, num_labels=NUM_LABELS,
                                               output_hidden_states=False, output_attentions=False)

    bert_optimizer = AdamW(bert_model.parameters(), lr=BERT_LR, eps=1e-8)
    num_train_optimization_steps = len(train_batches) * N_EPOCHS
    num_train_warmup_steps = int(WARMUP_PROPORTION * num_train_optimization_steps)
    bert_scheduler = get_linear_schedule_with_warmup(bert_optimizer, num_warmup_steps=num_train_warmup_steps, num_training_steps=num_train_optimization_steps)  # PyTorch scheduler

    bert_model.to(device)
    bert_model.train()

    for ep in range(N_EPOCHS):
        start = time.time()
        epoch_loss = 0
        for step, batch in enumerate(train_batches):
            batch = tuple(t.to(device) for t in batch)
            token_ids, token_masks, tok_seg_ids, contexts, labels, positions = batch

            bert_model.zero_grad()
            loss, probs, sequence_output, pooled_output = bert_model(input_ids=token_ids, attention_mask=token_masks,
                                                              token_type_ids=tok_seg_ids, labels=labels)

            loss.backward()
            bert_optimizer.step()
            bert_scheduler.step()

            epoch_loss += loss.item()

        elapsed = time.time() - start

        bert_model.eval()
        dev_preds = []
        for step, batch in enumerate(dev_batches):
            batch = tuple(t.to(device) for t in batch)
            token_ids, token_masks, tok_seg_ids, contexts, labels, positions = batch

            with torch.no_grad():
                loss, probs, sequence_output, pooled_output = bert_model(input_ids=token_ids, attention_mask=token_masks,
                                                                  token_type_ids=tok_seg_ids, labels=labels)
                probs = probs.detach().cpu().numpy()

            if len(dev_preds) == 0: dev_preds.append(probs)
            else: dev_preds[0] = np.append(dev_preds[0], probs, axis=0)
        dev_preds = np.argmax(dev_preds[0], axis=1)
        bert_model.train()

        _, val_perf_string = my_eval(fold['dev'].label, dev_preds)
        logger.info(f'epoch {ep} (took {elapsed}): Av loss = {epoch_loss / len(train_batches)}, Val perf: {val_perf_string}')
        save_bert_model(bert_model, CHECKPOINT_DIR, f'epoch{ep}')

    # Save final model
    final_name = f'bert_for_embed_finetuned'
    save_bert_model(bert_model, 'models/', final_name)
