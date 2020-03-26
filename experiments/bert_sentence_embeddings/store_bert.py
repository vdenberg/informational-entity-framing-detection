from __future__ import absolute_import, division, print_function
from lib.classifiers.BertForEmbed import BertForSequenceClassification, Inferencer
from lib.handle_data.PreprocessForBert import InputFeatures
import torch
from lib.utils import get_torch_device
import pickle, logging, argparse
from lib.handle_data.PreprocessForBert import *
from torch.utils.data import (DataLoader, SequentialSampler, RandomSampler, TensorDataset)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
device, USE_CUDA = get_torch_device()

# structure of project
TASK_NAME = 'bert_for_embed'
DATA_DIR = 'data/features_for_bert/'
CHECKPOINT_DIR = f'models/checkpoints/{TASK_NAME}/'
REPORTS_DIR = f'reports/{TASK_NAME}_evaluation_report/'
CACHE_DIR = 'models/cache/' # This is where BERT will look for pre-trained models to load parameters from.

cache_dir = CACHE_DIR
if os.path.exists(REPORTS_DIR) and os.listdir(REPORTS_DIR):
    REPORTS_DIR += f'/report_{len(os.listdir(REPORTS_DIR))}'
    os.makedirs(REPORTS_DIR)
if not os.path.exists(REPORTS_DIR):
    os.makedirs(REPORTS_DIR)
    REPORTS_DIR += f'/report_{len(os.listdir(REPORTS_DIR))}'
    os.makedirs(REPORTS_DIR)


def to_tensor(features, OUTPUT_MODE):
    example_ids = [f.my_id for f in features]
    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

    if OUTPUT_MODE == "classification":
        label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif OUTPUT_MODE == "regression":
        label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    data = TensorDataset(input_ids, input_mask, segment_ids, label_ids)
    return example_ids, data, label_ids  # example_ids, input_ids, input_mask, segment_ids, label_ids

################
# HYPERPARAMETERS
################


# =====================================================================================
#                    PARAMETERS
# =====================================================================================
# Read arguments from command line

parser = argparse.ArgumentParser()
parser.add_argument('-emb', '--embedding_type', type=str, help='options: avbert|poolbert', default='poolbert')
parser.add_argument('-pth', '--model_path', type=str, default='models/checkpoints/bert_for_embed/epoch17/')
args = parser.parse_args()

MODEL_PATH = args.model_path
EMB_TYPE = args.embedding_type
OUTPUT_MODE = 'classification'
NUM_LABELS = 2

DEV_DATA_FP = DATA_DIR + "folds/fan_dev_features.pkl"
ALL_DATA_FP = DATA_DIR + "folds/all_features.pkl"

# =====================================================================================
#                    DATA
# =====================================================================================
# Read arguments from command line
with open(DEV_DATA_FP, "rb") as f:
    dev_features = pickle.load(f)
with open(ALL_DATA_FP, "rb") as f:
    all_features = pickle.load(f)
dev_ids, dev_data, dev_labels = to_tensor(dev_features, OUTPUT_MODE)
all_ids, all_data, all_labels = to_tensor(all_features, OUTPUT_MODE)

# =====================================================================================
#                    PREDICT
# =====================================================================================
# Read arguments from command line

inferencer = Inferencer(REPORTS_DIR, OUTPUT_MODE, logger, device, USE_CUDA)
if __name__ == '__main__':
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=NUM_LABELS,
                                                          output_hidden_states=True, output_attentions=True)
    logger.info(f'Loaded data from {ALL_DATA_FP}')
    logger.info(f'Loaded model from {MODEL_PATH}')
    inferencer.eval(model, dev_data, dev_labels, set_type='dev', name=f'{MODEL_PATH}-Dev')
    inferencer.eval(model, all_data, all_labels, set_type='all',  name=f'{MODEL_PATH}-All')
    embeddings = inferencer.predict(model, all_data, return_embeddings=True) #, embedding_type=EMB_TYPE)ls
    logger.info(f'Finished {len(embeddings)} embeddings')
    basil_w_BERT = pd.DataFrame(index=all_ids)
    basil_w_BERT[EMB_TYPE] = embeddings
    basil_w_BERT.to_csv(f'data/basil_w_{EMB_TYPE}.csv')