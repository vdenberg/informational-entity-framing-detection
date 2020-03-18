from __future__ import absolute_import, division, print_function
from lib.classifiers.BertForEmbed import BertForSequenceClassification, Inferencer
from lib.handle_data.PreprocessForBert import InputFeatures
import torch
import pickle, logging, os, sys
from lib.handle_data.PreprocessForBert import *
from lib.utils import to_tensor_for_bert, get_torch_device
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

################
# HYPERPARAMETERS
################

LOAD_PATH = sys.argv[1] if len(sys.argv) > 1 else 'models/checkpoints/bert_for_embed/epoch11'
OUTPUT_MODE = 'classification'
NUM_LABELS = 2

with open(DATA_DIR + "all_features.pkl", "rb") as f:
    all_features = pickle.load(f)
all_ids, all_data, all_labels = to_tensor_for_bert(all_features, OUTPUT_MODE)

inferencer = Inferencer(REPORTS_DIR, OUTPUT_MODE, logger, device, use_cuda=USE_CUDA)
if __name__ == '__main__':
    model = BertForSequenceClassification.from_pretrained(LOAD_PATH, num_labels=NUM_LABELS,
                                                          output_hidden_states=True, output_attentions=True)
    print(f'Loaded model from {LOAD_PATH}')
    #inferencer.eval(model, all_data, all_labels, name=f'{LOAD_PATH}')
    embeddings = inferencer.predict(model, all_data, return_embeddings=True)
    basil_w_BERT = pd.DataFrame(index=all_ids, columns=['avbert'])
    basil_w_BERT['avbert'] = embeddings
    basil_w_BERT.to_csv('data/basil_w_avBERT.csv')