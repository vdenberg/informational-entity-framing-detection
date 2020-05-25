from __future__ import absolute_import, division, print_function
from transformers import BertTokenizer
import pickle
from lib.handle_data.PreprocessForBert import *
import csv
from lib.handle_data.SplitData import Split

def write_huggingface_input(basil):
    basil['alpha'] = ['a']*len(basil)
    basil['id'] = basil['uniq_idx.1'].str.lower()
    basil[['id', 'bias', 'alpha', 'sentence']].to_csv('../data/huggingface_input/basil.tsv', sep='\t', index=False, header=False)

write_huggingface_input(basil)

# structure of project
DATA_DIR = 'data/huggingface_input/'
TASK_NAME = 'bert_for_embed'
FEAT_DIR = 'data/features_for_bert/redo/'

# The maximum total input sequence length after WordPiece tokenization.
# Sequences longer than this will be truncated, and sequences shorter than this will be padded.
MAX_SEQ_LENGTH = 95
OUTPUT_MODE = 'classification'

tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

processor = BinaryClassificationProcessor()

for set_type in ['basil', 'train', 'dev', 'test']:
    print('Processing', set_type)

    if set_type == 'basil':
        infp = os.path.join(DATA_DIR, set_type + ".csv")
        examples = processor.get_examples(infp, set_type, sep=',')
    else:
        infp = os.path.join(DATA_DIR, set_type + ".tsv")
        examples = processor.get_examples(infp, set_type, sep='\t')

    label_list = processor.get_labels() # [0, 1] for binary classification
    num_labels = len(label_list)
    label_map = {label: i for i, label in enumerate(label_list)}

    examples = [(example, label_map, MAX_SEQ_LENGTH, tokenizer, OUTPUT_MODE) for example in examples]
    features = [convert_example_to_feature(row) for row in examples]

    with open(FEAT_DIR + set_type + "_features.pkl", "wb") as f:
        pickle.dump(features, f)

tokenizer.save_vocabulary(FEAT_DIR)
