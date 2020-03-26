from __future__ import absolute_import, division, print_function
from transformers import BertTokenizer
import pickle
from lib.handle_data.PreprocessForBert import *
import csv
from lib.handle_data.SplitData import Split

# structure of project
CONTEXT_TYPE = 'article'
DATA_DIR = f'data/cam_input/{CONTEXT_TYPE}'
string_data_fp = os.path.join(DATA_DIR, 'merged_basil.tsv')
TASK_NAME = 'bert_for_embed'
FEAT_DIR = 'data/features_for_bert/folds/'
SPLIT_TYPE = 'both'
DEBUG = True
SUBSET = 1.0 if not DEBUG else 0.1

# The maximum total input sequence length after WordPiece tokenization.
# Sequences longer than this will be truncated, and sequences shorter than this will be padded.
MAX_SEQ_LENGTH = 122
OUTPUT_MODE = 'classification'
tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
processor = BinaryClassificationProcessor()

# load and split data
sentences = pd.read_csv('data/basil.csv', index_col=0).fillna('')['sentence'].values
string_data = pd.read_csv(string_data_fp, sep='\t',
                          names=['id', 'context_document', 'label', 'position'],
                          dtype={'id': str, 'tokens': str, 'bias': int, 'position': int})
string_data['sentence'] = sentences
string_data['alpha'] = ['a']*len(string_data)
spl = Split(string_data, which=SPLIT_TYPE, subset=SUBSET)
folds = spl.apply_split(features=['id', 'bias', 'alpha', 'sentence'])

if DEBUG:
    folds = [folds[0], folds[1]]
NR_FOLDS = len(folds)

# start
for fold in folds:
    for set_type in ['train', 'dev']:
        infp = os.path.join(DATA_DIR, set_type + ".tsv")
        fold[set_type][['id', 'bias', 'alpha', 'sentence']].to_csv(infp)

        print('Processing', fold['name'], set_type)

        examples = processor.get_examples(os.path.join(DATA_DIR, set_type + ".csv"), set_type, sep=',')

        label_list = processor.get_labels() # [0, 1] for binary classification
        num_labels = len(label_list)
        label_map = {label: i for i, label in enumerate(label_list)}

        examples = [(example, label_map, MAX_SEQ_LENGTH, tokenizer, OUTPUT_MODE) for example in examples]
        features = [convert_example_to_feature(row) for row in examples]

        ofp = os.path.join(FEAT_DIR, f"{fold['name']}_{set_type}_features.pkl")
        with open(ofp, "wb") as f:
            pickle.dump(features, f)

tokenizer.save_vocabulary(FEAT_DIR)
