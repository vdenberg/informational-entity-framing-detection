from __future__ import absolute_import, division, print_function
from transformers import RobertaTokenizer
from transformers.configuration_roberta import RobertaConfig
import pickle
from lib.handle_data.PreprocessForRoberta import *
import csv
from lib.handle_data.SplitData import split_input_for_bert


def preprocess(rows):
    count = 0
    total = len(rows)
    features = []
    for row in rows:
        feats = convert_example_to_feature(row)
        features.append(feats)
        count += 1

        if count % 250 == 0:
            status = f'Processed {count}/{total} rows'
            print(status)
    return features


# choose sentence or bio labels
task = 'sent_clf'
DATA_DIR = f'data/{task}/ft_input'

# load and split data
folds = split_input_for_bert(DATA_DIR, task)

# structure of project
CONTEXT_TYPE = 'article'
FEAT_DIR = f'data/{task}/features_for_roberta/'
DEBUG = False
SUBSET = 1.0 if not DEBUG else 0.1

# The maximum total input sequence length after WordPiece tokenization.
# Sequences longer than this will be truncated, and sequences shorter than this will be padded.
MAX_SEQ_LENGTH = 124
OUTPUT_MODE = 'classification' # or 'classification', or 'regression'
NR_FOLDS = len(folds)

if OUTPUT_MODE == 'bio_classification':
    spacy_tokenizer = spacy.load("en_core_web_sm")
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=False, do_basic_tokenize=False)
else:
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=False)

dataloader = BinaryClassificationProcessor()

label_list = dataloader.get_labels(output_mode=OUTPUT_MODE)  # [0, 1] for binary classification
label_map = {label: i for i, label in enumerate(label_list)}
config = RobertaConfig.from_pretrained('roberta-base')
config.num_labels = len(label_map)

all_infp = os.path.join(DATA_DIR, f"all.tsv")
ofp = os.path.join(FEAT_DIR, f"all_features.pkl")

FORCE = True
if not os.path.exists(ofp) or FORCE:
    examples = dataloader.get_examples(all_infp, 'train', sep='\t')

    examples = [(example, label_map, MAX_SEQ_LENGTH, tokenizer, OUTPUT_MODE) for example in examples]
    features = preprocess(examples)
    features_dict = {feat.my_id: feat for feat in features}
    print(f"Processed fold all - {len(features)} items")

    with open(ofp, "wb") as f:
        pickle.dump(features, f)
else:
    with open(ofp, "rb") as f:
       features = pickle.load(f)
       features_dict = {feat.my_id: feat for feat in features}
       print(f"Processed fold all - {len(features)} items")

# start
for fold in folds:
    fold_name = fold['name']
    for set_type in ['train', 'dev', 'test']:
        infp = os.path.join(DATA_DIR, f"{fold_name}_{set_type}.tsv")
        ofp = os.path.join(FEAT_DIR, f"{fold_name}_{set_type}_features.pkl")

        #if not os.path.exists(ofp):
        examples = dataloader.get_examples(infp, set_type, sep='\t')

        label_list = dataloader.get_labels(output_mode=OUTPUT_MODE)  # [0, 1] for binary classification
        label_map = {label: i for i, label in enumerate(label_list)}

        #examples = [(example, label_map, MAX_SEQ_LENGTH, tokenizer, OUTPUT_MODE) for example in examples]
        #features = [convert_example_to_feature(row) for row in examples]
        features = [features_dict[example.my_id] for example in examples]
        print(f"Processed fold {fold_name} {set_type} - {len(features)} items and writing to {ofp}")

        with open(ofp, "wb") as f:
            pickle.dump(features, f)

tokenizer.save_vocabulary(FEAT_DIR)
