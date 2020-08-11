from __future__ import absolute_import, division, print_function
from transformers import RobertaTokenizer
from transformers.configuration_roberta import RobertaConfig
import pickle
from lib.handle_data.PreprocessForRoberta import *
import csv, time
from lib.handle_data.SplitData import split_input_for_bert


def preprocess_rows(rows):
    count = 0
    total = len(rows)
    features = []
    for i, row in enumerate(rows):
        feats = convert_example_to_feature(row)
        features.append(feats)
        count += 1

        if count % 250 == 0:
            status = f'Processed {count}/{total} rows'
            print(status)
    return features


def preprocess_voter(infp, ofp, set_type, voter=''):
    examples = dataloader.get_examples(infp, set_type, sep='\t')
    features = [features_dict[example.my_id] for example in examples if example.text_a]
    with open(ofp, "wb") as f:
        pickle.dump(features, f)

    print(f"Processed fold {fold_name} {set_type} {voter}- {len(features)} items and writing to {ofp}")

# choose sentence or bio labels
task = 'sent_clf'
DATA_DIR = f'data/{task}/ft_input'

# structure of project
#CONTEXT_TYPE = 'article'
FEAT_DIR = f'data/{task}/features_for_roberta_majvote/'

# The maximum total input sequence length after WordPiece tokenization.
# Sequences longer than this will be truncated, and sequences shorter than this will be padded.
MAX_SEQ_LENGTH = 124
OUTPUT_MODE = 'classification' # or 'classification', or 'regression'

dataloader = BinaryClassificationProcessor()
label_list = dataloader.get_labels(output_mode=OUTPUT_MODE)  # [0, 1] for binary classification

if OUTPUT_MODE == 'bio_classification':
    spacy_tokenizer = spacy.load("en_core_web_sm")
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=False, do_basic_tokenize=False)
    label_map = {label: i + 1 for i, label in enumerate(label_list)}

else:
    spacy_tokenizer = None
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    label_map = {label: i for i, label in enumerate(label_list)}

config = RobertaConfig.from_pretrained('roberta-base')
config.num_labels = len(label_map)

all_infp = os.path.join(DATA_DIR, f"all.tsv")
all_ofp = os.path.join(FEAT_DIR, f"all_features.pkl")
FORCE = False
if not os.path.exists(all_ofp) or FORCE:
    examples = dataloader.get_examples(all_infp, 'train', sep='\t')
    examples = [(example, label_map, MAX_SEQ_LENGTH, tokenizer, spacy_tokenizer, OUTPUT_MODE) for example in examples if example.text_a]

    features = preprocess_rows(examples)

    with open(all_ofp, "wb") as f:
        pickle.dump(features, f)

    print(f"Processed all - {len(features)} items")
    time.sleep(20)

with open(all_ofp, "rb") as f:
   features = pickle.load(f)
   features_dict = {feat.my_id: feat for feat in features}

   print(f"Loaded all - {len(features)} items")


# get split
N_VOTERS = 1
folds = split_input_for_bert(DATA_DIR, n_voters=N_VOTERS, recreate=True)

# start
for fold in folds:
    fold_name = fold['name']

    test_infp = os.path.join(DATA_DIR, f"{fold_name}_test.tsv")
    test_ofp = os.path.join(FEAT_DIR, f"{fold_name}_test_features.pkl")
    preprocess_voter(test_infp, test_ofp, 'test', voter='')

    for v in range(N_VOTERS):
        for set_type in ['train', 'dev']:
            infp = os.path.join(DATA_DIR, f"{fold_name}_{set_type}.tsv")
            ofp = os.path.join(FEAT_DIR, f"{fold_name}_{v}_{set_type}_features.pkl")

            examples = dataloader.get_examples(infp, set_type, sep='\t')

            #examples = [example for example in examples if example.label != '[]']

            features = [features_dict[example.my_id] for example in examples if example.text_a]
            print(f"Processed fold {fold_name} {v} {set_type} - {len(features)} items and writing to {ofp}")

            with open(ofp, "wb") as f:
                pickle.dump(features, f)

tokenizer.save_vocabulary(FEAT_DIR)

'''
# start
for fold in folds:
    fold_name = fold['name']

    test_infp = os.path.join(DATA_DIR, f"{fold_name}_test.tsv")
    test_ofp = os.path.join(FEAT_DIR, f"{fold_name}_test_features.pkl")
    preprocess_voter(test_infp, test_ofp, 'test', voter='')

    for v in range(N_VOTERS):
        train_infp = os.path.join(DATA_DIR, f"{fold_name}_{v}_train.tsv")
        train_ofp = os.path.join(FEAT_DIR, f"{fold_name}_{v}_train_features.pkl")

        dev_infp = os.path.join(DATA_DIR, f"{fold_name}_{v}_dev.tsv")
        dev_ofp = os.path.join(FEAT_DIR, f"{fold_name}_{v}_dev_features.pkl")

        preprocess_voter(train_infp, train_ofp, 'train', voter=v)
        preprocess_voter(dev_infp, dev_ofp, 'dev', voter=v)

tokenizer.save_vocabulary(FEAT_DIR)
'''