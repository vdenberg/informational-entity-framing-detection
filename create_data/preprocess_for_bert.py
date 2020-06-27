from __future__ import absolute_import, division, print_function
from transformers import BertTokenizer
import pickle
from lib.handle_data.PreprocessForBert import *
import csv, time
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
task = 'tok_clf'
DATA_DIR = f'data/{task}/ft_input'

# load and split data
folds = split_input_for_bert(DATA_DIR, task)
NR_FOLDS = len(folds)

# structure of project
FEAT_DIR = f'data/{task}/features_for_bert/'
DEBUG = False
SUBSET = 1.0 if not DEBUG else 0.1

MAX_SEQ_LENGTH = 124
OUTPUT_MODE = 'bio_classification' # 'classification' ,or 'bio-classification'

dataloader = BinaryClassificationProcessor()
label_list = dataloader.get_labels(output_mode=OUTPUT_MODE)  # [0, 1] for binary classification

if OUTPUT_MODE == 'bio_classification':
    spacy_tokenizer = spacy.load("en_core_web_sm")
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False, do_basic_tokenize=False)
    label_map = {label: i + 1 for i, label in enumerate(label_list)}
else:
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
    label_map = {label: i for i, label in enumerate(label_list)}

all_infp = os.path.join(DATA_DIR, f"all.tsv")
ofp = os.path.join(FEAT_DIR, f"all_features.pkl")

FORCE = False
if not os.path.exists(ofp) or FORCE:
    examples = dataloader.get_examples(all_infp, 'train', sep='\t')

    examples = [(example, label_map, MAX_SEQ_LENGTH, tokenizer, OUTPUT_MODE) for example in examples]
    features = preprocess(examples)
    features_dict = {feat.my_id: feat for feat in features}
    print(f"Processed fold all - {len(features)} items")

    with open(ofp, "wb") as f:
        pickle.dump(features, f)
    time.sleep(15)
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

        examples = dataloader.get_examples(infp, set_type, sep='\t')
        features = [features_dict[example.my_id] for example in examples]

        l = []
        for f in features:
            l.extend(f.label_id)
            if 1 in set(f.label_id):
                print(f.label_id)
        print(set(l))

        print(f"Processed fold {fold_name} {set_type} - {len(features)} items - to {ofp}")

        with open(ofp, "wb") as f:
            pickle.dump(features, f)

tokenizer.save_vocabulary(FEAT_DIR)
