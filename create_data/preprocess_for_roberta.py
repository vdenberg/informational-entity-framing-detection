from __future__ import absolute_import, division, print_function
from transformers import RobertaTokenizer
from transformers.configuration_roberta import RobertaConfig
import pickle
from lib.handle_data.PreprocessForRoberta import *
import csv
from lib.handle_data.SplitData import split_input_for_bert
import torch

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


def enforce_max_sent_per_example(sentences, max_sent_per_example, labels=None):
    """
    Splits examples with len(sentences) > self.max_sent_per_example into multiple smaller examples
    with len(sentences) <= self.max_sent_per_example.
    Recursively split the list of sentences into two halves until each half
    has len(sentences) < <= self.max_sent_per_example. The goal is to produce splits that are of almost
    equal size to avoid the scenario where all splits are of size
    self.max_sent_per_example then the last split is 1 or 2 sentences
    This will result into losing context around the edges of each examples.
    """
    if labels is not None:
        assert len(sentences) == len(labels)

    if len(sentences) > max_sent_per_example > 0:
        i = len(sentences) // 2
        l1 = enforce_max_sent_per_example(
                sentences[:i], max_sent_per_example, None if labels is None else labels[i:])
        l2 = enforce_max_sent_per_example(
                sentences[i:], max_sent_per_example, None if labels is None else labels[i:])
        return l1 + l2
    else:
        return [sentences]


def get_art_id(feat_id):
    if not feat_id[1].isdigit():
        feat_id = '0' + feat_id
    return feat_id[:5]


def convert_bunched_example_to_feat(features_of_example, cls_token, pad_token, max_ex_len):
    example_input_ids = []
    example_labels = []

    for feats in features_of_example:
        input_ids = [el for el in feats.input_ids if el not in [cls_token, pad_token]]
        example_input_ids.extend(input_ids)
        example_labels.extend([feats.label_id])

    pad_len = max_ex_len - len(input_ids)
    example_input_ids += [1] * pad_len
    example_mask = [1] * len(input_ids) + [0] * pad_len
    return InputFeatures(my_id='',
                         input_ids=example_input_ids,
                         input_mask=example_mask,
                         segment_ids=[],
                         label_id=example_labels)


def bunch_features(features, cls_token=0, pad_token=1, max_ex_sents=10, max_doc_len=76, max_sent_len=120):
    by_id = {feat.my_id: feat for feat in features}

    by_article = {}
    for feat in features:
        article_id = get_art_id(feat.my_id)
        #print(article_id)
        by_article.setdefault(article_id, [])
        by_article[article_id].append(feat.my_id)

    # print(len(by_article))

    example_ids = []
    nr_of_examples_per_article = 0  # todo: compute this
    for article_id, sentences in by_article.items():
        example_sentences = enforce_max_sent_per_example(sentences, max_ex_sents)
        example = []
        for sent in example_sentences:
            example.extend(sent)
        example_ids.append(example)

    max_ex_len = max([sum([len(by_id[feat_id].input_ids) for feat_id in ex]) for ex in example_ids])

    bunched_features = []
    for example in example_ids:
        features_of_example = [by_id[feat_id] for feat_id in sorted(example)]
        feats = convert_bunched_example_to_feat(features_of_example, cls_token, pad_token, max_ex_len)
        bunched_features.append(feats)

    return bunched_features


# choose sentence or bio labels
task = 'sent_clf'
DATA_DIR = f'data/{task}/ft_input'

# load and split data
folds = split_input_for_bert(DATA_DIR, task)
MAX_DOC_LEN = 76
MAX_SENT_LEN = 122
MAX_EX_LEN = 5

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
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    tokenizer.encode_plus('test')

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

    examples = [(example, label_map, MAX_SEQ_LENGTH, tokenizer, OUTPUT_MODE) for example in examples if example.text_a]
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

        #examples = [(example, label_map, MAX_SEQ_LENGTH, tokenizer, OUTPUT_MODE) for example in examples]
        #features = [convert_example_to_feature(row) for row in examples]
        features = [features_dict[example.my_id] for example in examples if example.text_a]

        features = bunch_features(features, cls_token=0, pad_token=1, max_ex_sents=MAX_EX_LEN, max_doc_len=MAX_DOC_LEN, max_sent_len=MAX_SENT_LEN)

        print(f"Processed fold {fold_name} {set_type} - {len(features)} items and writing to {ofp}")

        with open(ofp, "wb") as f:
            pickle.dump(features, f)

tokenizer.save_vocabulary(FEAT_DIR)
