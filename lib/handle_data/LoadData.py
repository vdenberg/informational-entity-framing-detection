from pprint import pprint

import os
import json
import pandas as pd
import numpy as np
import json
from transformers import BertTokenizer
import spacy

nlp = spacy.load("en_core_web_sm")
def tokenize(x):
    global nlp
    return [token.text for token in nlp(x)]

class LoadBasil:
    def __init__(self):
        self.raw_dir = 'create_data/emnlp19-BASIL/data/'

    def load_basil_all(self):
        # load exactly as published by authors

        collection = {}
        for file in os.listdir(self.raw_dir):
            idx = int(file.split('_')[0])
            source = file.split('_')[1][:3]
            story = collection.setdefault(idx, {'entities': set(), 'hpo': None, 'fox': None, 'nyt': None})
            with open(self.raw_dir + file) as f:
                content = json.load(f)
            story[source] = content
            story['entities'].update(content['article-level-annotations']['author-sentiment'])

        return collection

    def load_spans(self):
        all_spans = {}

        for file in os.listdir(self.raw_dir):
            story = file.split('_')[0]
            source = file.split('_')[1][:3]

            with open(self.raw_dir + file) as f:
                file_content = json.load(f)
                sentences = file_content['body']

                for sent in sentences:
                    sentence = sent['sentence']
                    sent_idx = str(sent['sentence-index'])
                    id = story + source + sent_idx

                    lexical_ann = [ann for ann in sent['annotations'] if ann['bias'] == 'Lexical']
                    informational_ann = [ann for ann in sent['annotations'] if ann['bias'] == 'Informational']

                    all_spans.setdefault(id, {'inf_spans': [], 'lex_spans': []})

                    if informational_ann:

                        for ann in informational_ann:
                            inf_spans.append((ann['start'],ann['end']))

                    if lexical_ann:
                        all_spans[id]['lex_spans'] = []

                        for ann in lexical_ann:
                            lex_spans.append((ann['start'], ann['end']))



    def load_basil_raw(self):
        # load raw but stripped off fields that are not as relevant
        pre_df = []
        for file in os.listdir(self.raw_dir):
            story = file.split('_')[0]
            source = file.split('_')[1][:3]
            with open(self.raw_dir + file) as f:
                file_content = json.load(f)
                #pprint(file_content)
                sentences = file_content['body']
                for sent in sentences:
                    sentence = sent['sentence']
                    sent_idx = str(sent['sentence-index'])
                    lexical_ann = [ann for ann in sent['annotations'] if ann['bias'] == 'Lexical']
                    informational_ann = [ann for ann in sent['annotations'] if ann['bias'] == 'Informational']
                    lex_bias_present = 1 if lexical_ann else 0
                    inf_bias_present = 1 if informational_ann else 0
                    inf_start_ends = []
                    lex_start_ends = []
                    if inf_bias_present:
                        for ann in informational_ann:
                            inf_start_ends.append((ann['start'],ann['end']))
                    if lex_bias_present:
                        for ann in lexical_ann:
                            lex_start_ends.append((ann['start'], ann['end']))
                    pre_df.append([story, source, sent_idx, lex_bias_present, inf_bias_present, sentence, lex_start_ends, inf_start_ends])
        df = pd.DataFrame(pre_df, columns=['story', 'source', 'sent_idx', 'lex_bias', 'bias', 'sentence', 'lex_start_ends', 'inf_start_ends'])
        df['uniq_idx'] = df['story'] + df['source'] + df['sent_idx']
        df = df.set_index(df['uniq_idx'])
        df.to_csv('data/basil.csv')
        return df

    def to_token(self):
        df = pd.read_csv('data/basil_w_tokens.csv', index_col=0)
        df = df[df.bias == 1]
        sentences = df.sentence.values[:3]
        tokens = df.tokens.values[:3]
        spans = df.inf_start_ends[:3]

        spanbased_labels = []
        for s, t, sps in zip(sentences, tokens, spans):
            spb_lbls = [0] * len(t)

            sps = list(sps)
            print(sps)
            for sp in sps:
                sp = tuple(sp)
                print(sp[0])
                before = s[:sp[0]]
                during = s[sp[0]:sp[1]]
                after = s[sp[1]:]

                before_tokens = tokenize(before)
                during_tokens = tokenize(during)
                after_tokens = tokenize(after)

                before_labels = len(before_tokens) * [0]
                during_labels = len(during_tokens) * [1]
                after_labels = len(after_tokens) * [0]

                label_sequence = before_labels + during_labels + after_labels

                for i, l in enumerate(label_sequence):
                    if l == 1:
                        spb_lbls[i] = 1
            spanbased_labels.append(spb_lbls)
        targetoutput = [['The', 'agents', 'were', 'confused', '.'],[0,1,0,0,0]]
        print(spanbased_labels)
        print(targetoutput)

def load_basil_features():
    # features are added in create_data folder
    return pd.read_csv('data/basil_w_features.csv', index_col=0).fillna('')


LoadBasil().to_token()
'''
def load_basil_USE():
    basil_use = pd.read_csv('data/basil_USE.csv')
    X = basil_use.USE.values
    X = [[fl.split('=')[-1].strip('>)') for fl in x.split('>, <')] for x in X]
    basil_use['USE'] = X
    return basil_use
'''

