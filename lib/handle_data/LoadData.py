from pprint import pprint

import os
import json
import pandas as pd
import numpy as np
import json
from transformers import BertTokenizer
import re
import spacy


def load_basil_spans(start_ends):
    start_ends = start_ends[2:-2]
    if list(start_ends):
        start_ends = re.sub('\), \(', ';', start_ends)
        start_ends = start_ends.split(';')
        start_ends = [tuple(map(int, s_e.split(', '))) for s_e in start_ends]
    return start_ends


class LoadBasil:
    """
    This is where basil.csv is created
    """
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


