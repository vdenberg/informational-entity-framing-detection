from pprint import pprint

import os
import json
import pandas as pd
import numpy as np
import json


class LoadBasil:
    def __init__(self):
        self.raw_dir = '../../create_data/emnlp19-BASIL/data/'

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
            triple_idx = file.split('_')[0]
            source_idx = file.split('_')[1][:3]
            with open(self.raw_dir + file) as f:
                file_content = json.load(f)
                pprint(file_content)
                sentences = file_content['body']
                for sent in sentences:
                    sentence = sent['sentence']
                    sent_idx = str(sent['sentence-index'])
                    lexical_ann = [ann for ann in sent['annotations'] if ann['bias'] == 'Lexical']
                    informational_ann = [ann for ann in sent['annotations'] if ann['bias'] == 'Informational']
                    lex_bias_present = 1 if lexical_ann else 0
                    inf_bias_present = 1 if informational_ann else 0
                    pre_df.append([triple_idx, source_idx, sent_idx, lex_bias_present, inf_bias_present, sentence])
        df = pd.DataFrame(pre_df, columns=['story', 'source', 'sent_idx', 'lex_bias', 'bias', 'sentence'])
        df['uniq_idx'] = df['story'] + df['source'] + df['sent_idx']
        df = df.set_index(df['uniq_idx'])
        return df

def load_basil_features():
    # features are added in create_data folder
    return pd.read_csv('data/basil_w_features.csv', index_col=0).fillna('')


'''
def load_basil_USE():
    basil_use = pd.read_csv('data/basil_USE.csv')
    X = basil_use.USE.values
    X = [[fl.split('=')[-1].strip('>)') for fl in x.split('>, <')] for x in X]
    basil_use['USE'] = X
    return basil_use
'''

