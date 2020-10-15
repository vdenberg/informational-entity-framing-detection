import spacy
from lib.handle_data.LoadData import LoadBasil
import pandas as pd


def tokenize(x):
    global nlp
    return [token.text for token in nlp(x)]


if __name__ == '__main__':
    basil = pd.read_csv('data/basil.csv', index_col=0).fillna('')

    # tokenize
    nlp = spacy.load("en_core_web_sm")
    basil['tokens'] = basil.sentence.apply(tokenize)
    basil.to_csv('data/basil_w_tokens.csv')

