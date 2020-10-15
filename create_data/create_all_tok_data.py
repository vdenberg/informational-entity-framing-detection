import pandas as pd
import spacy
import re

nlp = spacy.load("en_core_web_sm")


def tokenize(x):
    global nlp
    return [token.text for token in nlp(x)]


def load_start_ends(start_ends):
    start_ends = start_ends[2:-2]
    start_ends = re.sub('\), \(', ';', start_ends)
    start_ends = start_ends.split(';')
    start_ends = [tuple(map(int, s_e.split(', '))) for s_e in start_ends]
    return start_ends


def load_tokens(tokens):
    tokens = tokens[2:-2]
    clean_tokens = tokens.split("', '")
    return clean_tokens


def get_label_sequence_for_span_OLD(sentence, span):
    before = sentence[:span[0]]
    during = sentence[span[0]:span[1]]
    after = sentence[span[1]:]

    before_tokens = tokenize(before)
    during_tokens = tokenize(during)
    after_tokens = tokenize(after)

    token_sequence = before + during + after

    before_labels = len(before_tokens) * ['O']
    during_labels = len(during_tokens) * [1]
    after_labels = len(after_tokens) * ['O']

    label_sequence = before_labels + during_labels + after_labels
    return label_sequence

"""
Converts basil to format with BIO tags

Output looks like: [{'seq_words': out_toks, "BIO": out_labels}, ...] for each sentence
"""

df = pd.read_csv('data/basil_w_tokens.csv', index_col=0) #todo: make sure we cant just use basil.csv

df = df[df.bias == 1]
sentences = df.sentence.values[:100]
tokens = df.tokens.values[:100]
spans = df.inf_start_ends[:100]

spanbased_labels = []
for s, t, sps in zip(sentences, tokens, spans):
    t = load_tokens(t)
    sps = load_start_ends(sps)
    spb_lbls = {i: (subt, 'O') for i, subt in enumerate(t)}

    print(sps)
    if len(sps) > 1:
        print(s)

    for sp in sps:
        label_sequence = get_label_sequence_for_span(s, t, sp)

        for i in range(len(t)):
            label = label_sequence[i]
            if label == 1:
                spb_lbls[i] = (t[i], label)

    prev_label = None
    for i in range(len(t)):
        subt, label = spb_lbls[i]
        if label == 1:
            if (prev_label is None) or (prev_label == "O"):
                bio_label = 'B-BIAS'
            else:
                bio_label = 'I-BIAS'
            spb_lbls[i] = (subt, bio_label)
        prev_label = label

    out_toks = []
    out_labels = []
    for i in range(len(t)):
        subt, label = spb_lbls[i]
        out_toks.append(subt)
        out_labels.append(label)
    out_dict = {'seq_words': out_toks, "BIO": out_labels}

    if len(sps) > 1:
        print(out_dict['BIO'])

    spanbased_labels.append(out_dict)
print(spanbased_labels)

