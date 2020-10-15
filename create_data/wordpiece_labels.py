from transformers import BertTokenizer
import pandas as pd
import spacy


def tokenize(x):
    global nlp
    if not isinstance(x, float):
        return [token.text for token in nlp(x)]
    else:
        return []


def tokenize_for_bio(x):
    if not isinstance(x, float):
        return x.split(" ")
    else:
        return []


def re_index(sentence_str):
    return " ".join([f"{i+1}_{word}" for i, word in enumerate(sentence_str.split())])


def expand_to_wordpieces(original_sentence, original_labels, tokenizer):
    """
    Maps a BIO Label Sequence to the BERT WordPieces length preserving the BIO Format
    :param original_sentence: String of complete-word tokens separated by spaces
    :param original_labels: List of labels 1-1 mapped to tokens
    :param tokenizer: BertTokenizer with do_basic_tokenize=False to respect the original_sentence tokenization.
    :return:
    """

    word_pieces = tokenizer.tokenize(original_sentence)

    print(original_sentence)
    print(original_labels)
    print(word_pieces)

    tmp_labels, lbl_ix = [], 0
    for tok in word_pieces:
        if "##" in tok:
            tmp_labels.append("X")
        else:
            tmp_labels.append(original_labels[lbl_ix])
            lbl_ix += 1
    # print("TMP ", tmp_labels)
    expanded_labels = []
    for i, lbl in enumerate(tmp_labels):
        if lbl == "X":
            # prev = tmp_labels[i-1]
            prev = expanded_labels[-1]
            if prev.startswith("B-"):
                expanded_labels.append("I-"+prev[2:])
            else:
                expanded_labels.append(prev)
        else:
            expanded_labels.append(lbl)
    assert len(word_pieces) == len(expanded_labels)

    print(expanded_labels)

    return word_pieces, expanded_labels


def recover_from_wordpieces(expanded_tokens, expanded_labels):
    r_toks, r_lbls = [], []
    for index, token in enumerate(expanded_tokens):
        if token.startswith("##"):
            if r_toks:
                r_toks[-1] = f"{r_toks[-1]}{token[2:]}"
        else:
            r_toks.append(token)
            r_lbls.append(expanded_labels[index])
    return r_toks, r_lbls


sentences = [{"seq_words": ["No", ",", "it", "was", "n't", "Black", "Monday", "."], "BIO": ["O", "O", "O", "O", "O", "O", "O", "O"]},
             {"seq_words": ["But", "as", "panic", "spread", ",", "speculators", "began", "to", "sell", "blue", "-", "chip", "stocks", "such", "as", "Philip", "Morris", "and", "International", "Business", "Machines", "to", "offset", "their", "losses", "."], "BIO": ["B-AM-DIS", "B-AM-TMP", "O", "O", "O", "B-A0", "B-V", "B-A1", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"]}
            ]


basil = pd.read_csv('data/basil_w_bio.csv', index_col=0)

'''
nlp = spacy.load("en_core_web_sm")
tokens = basil.sentence.apply(tokenize).values
bio = basil.bio.apply(tokenize_for_bio).values
print(tokens)
print(bio)

sentences = [{"seq_words": s, "BIO": b} for s, b in zip(tokens, bio)]


tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased", do_basic_tokenize=False)

all_bert_labels = []
for sent in sentences:
    # BERT Tokenize the Source!
    sent_str = " ".join(sent["seq_words"])
    bert_pieces, bert_labels = expand_to_wordpieces(sent_str, sent["BIO"], tokenizer)
    all_bert_labels.append(" ".join(bert_labels))

    original_toks, original_lbls = recover_from_wordpieces(bert_pieces, bert_labels)

    print("\n -- Recovered Tokens and Labels --")
    print(original_toks)
    print(original_lbls)
    print("--------")
'''

basil['id'] = basil['uniq_idx.1'].str.lower()
basil = basil.rename(columns={'label': 'inf_bias'})
basil = basil.rename(columns={'bio': 'label'})
basil['alpha'] = ['a']*len(basil)

basil = basil[['id', 'label', 'alpha', 'sentence']]
print(basil.head)
basil.to_csv('data/tok_clf/basil.csv', header=False)
