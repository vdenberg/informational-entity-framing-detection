import json
import os


def find_102(vocab):
    for k, v in vocab.items():
        if v == 102:
            return k

special_tokens = {"<s>": "[ClS]", "</s>": "[SEP]", "<unk>": "[UNK]", "<pad>": "[PAD]", "<mask>": "[MASK]"}
models_dir = '../pretrained_models'

for model_dir in os.listdir(models_dir):
    vocab_json_fp = os.path.join(models_dir, model_dir, 'vocab.json')
    vocab_text_fp = os.path.join(models_dir, model_dir, 'vocab.txt')

    # load vocab
    vocab = json.load(open(vocab_json_fp, encoding='utf-8'))

    # get rid of weird chars
    vocab = {tok.strip('Ġ'): idx for tok, idx in vocab.items()}

    # replace special tokens
    vocab = {special_tokens[tok] if tok in special_tokens else tok: idx for tok, idx in vocab.items()}

    # replace token with index 102
    currently_at_102 = find_102(vocab)
    current_loc_of_sep = vocab["[SEP]"]

    new_loc_of_current_at_102 = current_loc_of_sep
    new_loc_of_sep = 102

    vocab["[SEP]"] = new_loc_of_sep
    vocab[currently_at_102] = new_loc_of_current_at_102

    # make text lines out of dict
    reverse_vocab = {idx: tok for tok, idx in vocab.items()}

    tokens_txt = []
    unused_counter = 0
    for idx in range(len(vocab)):
        if idx in reverse_vocab:
            tok = reverse_vocab[idx]
        else:
            tok = f"[unused{unused_counter}]"
            unused_counter += 1
        tokens_txt.append(tok)


    # write output
    with open(vocab_text_fp, "w") as f:
        for line in tokens_txt:
            f.write(line + "\n")

    print(f"Written to {vocab_text_fp}, {unused_counter} / {len(tokens_txt)} unused")

