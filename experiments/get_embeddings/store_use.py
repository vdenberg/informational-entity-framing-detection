import tensorflow_hub as hub
import pandas as pd
import numpy as np

if __name__ == '__main__':
    basil = pd.read_csv('data/basil.csv', index_col=0).fillna('')

    # add embeds
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    embs = []
    for sent in basil.sentence.values:
        em = embed([sent])
        em = list(np.array(em[0]))
        embs.append(em)
    basil['USE'] = embs
    basil.to_csv('data/basil_w_USE.csv')
