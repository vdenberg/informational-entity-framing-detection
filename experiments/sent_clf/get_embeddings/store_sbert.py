from sentence_transformers import SentenceTransformer
import pandas as pd

basil = pd.read_csv('data/basil.csv', index_col=0).fillna('')
sentences = basil.sentence.values

model = SentenceTransformer('bert-base-nli-mean-tokens')
sentence_embeddings = model.encode(sentences)
sentence_embeddings = [list(el) for el in sentence_embeddings]
basil['sbert_pre'] = sentence_embeddings
basil.to_csv('data/basil_w_sbert_pre.csv')