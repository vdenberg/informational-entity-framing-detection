from sentence_transformers import SentenceTransformer
import pandas as pd

basil = pd.read_csv('data/basil_w_features.csv', index_col=0).fillna('')
sentences = basil.sentence.values

model = SentenceTransformer('bert-base-nli-mean-tokens')
sentence_embeddings = model.encode(sentences)
basil['sbert_pre'] = sentence_embeddings
basil.to_csv('data/basil_w_sbert_pre')

exit(0)
# Use BERT for mapping tokens to embeddings
word_embedding_model = models.BERT('bert-base-uncased')

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])