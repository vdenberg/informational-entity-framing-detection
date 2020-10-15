import pandas as pd

BASIL = 'data/basil.csv'
basil = pd.read_csv(BASIL, index_col=0).fillna('')

lex_dist = basil.lex_bias.value_counts()
inf_dist = basil.bias.value_counts()
both_lexinf = basil[(basil.lex_bias == 1) & (basil.bias == 1)].shape[0]
lexinf_corr = basil[['lex_bias', 'bias']].corr()

print(len(basil))

print(lex_dist)
print(inf_dist)
print(both_lexinf)
print(lexinf_corr)

sentences = basil.sentence.values
lens = [len(s.split(' ')) for s in sentences]
print("max sentence len:", max(lens))

article_lens = []
for n, gr in basil.groupby(['story', 'source']):
    article_lens.append(len(gr))
print("max doc len (article):", max(article_lens))

story_lens = []
for n, gr in basil.groupby(['story']):
    story_lens.append(len(gr))
print("max doc len (story):", max(story_lens))
