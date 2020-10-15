import numpy as np
import random as rm
from collections import Counter
import pandas as pd

# The statespace
states = ["Neut", "Bias"]
hiddenstates = ["Obj","Lex"]

# Possible sequences of events
transitionName = [["NN","NB"], ["BN", "BB"]]
emissionName = [["LexN","LexB"],["ObjN","ObjB"]]

# Preproc
BASIL = 'data/basil.csv'
basil = pd.read_csv(BASIL, index_col=0).fillna('')
print(basil.columns)
basil = basil[(basil.lex_bias == 1) & (basil.label == 1)]
print(len(basil))
print(basil)
exit(0)

basil['lex_bias'] = basil.lex_bias.replace(0, "Obj")
basil['lex_bias'] = basil.lex_bias.replace(1, "Lex")
basil['bias'] = basil.bias.replace(0, "N")
basil['bias'] = basil.bias.replace(1, "B")

# data
sequences = basil[['bias', 'lex_bias']].values

# transitions
transitionNames = [["NN","NB"], ["BN", "BB"]]

observations = [el[0] for el in sequences]
denoms = Counter(observations)

start_mat = [1, 0]

transitions = []
for i_plus in range(1, len(observations)):
    t = str(observations[i_plus-1]) + str(observations[i_plus])
    transitions.append(t)
transitions = {name: round(cnt/denoms[name[0]],2) for name, cnt in Counter(transitions).items()}
print(transitions)
transitionMatrix = [[.86, .14], [.78, .22]]
print('Conclusions: Bias is somewhat more likely to happen after other bias')

# emissions
emissionName = [["LexN","LexB"],["ObjN","ObjB"]]

hidden = [str(el[1]) for el in sequences]
denoms = Counter(hidden)

emissions = [str(el[1]) + str(el[0]) for el in sequences]
emissions = {name: round(cnt/denoms[name[:3]],2) for name, cnt in Counter(emissions).items()}
print(emissions)
emissionMatrix = [[.9, .1], [.84, .16]]
print('Conclusions: Bias is somewhat more likely to happen after objective language')






