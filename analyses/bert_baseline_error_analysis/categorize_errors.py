import os
from collections import Counter

errdir = 'errors_in_context'

data = {'fp': [], 'fn': []}
cnts = {'fp': 0, 'fn': 0}
for fn in os.listdir(errdir):
    if fn.endswith('notes.txt'):
        errtyp = fn.split('_')[1]
        with open(os.path.join(errdir, fn)) as f:
            scndline = f.readlines()[1]
        labels = scndline.strip('\n').split('-')
        data[errtyp].extend(labels)
        cnts[errtyp] += 1

print(cnts)
print(19+42)
import pandas as pd
df = pd.DataFrame(index=['quote', 'quotemarks','lexical','adjv','nv','implied', 'opinion', 'information', 'noiseingold'], columns=['fp', 'fn'])
for k, v in data.items():
    for t, cnt in Counter(v).items():
        df.loc[t, k] = cnt
df = df.fillna(0)
print(df)
print(df.apply(lambda x: round(100*x/sum(x),2)))

