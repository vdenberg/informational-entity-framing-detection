import pandas as pd

if __name__ == '__main__':
    BASIL = 'data/basil.csv'
    BASIL_ERR = 'data/error_analysis/basil_devtest_w_errors.csv'

    basil = pd.read_csv(BASIL, index_col=0).fillna('')
    basil_err = pd.read_csv(BASIL_ERR, index_col=0).fillna('')

    basil['error'] = 'not in dev'

    basil.loc[basil_err.index, 'error'] = basil_err.error
    basil = basil[basil.error != 'not in dev']

    print(basil.columns)
    print(basil[['sentence', 'error']])

    basil['quotes'] = basil.sentence.apply(lambda x: '"' in x)
    N = len(basil)
    N_q = len(basil[basil.quotes == True])
    for n, gr in basil.groupby('error'):
        n_q = len(gr[gr.quotes == True])
        prop_q = round(n_q / len(gr) * 100, 2)
        print(n, n_q, len(gr), prop_q)

    overall_prop = round(N_q / len(basil) * 100, 2)
    print('All', N_q, len(basil), overall_prop)

