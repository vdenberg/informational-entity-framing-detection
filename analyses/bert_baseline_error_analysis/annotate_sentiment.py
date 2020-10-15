import pandas as pd


def sent_in_sentence(s, sent_dict):
    sent = False
    s = ' ' + s
    for el in sent_dict:
        if ' ' + el in s:
            sent = True
    return sent


if __name__ == '__main__':
    BASIL = 'data/basil.csv'
    BASIL_ERR = 'data/error_analysis/basil_devtest_w_errors.csv'
    basil = pd.read_csv(BASIL, index_col=0).fillna('')
    basil_err = pd.read_csv(BASIL_ERR, index_col=0).fillna('')
    basil['error'] = 'not in dev'
    basil.loc[basil_err.index, 'error'] = basil_err.error
    basil = basil[basil.error != 'not in dev']

    sent_fp = 'analyses/bert_baseline_error_analysis/lexica/subjectivity_clues_hltemnlp05/subjclueslen1-HLTEMNLP05.tff'
    sent_dict = {}
    with open(sent_fp, 'r') as f:
        content = f.readlines()
        for line in content:
            subj = line.split(' ')[0].split('=')[-1]
            if subj == 'strongsubj':
                word, polarity = line.split(' ')[2], line.split(' ')[-1]
                word, polarity = word.split('=')[-1], polarity.split('=')[-1]
                sent_dict[word] = polarity

    print(basil.columns)

    basil['sent'] = basil.sentence.apply(lambda x: sent_in_sentence(x, sent_dict))
    N = len(basil)
    N_q = len(basil[basil.sent == True])
    for n, gr in basil.groupby('error'):
        n_q = len(gr[gr.sent == True])
        prop_q = round(n_q / len(gr) * 100, 2)
        print(n, n_q, len(gr), prop_q)

    overall_prop = round(N_q / len(basil) * 100, 2)
    print('All', N_q, len(basil), overall_prop)

