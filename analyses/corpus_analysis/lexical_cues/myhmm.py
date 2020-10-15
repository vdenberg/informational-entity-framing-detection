from hmmlearn.hmm import MultinomialHMM
import pandas as pd
from lib.handle_data.SplitData import Split

if __name__ == '__main__':
    BASIL = '../../data/basil_w_features.csv'
    basil = pd.read_csv(BASIL, index_col=0).fillna('')
    spl = Split(basil, which='fan', split_loc='../../data/fan_split')
    data = spl.apply_split(features=['sentence', 'lex_bias'], output_as='df')[0]

    data['train']['dummy'] = len(data['train'])*[0]
    data['dev']['dummy'] = len(data['dev'])*[0]
    data['test']['dummy'] = len(data['test'])*[0]

    trn = data['train'][['lex_bias', 'bias']].values
    dev = data['dev'][['lex_bias', 'bias']].values
    tst = data['test'][['bias']].values

    m = MultinomialHMM(n_components=2,
                       algorithm="viterbi", random_state=124, n_iter=10,
                       tol=1e-2, verbose=True, params="ste", init_params="ste")
