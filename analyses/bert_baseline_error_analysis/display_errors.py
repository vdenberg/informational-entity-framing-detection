import pandas as pd

if __name__ == '__main__':
    BASIL = 'data/basil.csv'
    BASIL_ERR = 'data/analyses/error_analysis/basil_devtest_w_errors.csv'
    basil = pd.read_csv(BASIL, index_col=0).fillna('')
    basil_err = pd.read_csv(BASIL_ERR, index_col=0).fillna('')

    basil['error'] = 'not in dev'
    basil.loc[basil_err.index, 'error'] = basil_err.error

    errors = basil_err.index
    for err in basil_err.index:
        _err, idx = basil.loc[err], basil.loc[err].sent_idx

        with open('analyses/bert_baseline_error_analysis/errors_in_context/{}_{}.txt'.format(err, _err.error), 'w') as f:
            story_context = basil[(basil.story == _err.story) & (basil.source != _err.source)]
            for n, source in story_context.groupby('source'):
                f.write('--- start ' + n.upper() + '----\n')
                f.write('\n'.join(source.sentence.to_list()) + '\n')
                f.write('--- end ' + n.upper() + '----\n\n')

            article_context = basil[(basil.story == _err.story) & (basil.source == _err.source)]
            _article_context = article_context.sentence.to_list()
            f.write('--- start ' + _err.source.upper() + ' (w error) ----\n')
            f.write('\n'.join(_article_context[:idx]) + '\n')
            f.write('(' + _err.error.upper() + ') ' + _err.sentence + '\n')
            f.write('\n'.join(_article_context[idx+1:])  + '\n')
            f.write('--- end ' + _err.source.upper() + ' (w error) ----')

        with open('analyses/bert_baseline_error_analysis/errors_in_context/{}_{}_notes.txt'.format(err, _err.error), 'w') as f:
            f.write('(' + _err.error.upper() + ') ' + _err.sentence + '\n')