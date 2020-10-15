import json
from lib.handle_data.SplitData import Split
import pandas as pd
import os
import json


def make_ktrain_dirs(fold_i, sn):
    d = '../data/ktrain_input/fold{}/{}'.format(fold_i, sn)
    d_pos = os.path.join(d, 'pos')
    d_neg = os.path.join(d, 'neg')
    for el in [d, d_pos, d_neg]:
        if not os.path.exists(el):
            os.mkdir(el)
    return d


def write_ktrain_line(sent, f):
    # sentence per line
    f.write(sent)
    f.write('\n')


def make_dummy(sentences):
    dummylist = [sentence.split(' ')[0] for sentence in sentences]
    return ' '.join(dummylist)


def write_allennlp_line(i, sentences, embeds, bias, f):
    line = {'embeddings':  '%'.join(embeds),
            'sentences': '%'.join(sentences),
            'idx': str(i),
            'label': str(bias[i])}
    json.dump(line, f)
    f.write('\n')


def write_allen_input(folds):
    for fold_i, fld in enumerate(folds):
        for sn, data in fld.items():
            # for allennlp competitor
            allenfp = '../data/allen_input/text/fold{}_'.format(fold_i) + '{}.json'.format(sn)

            with open(allenfp, 'w') as allenf: # each fold and set gets their own file, articles are ordered
                for n, gr in data.groupby('article'):
                    grouped_sentences = gr.sentence.to_list()
                    grouped_embeds = gr.USE.to_list()
                    grouped_bias = gr.bias.values

                    for sent_i, sent in enumerate(grouped_sentences):
                        write_allennlp_line(sent_i, grouped_sentences, grouped_embeds, grouped_bias, allenf)


def write_huggingface_input(basil):
    basil['alpha'] = ['a']*len(basil)
    basil['id'] = basil['uniq_idx.1'].str.lower()
    basil[['id', 'bias','alpha', 'sentence']].to_csv('../data/huggingface_input/basil.csv')


def make_cam_lines(group, art_sent_ids_list, cov1_sent_ids_list, cov2_sent_ids_list):
    sent_ids_string = ' '.join(art_sent_ids_list)
    cov1_sent_ids_string = ' '.join(cov1_sent_ids_list)
    cov2_sent_ids_string = ' '.join(cov2_sent_ids_list)

    lines = []
    for i in range(len(group)):
        uniq_id = art_sent_ids_list[i].lower()
        index = str(i)
        label = str(group.bias.values[i])
        line = '\t'.join([uniq_id, sent_ids_string, cov1_sent_ids_string, cov2_sent_ids_string, label, index])
        lines.append(line)
    return lines


def write_cam_input(basil):
    both_fp = 'data/sent_clf/cam_input/basil_art_and_cov.tsv'

    with open(both_fp, 'w') as f:

        for n, cov_gr in basil.groupby(['story']):
            for src, art_gr in cov_gr.groupby('source'):

                art_ids = cov_gr[cov_gr.source == src]['uniq_idx.1'].to_list()

                if src == 'hpo':
                    cov1_ids = cov_gr[cov_gr.source == 'nyt']['uniq_idx.1'].to_list()
                    cov2_ids = cov_gr[cov_gr.source == 'fox']['uniq_idx.1'].to_list()
                elif src == 'nyt':
                    cov1_ids = cov_gr[cov_gr.source == 'hpo']['uniq_idx.1'].to_list()
                    cov2_ids = cov_gr[cov_gr.source == 'fox']['uniq_idx.1'].to_list()
                elif src == 'fox':
                    cov1_ids = cov_gr[cov_gr.source == 'hpo']['uniq_idx.1'].to_list()
                    cov2_ids = cov_gr[cov_gr.source == 'nyt']['uniq_idx.1'].to_list()

                group_lines = make_cam_lines(art_gr, art_ids, cov1_ids, cov2_ids)

                for line in group_lines:
                    f.write(line)
                    f.write('\n')



def write_ssc_input(basil_df, ssc_article_fp):
    basil_df['label'] = basil_df.label.replace({0: 'neutral', 1: 'bias'})

    ssc_input = []
    max_doc_len = 0
    max_sent_len = 0
    for n, gr in basil_df.groupby('article'):

        article_id = gr.article.values[0]
        sentences = gr.sentence.values
        labels = gr.label.values
        print(labels)
        article_dict = {'article_id': str(article_id), 'labels': list(labels), 'sentences': list(sentences)}

        ssc_input.append(article_dict)

        if len(sentences) > max_doc_len:
            max_doc_len = len(sentences)
        for sent in sentences:
            if len(sent.split(' ')) > max_sent_len:
                max_sent_len = len(sent.split(' '))

    with open(ssc_article_fp, 'w') as f:
        for l in ssc_input:
            json.dump(l, f)
            f.write('\n')

    print(f"Wrote {len(ssc_input)} (max doc len: {max_doc_len}, max sent len: {max_sent_len}) lines to {ssc_article_fp}")

    return ssc_input


def write_tok_ft_input(basil):
    #basil['alpha'] = ['a']*len(basil)
    #basil['id'] = basil['uniq_idx.1'].str.lower()
    #basil[['id', 'bias','alpha', 'sentence']].to_csv('../data/huggingface_input/basil.csv')

    basil['id'] = basil['uniq_idx.1'].str.lower()
    basil = basil.rename(columns={'label': 'inf_bias'})
    basil = basil.rename(columns={'inf_start_ends': 'label'})
    basil['alpha'] = ['a'] * len(basil)

    basil = basil[['id', 'label', 'alpha', 'sentence']]
    basil.to_csv('data/tok_clf/basil.csv', header=False)


def write_roberta_ssc_input():
    outfp = 'data/sent_clf/roberta_ssc_input'


def write_tapt_input(basil):
    article_counter = 0
    for n, gr in basil.groupby('article'):

        if article_counter <= 250:
            file_path = 'data/tapt/basil_train.txt'
        else:
            file_path = 'data/tapt/basil_dev.txt'

        with open(file_path, 'a') as f:
            sentences = gr.sentence.values
            for s in sentences:
                f.write(s)
                f.write(' ')
            f.write('\n')
        article_counter += 1


if __name__ == '__main__':
    #basil = LoadBasil().load_basil_raw()
    #basil.to_csv('data/basil.csv')
    basil = pd.read_csv('data/basil.csv', index_col=0).fillna('')
    basil['article'] = basil.story.astype(str) + basil.source
    #my_spl = Split(basil, which='berg', split_loc='../data/berg_split')
    #folds = my_spl.apply_split(features=['article', 'sentence', 'USE'], output_as='df')
    #write_cam_input(basil)

    #spl = Split(basil, which='berg')
    #folds = spl.apply_split(features=['article', 'sentence'])
    #for fold in folds:
    #    for set_type in ['train', 'dev', 'test']:
    #        fp = f"data/ssc_input/article/folds/{fold['name']}_{set_type}_ssc.jsonl"
    #        write_ssc_input(fold[set_type], fp)

    #write_tok_ft_input(basil)

    #write_tapt_input(basil)

    write_cam_input(basil)

#write_allen_input()
