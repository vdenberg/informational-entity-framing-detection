import random
import json
import os, re
import pandas as pd


# Berg split helpers

def order_stories(basil):
    sizes = basil.story.value_counts()
    return sizes.index.to_list()


def cut_in_ten(ordered_stories):
    splits = [ordered_stories[i:i+10] for i in range(0, 100, 10)]
    return splits


def draw_1(split):
    i = random.randint(0, 9)
    drawn = split.pop(i)
    return drawn, split


def mix_into_ten_folds(list_of_non_random_stories):
    # shuffle within section of similar sizes
    for s in list_of_non_random_stories:
        random.shuffle(s)

    # zip up to mix sizes and make list to enable shuffling
    randomizes_stories = [list(tup) for tup in zip(*list_of_non_random_stories)]

    # shuffle again just to be sure
    for stories in randomizes_stories:
        random.shuffle(stories)

    return randomizes_stories


# Fan split helpers

def strip_totally(s):
    if isinstance(s, list):
        s = " ".join(s)

    regex = re.compile('[^a-zA-Z]')
    return regex.sub('', s)


def match_set_to_basil(tokens, basil):
    # gathers ids of tokens in set
    set_us = []

    for s in list(tokens):
        if s in basil:
            u = basil[s]
            set_us.append(u)
            basil.pop(s)
            tokens.pop(s)

    if len(tokens) > 0:
        handmade_mapping = {'TheFBIdirectorshouldbedrawnfromtheranksofcareerlawenforcementprosecutorsortheFBIitselfnotpoliticiansDurbintoldHuffPostlater': ['28hpo21'],
                            'ImonlyinterestedinLibyaifwegettheoilTrumpsaid': ['84fox26', '84hpo24'],
                            'HisconductisunbecomingofamemberofCongress': ['48nyt14','48fox12'],
                            'Iamtryingtosavelivesandpreventthenextterroristattack': ['64hpo6', '64nyt15'],
                            'Andyouexemplifyit': ['73hpo4', '73nyt2'],
                            'AndwhenyoursonlooksatyouandsaysMommalookyouwonBulliesdontwin': ['63nyt22', '63fox12'],
                            'Todaysrulingprovidescertaintyandclearcoherenttaxfilingguidanceforalllegallymarriedsamesexcouplesnationwide': ['66fox6', '66nyt7'],
                            'EricisagoodfriendandIhavetremendousrespectforhim': ['83fox8', '83hpo8'],
                            'ThecampaignandthestatepartyintendtocooperatewiththeUSAttorneysofficeandthestatelegislativecommitteeandwillrespondtothesubpoenasaccordingly': ['97fox4', '97hpo4'],
                            'AmericansdontbelievetheirleadersinWashingtonarelisteningandnowisthetimetochangethat': ['83fox4', '83fox12'],
                            'Icanthelpbutthinkthatthoseremarksarewellovertheline': ['50fox11', '50nyt6'],
                            'FaithmadeAmericastrongItcanmakeherstrongagain': ['87hpo4', '87nyt6'],
                            'ThefinalwordingwontbereleaseduntiltheNAACPsnationalboardofdirectorsapprovestheresolutionduringitsmeetinginOctober': ['42HPO7', '42FOX15'],
                            'Heobtainedatleastfivemilitarydefermentsfromtoandtookrepeatedstepsthatenabledhimtoavoidgoingtowaraccordingtorecords': ['73hpo7', '73nyt5'],
                            'Nomatterhowintrusiveandpartisanourpoliticscanbecomethisdoesnotjustifyapoorresponse': ['48nyt3', '48hpo42'],
                            'Itsaidsomethingcouldevolveandbecomemoredangerousforthatsmallpercentageofpeoplethatreallythinkourcountryhasbeentakenawayfromthem': ['42FOX17', '42HPO9']}
        for t in tokens:
            try:
                set_us.extend(handmade_mapping[t])
            except:
                pass #print(t)

    return set_us


# helpers for both

def load_basil():
    fp = 'data/basil.csv'
    basil_df = pd.read_csv(fp, index_col=0).fillna('')
    basil_df.index = [el.lower() for el in basil_df.index]
    basil_df = basil_df.rename({'bias': 'label'})
    return basil_df


def load_basil_w_tokens():
    fp = 'data/basil_w_tokens.csv'
    basil_df = pd.read_csv(fp, index_col=0).fillna('')
    basil_df.index = [el.lower() for el in basil_df.index]
    return basil_df


class BergSplit:
    def __init__(self, split_input, split_dir='data/splits/berg_split', permutation=False):
        self.split_dir = split_dir
        self.split_input = split_input
        self.basil = load_basil()
        self.split = self.load_berg_split(self.split_dir, permutation)

    def create_split(self, permutation):
        # order stories from most to least sentences in a story
        ordered_stories = order_stories(self.basil)

        if not permutation:
            #make ten cuts
            list_of_non_random_stories = cut_in_ten(ordered_stories)
        else:
            # make ten "cuts" that are just the same list that's gonna get mixed up to have varying sizes in dev and test
            list_of_non_random_stories = [ordered_stories for i in range(10)]

        splits = mix_into_ten_folds(list_of_non_random_stories)

        # now there's 10 folds of either size = 10 stories or size = 100 stories depending on whether we cut in 10 or not
        if not permutation:
            split = {'train': list(zip(*splits[:-2])),
                     'dev': [[el] for el in splits[-2]],  # 1 or 10 dev stories for each 8 or 80 training stories
                     'test': [[el] for el in splits[-1]]}  # 1 or 10 test stories for each 8 or 80 training stories
        else:
            split = {'train': list(zip(*splits[:-20])),
                     'dev': [[el] for el in splits[-20:-10]],  # 1dev stories for each 8 or 80 training stories
                     'test': [[el] for el in splits[-10:]]}  # 1 or 10 test stories for each 8 or 80 training stories

        with open(self.split_dir, 'w') as f:
            string = json.dumps(split)
            f.write(string)

        return split

    def load_berg_split(self, split_dir, permutation):
        split_fn = 'split_permuted.json' if permutation else 'split.json'
        split_fp = os.path.join(split_dir, split_fn)
        if not os.path.exists(split_fp):
            self.create_split(permutation)

        with open(split_fp, 'r') as f:
            return json.load(f)

    def map_split_to_sentences(self):
        by_st = self.basil.groupby('story')
        sent_by_st = {n: gr.index.to_list() for n, gr in by_st}

        mapping = pd.DataFrame(self.basil.index, index=self.basil.index, columns=['uniq_id'])
        mapping['split'] = None

        for fold_i in range(10):
            for section in self.split:
                stories = self.split[section][fold_i]
                for story in stories:
                    sents = sent_by_st[int(story)]
                    mapping.loc[sents,'split'] = str(fold_i) + '-' + section

        mapping = mapping.set_index('split')['uniq_id']
        return mapping

    def return_split(self):
        mapping = self.map_split_to_sentences()
        split = {}
        for fold_i in range(10):
            fold = split.setdefault(f'{fold_i}', {})

            train_sents = mapping.loc[str(fold_i) + '-train']
            dev_sents = mapping.loc[str(fold_i) + '-dev']
            test_sents = mapping.loc[str(fold_i) + '-test']

            fold['train'] = train_sents
            fold['dev'] = dev_sents
            fold['test'] = test_sents
        return split


class FanSplit:
    def __init__(self, split_input, split_dir):
        self.split_input = split_input
        self.basil = load_basil_w_tokens()
        self.split_dir = split_dir

    def load_fan_tokens(self, setname):
        with open(self.split_dir + '/' + setname + '_tokens.txt', encoding='utf-8') as f:
            toks = [el.strip() for el in f.readlines()]
        return toks

    def match_fan(self):
        basil = self.basil
        basil['split'] = 'train'
        basil['for_matching'] = basil.tokens.apply(strip_totally)

        basil_for_matching = {s: u for s, u in zip(basil.for_matching.values, basil.index.values)}
        sents = []
        for sn in ['train', 'val', 'test']:
            tokens = self.load_fan_tokens(sn)
            tokens = {strip_totally(s): None for s in tokens}
            us = match_set_to_basil(tokens, basil_for_matching)
            sents.append(us)
        train_sents, dev_sents, test_sents = sents
        return train_sents, dev_sents, test_sents

    def return_split(self):
        train_sents, dev_sents, test_sents = self.match_fan()
        return {'0': {'train': train_sents, 'dev': dev_sents, 'test': test_sents}}


class Split:
    def __init__(self, split_input, which='berg', split_loc='data/splits/', tst=False, permutation=False):
        """

        :type which: string specifying whether fan split or own split should be used
        :type feature: whether you want tokens, embeddings, or other from basil dataset
        """
        #assert isinstance(split_input, dict)

        self.split_input = split_input
        self.which = which
        self.basil = load_basil()
        self.tst = tst

        if self.which == 'fan':
            self.spl = FanSplit(split_input, split_dir=split_loc + '/fan_split', tst=tst)
            self.nr_folds = 1

        elif self.which == 'berg':
            self.spl = BergSplit(split_input, split_dir=split_loc + '/berg_split', tst=tst, permutation=permutation)
            self.nr_folds = 10

        else:
            print("Which split?")

        self.split = self.spl.return_split()

    def apply_split(self, features):
        split = self.spl.return_split()

        applied_split = {}
        for fold_i in range(self.nr_folds):

            train_sents = split[str(fold_i)]['train']
            dev_sents = split[str(fold_i)]['dev']
            test_sents = split[str(fold_i)]['test']

            if self.tst:
                train_sents, dev_sents, test_sents = train_sents[:50], dev_sents[:10], test_sents[:10]

            #train_dict = [self.split_input[id.lower()] for id in train_sents]
            #dev_dict = [self.split_input[id.lower()] for id in dev_sents]
            #test_dict = [self.split_input[id.lower()] for id in test_sents]

            train_df = self.basil.loc[dev_sents, features + ['label']]
            dev_df = self.basil.loc[dev_sents, features + ['label']]
            test_df = self.basil.loc[test_sents, features + ['label']]

            #train_X, train_y = train_df[features], train_df.label
            #dev_X, dev_y = dev_df[features], dev_df.label
            #test_X, test_y = test_df[features], test_df.label

            applied_split[fold_i]['train'] = {'as_dict': train_dict, 'as_df': train_df}
            applied_split[fold_i]['dev'] = {'as_dict': dev_dict, 'as_df': dev_df}
            applied_split[fold_i]['test'] = {'as_dict': test_dict, 'as_df': test_df}

            applied_split[fold_i]['sizes'] = (len(train_df), len(dev_df), len(test_df))

        return applied_split



def split_input_for_bert():
    infp = 'data/huggingface_input/basil.csv'
    data = pd.read_csv(infp, index_col=0)

    SPL = 'fan'
    spl = Split(data, which=SPL, split_loc='data/splits/fan_split',
                tst=False)
    folds = spl.apply_split(features=['id', 'bias', 'alpha', 'sentence'], input_as='huggingface',
                            output_as='huggingface')
    for fold in folds:
        fold['train'].to_csv('data/train.tsv', sep='\t', index=False, header=False)
        fold['dev'].to_csv('data/dev.tsv', sep='\t', index=False, header=False)
        fold['test'].to_csv('data/test.tsv', sep='\t', index=False, header=False)
        # note: data/all.tsv was made by hand
