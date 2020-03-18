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


def sample(ordered_stories):
    split_stories = cut_in_ten(ordered_stories)
    # shuffle
    for stories in split_stories:
        random.shuffle(stories)
    # zip and make list to enable shuffling
    mixed_stories = [list(tup) for tup in zip(*split_stories)]
    # shuffle again
    for stories in mixed_stories:
        random.shuffle(stories)
    return mixed_stories


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

def seperate_sets(basil, sents, features, output_as='df', tst=False):
    train_sents, dev_sents, test_sents = sents
    if tst:
      train_sents, dev_sents, test_sents = train_sents[:50], dev_sents[:10], test_sents[:10]

    if output_as != 'huggingface':
        train_data = basil.loc[train_sents, features + ['bias']]
        dev_data = basil.loc[dev_sents, features + ['bias']]
        test_data = basil.loc[test_sents, features + ['bias']]

    else:
        train_data = basil.loc[train_sents, features]
        dev_data = basil.loc[dev_sents, features]
        test_data = basil.loc[test_sents, features]

    if output_as == 'df' or output_as == 'huggingface':
        return {'train': train_data, 'dev': dev_data, 'test': test_data}

    else:
        train_X, train_y = train_data[features], train_data.bias
        dev_X, dev_y = dev_data[features], dev_data.bias
        test_X, test_y = test_data[features], test_data.bias
        sets = ((train_X, train_y), (dev_X, dev_y), (test_X, test_y))
        return sets


def load_basil():
    fp = 'data/basil.csv'
    basil_df = pd.read_csv(fp, index_col=0).fillna('')
    return basil_df


def load_basil_w_tokens():
    fp = 'data/basil_w_tokens.csv'
    basil_df = pd.read_csv(fp, index_col=0).fillna('')
    return basil_df


class BergSplit:
    def __init__(self, split_input, split_fp='data/splits/berg_split/split.json', tst=False):
        self.split_fp = split_fp
        self.split_input = split_input
        self.basil = load_basil()
        self.split = self.load_split(self.split_fp)
        self.tst = tst

    def create_split(self):
        ordered_stories = order_stories(self.basil)
        splits = sample(ordered_stories)
        split = {'train': list(zip(*splits[:-2])),
                'dev': [[el] for el in splits[-2]],  # 1 dev story for each 8 training stories
                'test': [[el] for el in splits[-1]]}  # 1 test story for each 8 training stories

        with open(self.split_fp, 'w') as f:
            string = json.dumps(split)
            f.write(string)

        return split

    def load_split(self, split_fp):
        if not os.path.exists(self.split_fp):
            self.create_split()

        with open(split_fp, 'r') as f:
            return json.load(f)

    def map_split_to_sentences(self):
        '''
        returns df
        '''
        by_st = self.basil.groupby('story')
        sent_by_st = {n: gr.index.to_list() for n, gr in by_st}

        mapping = pd.DataFrame(self.basil.index, index=self.basil.index)
        mapping['split'] = None

        for fold_i in range(10):
            for section in self.split:
                stories = self.split[section][fold_i]
                for story in stories:
                    sents = sent_by_st[int(story)]
                    mapping.loc[sents,'split'] = str(fold_i) + '-' + section

        mapping = mapping.set_index('split')['uniq_idx']
        return mapping

    def apply_split(self, features, input_as='df', output_as='df'):
        mapping = self.map_split_to_sentences()

        folds = []
        for fold_i in range(10):
            train_sents = mapping.loc[str(fold_i) + '-train']
            dev_sents = mapping.loc[str(fold_i) + '-dev']
            test_sents = mapping.loc[str(fold_i) + '-test']

            if input_as == 'df':
                sents = train_sents, dev_sents, test_sents
                sets = seperate_sets(self.split_input, sents, features, output_as=output_as, tst=self.tst)
                folds.append(sets)

            elif input_as == 'pytorch':
                fold = {}
                fold['train'] = [self.split_input[id.lower()] for id in train_sents]
                fold['dev'] = [self.split_input[id.lower()] for id in dev_sents]
                fold['test'] = [self.split_input[id.lower()] for id in test_sents]
                folds.append(fold)

        return folds


class FanSplit:
    def __init__(self, split_input, split_dir, tst):
        self.split_input = split_input
        self.basil = load_basil_w_tokens()
        self.split_dir = split_dir
        self.tst = tst

    def load_tokens(self, setname):
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
            tokens = self.load_tokens(sn)
            tokens = {strip_totally(s): None for s in tokens}
            us = match_set_to_basil(tokens, basil_for_matching)
            sents.append([el.lower() for el in us])
        return sents

    def apply_split(self, features, input_as=None, output_as='df'):
        folds = []
        for fold in range(1):
            sents = self.match_fan()

            if input_as == 'pytorch':
                train_sents, dev_sents, test_sents = sents

                fold = {}
                fold['train'] = [self.split_input[id.lower()] for id in train_sents]
                fold['dev'] = [self.split_input[id.lower()] for id in dev_sents]
                fold['test'] = [self.split_input[id.lower()] for id in test_sents]
                folds.append(fold)

            else:
                sets = seperate_sets(self.split_input, sents, features, output_as=output_as, tst=self.tst)
                folds.append(sets)

        return folds


class Split:
    def __init__(self, basil, which='berg', split_loc='data/splits/', tst=False):
        """

        :type which: string specifying whether fan split or own split should be used
        :type feature: whether you want tokens, embeddings, or other from basil dataset
        """
        if not isinstance(basil, dict):
            basil.index = [el.lower() for el in basil.index]
        self.basil = basil
        self.which = which
        if self.which == 'fan':
            self.spl = FanSplit(basil, split_dir=split_loc + '/fan_split', tst=tst)
        elif self.which == 'berg':
            self.spl = BergSplit(basil, split_fp=split_loc + '/berg_split/split.json', tst=tst)
        else:
            print("Which split?")

    def apply_split(self, features=['tokens'], input_as='df', output_as='df'):
        split_basil = self.spl.apply_split(features=features, input_as=input_as, output_as=output_as)
        if self.which == 'berg':
            for i, f in enumerate(split_basil):
                f['name'] = i
        elif self.which == 'fan':
            for i, f in enumerate(split_basil):
                f['name'] = 'fan'
        return split_basil


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
