import torch
from torch.utils.data import (DataLoader, SequentialSampler, RandomSampler, TensorDataset)
import os, math, time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker

#plt.switch_backend('agg')
#import matplotlib.pyplot as plt
#import matplotlib.ticker as ticker


def to_tensors(split, device):
    """ Tmp. """
    # to array if needed
    contexts = np.array([list(el) for el in split.context_doc_num.values])
    #token_ids = [list(el) for el in split.token_ids.values]
    #token_mask = [list(el) for el in split.token_mask.values]
    #tok_seg_ids = [list(el) for el in split.tok_seg_ids.values]
    #token_ids = np.array([list(el) for el in split.token_ids.values])
    #token_mask = np.array([list(el) for el in split.token_mask.values])
    #tok_seg_ids = np.array([list(el) for el in split.tok_seg_ids.values])

    # to tensors
    #token_ids = torch.tensor(token_ids, dtype=torch.long, device=device)
    #token_mask = torch.tensor(token_mask, dtype=torch.long, device=device)
    #tok_seg_ids = torch.tensor(tok_seg_ids, dtype=torch.long, device=device)
    contexts = torch.tensor(contexts, dtype=torch.long, device=device)
    labels_fl = torch.tensor(split.label.to_numpy(), dtype=torch.float, device=device)
    labels_long = torch.tensor(split.label.to_numpy(), dtype=torch.long, device=device)
    positions = torch.tensor(split.position.to_numpy(), dtype=torch.long, device=device)
    ids = torch.tensor(split.id_num.to_numpy(), dtype=torch.long, device=device)

    # to dataset
    #tensors = TensorDataset(ids, token_ids, token_mask, tok_seg_ids, contexts, labels_fl, labels_long, positions)
    tensors = TensorDataset(ids, labels_long)

    return tensors


def to_batches(tensors, batch_size):
    ''' Creates dataloader with input divided into batches. '''
    sampler = SequentialSampler(tensors) #RandomSampler(tensors)
    loader = DataLoader(tensors, sampler=sampler, batch_size=batch_size)
    return loader

'''
def to_tensor_for_bert(features, OUTPUT_MODE):
    example_ids = [f.my_id for f in features]
    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

    if OUTPUT_MODE == "classification":
        label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif OUTPUT_MODE == "regression":
        label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    data = TensorDataset(input_ids, input_mask, segment_ids, label_ids)
    return example_ids, data, label_ids  # example_ids, input_ids, input_mask, segment_ids, label_ids
'''


def indexesFromSentence(lang, sentence, EOS_token):
    indexes = [lang.word2index[word] for word in sentence.split(' ')]
    indexes.append(EOS_token)
    return indexes


def tensorFromSentence(lang, sentence, EOS_token, device):
    indexes = indexesFromSentence(lang, sentence, EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def get_torch_device():
    use_cuda = False
    if torch.cuda.is_available():
        # Tell PyTorch to use the GPU.
        device = torch.device("cuda")
        use_cuda = True

        #print('There are %d GPU(s) available.' % torch.cuda.device_count())
        #print('We will use the GPU:', torch.cuda.get_device_name(0))

    # If not...
    else:
        #print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    return device, use_cuda


def plot_scores(losses):
    tr_scores, dev_scores = zip(*losses)
    # print('debug loss plotting:')
    # print(tr_scores, dev_scores)
    plt.figure()
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(tr_scores)
    plt.plot(dev_scores)
    plt.legend(('train', 'dev'), loc='upper right')
    return plt

'''
def showPlot(points):
    # PLOTTING NOT CURRENTLY FUNCTIONING
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def asMinutes(s): # other option for formatting time
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
'''


def format_runtime(runtime):
    min = int(runtime // 60)
    sec = int(runtime % 60)
    return f'{min}m:{sec}s'


def format_checkpoint_filepath(cp_dir, bertcam=None, hidden_size='NA', epoch_number=None):
    if not epoch_number:
        print("Give epoch number to checkpoint name")
    cp_fn = f'{bertcam}_hidden{hidden_size}_lastepoch{epoch_number}.model'
    return os.path.join(cp_dir, cp_fn)

