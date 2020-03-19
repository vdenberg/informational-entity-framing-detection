import torch
from torch.utils.data import (DataLoader, SequentialSampler, RandomSampler, TensorDataset)
import os, math, time
#plt.switch_backend('agg')
#import matplotlib.pyplot as plt
#import matplotlib.ticker as ticker


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

        print('There are %d GPU(s) available.' % torch.cuda.device_count())

        print('We will use the GPU:', torch.cuda.get_device_name(0))

    # If not...
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    return device, use_cuda



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


def format_checkpoint_name(cp_dir, hidden_size, epoch_number=None):
    if not epoch_number:
        print("Give epoch number to checkpoint name")
    cp_fn = f'hidden{hidden_size}_epoch{epoch_number}.model'
    return os.path.join(cp_dir, cp_fn)

