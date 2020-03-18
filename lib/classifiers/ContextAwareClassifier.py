import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
import time
from datetime import timedelta
import random
from torch.utils.data import (DataLoader, SequentialSampler, RandomSampler, TensorDataset)
import matplotlib.pyplot as plt
from lib.evaluate.StandardEval import my_eval
from lib.utils import indexesFromSentence, format_runtime
#plt.switch_backend('agg')
import matplotlib.ticker as ticker
import os
from torch.optim.lr_scheduler import StepLR
import time
import math

SOS_token = 0
EOS_token = 1

"""
Based on: NLP From Scratch: Translation with a Sequence to Sequence Network and Attention
*******************************************************************************
**Author**: `Sean Robertson <https://github.com/spro/practical-pytorch>`_
"""


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


class FFLayer(nn.Module):
    def __init__(self, hidden_size):
        super(FFLayer, self).__init__()
        self.hidden_size = hidden_size
        self.out = nn.Sequential(nn.Linear(self.hidden_size, 1), nn.Sigmoid())

    def forward(self, input):
        return self.out(input)


class ContextAwareModel(nn.Module):
    def __init__(self, input_size, hidden_size, weights_matrix, device):
        super(ContextAwareModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device

        # weight matrix has shape vocab_size * embedding dimension[7987, 768]
        # input_size = sequence length = 96
        self.weights_matrix = torch.tensor(weights_matrix, dtype=torch.float, device=self.device)
        self.embedding = nn.Embedding.from_pretrained(self.weights_matrix)
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True)
        self.out = nn.Sequential(nn.Linear(self.hidden_size * 2, 1), nn.Sigmoid())

    def forward(self, input_tensor, target_idx, max_length):
        '''
        :param input_tensor: batchsize * seq_length
        :param target_idx: batchsize
        :param max_length: max sequence length, 96
        :return:
        '''
        # original
        # embedded = self.embedding(input).view(1, 1, -1)
        # output, hidden = self.lstm(embedded, hidden)
        # return output, hidden
        input_length = input_tensor.size(0)
        encoder_outputs = torch.zeros(max_length, self.hidden_size * 2, device=self.device)
        batch_size = target_idx.shape[0]

        hidden = self.initHidden(batch_size)
        # input tensor is batchsize * seq length
        for ei in range(input_length):
            embedded = self.embedding(input_tensor[:,ei]).view(1, batch_size, -1)
            output, hidden = self.lstm(embedded, hidden)
            encoder_outputs[ei] = output[0, 0]

        target_encoder_output = encoder_outputs[target_idx]
        output = self.out(target_encoder_output)

        return output

    def initHidden(self, batch_size):
        hidden = torch.zeros(2, batch_size, self.hidden_size, device=self.device)
        cell = torch.zeros(2, batch_size, self.hidden_size, device=self.device)
        return Variable(hidden), Variable(cell)


class ContextAwareClassifier():
    def __init__(self, model, input_lang, dev, device, batch_size=None, logger=None, cp_dir='models/checkpoints/cam', learning_rate=0.001, start_checkpoint=0, step_size=1, gamma=0.75):
        self.start_iter = start_checkpoint
        self.model = model
        self.input_lang = input_lang
        self.batch_size = batch_size
        self.max_length = input_lang.max_len
        self.criterion = None # depends on classweight which should be set on input
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        self.dev = dev
        self.cp_name = None # depends on split type and current fold
        self.best_perf = {'ep': 0, 'val': 30}
        self.device = device
        self.cp_dir = cp_dir
        self.logger = logger

    def to_tensor(self, triples):
        if self.batch_size == 1:
            indexedsentences = [indexesFromSentence(self.input_lang, t[0][0], EOS_token) for t in triples]
        else:
            indexedsentences = [indexesFromSentence(self.input_lang, t[0], EOS_token) for t in triples]
        input_tensor = torch.tensor(indexedsentences, dtype=torch.long, device=self.device)
        target_label_tensor = torch.tensor([t[1] for t in triples], dtype=torch.float, device=self.device)
        idx = torch.tensor([t[2] for t in triples], dtype=torch.long, device=self.device)
        data = TensorDataset(input_tensor, target_label_tensor, idx)
        return data

    def train(self, input_tensor, target_label_tensor, target_idx):
        #self.model.zero_grad()
        self.optimizer.zero_grad()

        loss = 0
        output = self.model(input_tensor, target_idx, self.max_length)
        self.logger(output)
        loss += self.criterion(output, target_label_tensor)
        loss.backward()

        self.optimizer.step()
        return loss.item()

    def save_checkpoint(self, cpdir, cpfn):
        checkpoint = {'model': self.model,
                      'state_dict': self.model.state_dict(),
                      'optimizer': self.optimizer.state_dict()}
        torch.save(checkpoint, os.path.join(cpdir, cpfn))

    def update_lr(self, best_ep, val_f1):
        self.best_perf = {'ep': best_ep, 'val': val_f1}
        self.scheduler.step()
        new_lr = self.scheduler.get_lr()
        print('\t\t{} - Updated LR: {} for f1 = {}'.format(best_ep, new_lr, self.best_perf['val']))
        self.logger.info('\t\t{} - Updated LR: {} for f1 = {}'.format(best_ep, new_lr, self.best_perf['val']))

    def check_performance(self, i):
        # check if its a good performance and whether to save / update LR

        val_f1 = self.evaluate(self.dev, which='f1')
        cpfn = 'cp_{}_{}_{}.pth'.format(self.cp_name, self.start_iter + i, val_f1)

        if val_f1 < 30:
            # val too low, do nothing
            pass
        elif val_f1 < self.best_perf['val']:
            # val below best, do nothing
            pass
        else:
            # save this configuration and check if lr reduction is warranted
            self.save_checkpoint(self.cp_dir + '/best', cpfn)

            diff = val_f1 - self.best_perf['val']  # e.g. 35 - 34.5
            if val_f1 < 32.5:
                # improvement has to be big
                if diff >= 0.5:
                    self.update_lr(i, val_f1)
            else:
                # improvement can be smaller
                if diff >= 0.25:
                    self.update_lr(i, val_f1)

    def train_batches(self, fold, print_step_every):
        self.cp_name = fold['name']

        training_triples = self.to_tensor(fold['train'])
        train_sampler = RandomSampler(training_triples)
        train_dataloader = DataLoader(training_triples, sampler=train_sampler, batch_size=self.batch_size)

        nr_steps = len(train_dataloader)

        loss_total = 0
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(self.device) for t in batch)
            input_tensor, target_label_tensor, target_idx = batch

            loss = self.train(input_tensor, target_label_tensor, target_idx)
            loss_total += loss

            if (step % print_step_every == 0) & (step > 0):
                update = f'\t\tFinished step {step}/{nr_steps} - loss: {loss}, lr: {self.scheduler.get_lr()}'
                self.logger.info(update)
        av_loss = loss_total / len(train_dataloader)

        return av_loss

    def train_epochs(self, fold, num_epochs, print_step_every, save_epoch_every):
        pos_freq = [t for t in fold['train'] if t[1] == 1]
        class_weight = 1 - (len(pos_freq) / len(fold['train']))
        self.criterion = nn.BCELoss(weight=torch.tensor(class_weight, dtype=torch.float, device=self.device))

        self.logger.info('Training...')
        total_loss = 0
        for ep in range(num_epochs):
            start_time = time.time()

            epoch_av_loss = self.train_batches(fold, print_step_every)
            val_f1, conf_mat_dict = self.evaluate(fold['dev'], which='f1', conf_mat=True)
            total_loss += epoch_av_loss

            elapsed = format_runtime(time.time()-start_time)
            if (ep % save_epoch_every == 0) & (ep > 0):
                epochs_av_loss = total_loss / ep
                update = f'\tEpoch {ep}/{num_epochs} (took {elapsed}): Av loss: {epoch_av_loss}, Val f1: {val_f1} ({conf_mat_dict})'
                self.logger.info(update)
                self.save_checkpoint(self.cp_dir, f"epoch{ep}")

    '''
    def evaluate(self, test_triples, which='f1'):
        with torch.no_grad():
            y_pred = []
            y_true = [t[1] for t in test_triples]
            for tr in test_triples:
                input_tensor, _, idx = self.to_tensor([tr])
                output = self.model(input_tensor, idx, self.max_length)
                pred = 1 if output > 0.5 else 0
                y_pred.extend([pred])
            metrics, metrics_string = evaluation(y_true, y_pred)
            if which == 'f1':
                f1 = round(metrics[-1] * 100,2)
                return f1
            elif which == 'all':
                return metrics, metrics_string
    '''

    def evaluate(self, test, which='f1', conf_mat=False):
        test_triples = self.to_tensor(test)
        test_sampler = RandomSampler(test_triples)
        test_dataloader = DataLoader(test_triples, sampler=test_sampler, batch_size=self.batch_size)

        y_true = []
        y_pred = []
        for batch in test_dataloader:
            # get output for batch
            batch = tuple(t.to(self.device) for t in batch)
            input_tensor, target_label_tensor, target_idx = batch
            outputs = self.model(input_tensor, target_idx, self.max_length)
            outputs = outputs.detach().cpu().numpy()

            # convert to predictions
            preds = [1 if output > 0.5 else 0 for output in outputs]
            y_true.extend([el for el in target_label_tensor.detach().cpu().numpy()])
            y_pred.extend(preds)

        metrics, metrics_df, metrics_string = my_eval('eval', y_true, y_pred)
        f1 = round(metrics['f1'] * 100, 2)
        conf_mat = {'tn': metrics['tn'], 'tp': metrics['tp'], 'fn': metrics['fn'], 'fp': metrics['fp']}

        if which == 'all':
            return metrics, metrics_df, metrics_string
        else:
            if which == 'f1':
                outputs = (f1,)
            if conf_mat:
                outputs += (conf_mat,)
            return outputs


#_, USE_CUDA = get_torch_device()
#LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
#FloatTensor = torch.cuda.FLoatTensor if USE_CUDA else torch.FloatTensor
