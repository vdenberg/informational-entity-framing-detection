import torch
from torch import nn
from torch.autograd import Variable
from transformers.optimization import AdamW
from torch.optim import lr_scheduler
from torch.utils.data import (DataLoader, SequentialSampler, RandomSampler, TensorDataset)
from lib.evaluate.StandardEval import my_eval
from lib.utils import format_runtime, format_checkpoint_filepath, get_torch_device
import os, time
import numpy as np

from torch.nn import CrossEntropyLoss

"""
Based on: NLP From Scratch: Translation with a Sequence to Sequence Network and Attention
*******************************************************************************
**Author**: `Sean Robertson <https://github.com/spro/practical-pytorch>`_
"""


class ContextAwareModel(nn.Module):
    """
    Model that applies BiLSTM and classification of hidden representation of token at target index.
    :param input_size: length of input sequences (= documents)
    :param hidden_size: size of hidden layer
    :param weights_matrix: matrix of embeddings of size vocab_size * embedding dimension
    """
    def __init__(self, input_size, hidden_size, bilstm_layers, weights_matrix, context_naive, device):
        super(ContextAwareModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bilstm_layers = bilstm_layers
        self.device = device

        self.weights_matrix = torch.tensor(weights_matrix, dtype=torch.float, device=self.device)
        self.embedding = nn.Embedding.from_pretrained(self.weights_matrix)

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, num_layers=self.bilstm_layers, bidirectional=True)

        self.dropout = nn.Dropout(0.1)
        #self.classifier = nn.Sequential(nn.Linear(self.hidden_size * 2, 1), nn.Sigmoid())
        self.classifier = nn.Linear(self.hidden_size * 2, 2)
        self.sigm = nn.Sigmoid()
        self.context_naive = context_naive

    def forward(self, input_tensor, target_idx):
        """
        Forward pass.
        :param input_tensor: batchsize * seq_length
        :param target_idx: batchsize, specifies which token is to be classified
        :return: sigmoid output of size batchsize
        """
        batch_size = input_tensor.shape[0]
        seq_length = input_tensor.shape[1]

        if self.context_naive:
            target_embeddings = torch.zeros(batch_size, 1, self.hidden_size * 2, device=self.device)
            for item in range(batch_size):
                my_idx = target_idx[item]
                print(input_tensor.shape)
                print(input_tensor[item, my_idx])
                target_embeddings[item] = self.embedding(input_tensor[item, my_idx]).view(1, 1, -1)
        else:
            context_encoder_outputs = torch.zeros(self.input_size, batch_size, self.hidden_size * 2, device=self.device)

            # loop through input and update hidden
            hidden = self.init_hidden(batch_size)
            for ei in range(seq_length):
                embedded = self.embedding(input_tensor[:, ei]).view(1, batch_size, -1) # get sentence embedding for that item
                output, hidden = self.lstm(embedded, # feed hidden of previous token/item, store in hidden again
                                           hidden)  # output has shape 1 (for token in question) * batchsize * (hidden * 2)
                context_encoder_outputs[ei] = output[0]

            # loop through batch to get token at desired index
            target_output = torch.zeros(batch_size, 1, self.hidden_size * 2, device=self.device)
            for item in range(batch_size):
                my_idx = target_idx[item]
                target_output[item] = context_encoder_outputs[my_idx, item, :]

        logits = self.classifier(target_output).view(batch_size)  # sigmoid function that returns batch_size * 1
        probs = self.sigm(logits)
        return logits, probs

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.bilstm_layers * 2, batch_size, self.hidden_size, device=self.device)
        cell = torch.zeros(self.bilstm_layers * 2, batch_size, self.hidden_size, device=self.device)
        return Variable(hidden), Variable(cell)


class ContextAwareClassifier():
    def __init__(self, emb_dim=768, hidden_size=32, bilstm_layers=1, weights_matrix=None, train_labels=None,
                 batch_size=24, cp_dir='models/checkpoints/cam',
                 learning_rate=0.001, start_epoch=0, patience=3, step_size=1, gamma=0.75, context_naive=False):
        self.start_epoch = start_epoch
        self.cp_dir = cp_dir
        self.device, self.use_cuda = get_torch_device()

        self.emb_dim = emb_dim
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.criterion = None  # depends on classweight which should be set on input

        if start_epoch > 0:
            self.model = self.load_model()
        else:
            self.model = ContextAwareModel(input_size=self.emb_dim, hidden_size=self.hidden_size,
                                           bilstm_layers=bilstm_layers, weights_matrix=weights_matrix,
                                           device=self.device, context_naive=context_naive)
        self.model = self.model.to(self.device)
        if self.use_cuda: self.model.cuda()

        # empty now and set during or after training
        self.train_time = 0
        self.prev_val_f1 = 0
        self.cp_name = None  # depends on split type and current fold
        self.full_patience = patience
        self.current_patience = self.full_patience
        self.test_perf = []
        self.test_perf_string = ''

        # set optim and scheduler
        nr_train_instances = len(train_labels)
        nr_train_batches = int(nr_train_instances / batch_size)
        half_tr_bs = int(nr_train_instances/2)
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate, eps=1e-8)
        self.scheduler = lr_scheduler.CyclicLR(self.optimizer, base_lr=learning_rate, step_size_up=half_tr_bs,
                                               cycle_momentum=False, max_lr=learning_rate*30)

        # set criterion
        n_pos = len([l for l in train_labels if l == 1])
        class_weight = 1 - (n_pos / len(train_labels))
        #self.criterion = nn.BCELoss(weight=torch.tensor(class_weight, dtype=torch.float, device=self.device))
        self.criterion = CrossEntropyLoss()

    def load_model(self, name):
        cpfp = os.path.join(self.cp_dir, name)
        cp = torch.load(cpfp)
        model = cp['model']
        model.load_state_dict(cp['state_dict'])
        self.model = model
        self.model.to(self.device)
        if self.use_cuda: self.model.cuda()
        return model

    def train_on_batch(self, batch):
        self.model.zero_grad()
        batch = tuple(t.to(self.device) for t in batch)

        _, _, _, documents, labels, labels_long, positions = batch

        logits, probs = self.model(documents, positions)
        #print(labels.type())
        loss = self.criterion(probs, labels)
        loss.backward()

        self.optimizer.step()
        self.scheduler.step()
        return loss.item()

    def save_model(self, name):
        checkpoint = {'model': self.model,
                      'state_dict': self.model.state_dict(),
                      'optimizer': self.optimizer.state_dict()}
        cpfp = os.path.join(self.cp_dir, name)
        torch.save(checkpoint, cpfp)

    def predict(self, batches):
        self.model.eval()

        y_pred = []
        sum_loss = 0
        for step, batch in enumerate(batches):
            batch = tuple(t.to(self.device) for t in batch)
            _, _, _, documents, labels, labels_long, positions = batch

            with torch.no_grad():
                #sigm_output = self.model(documents, positions)
                logits, probs = self.model(documents, positions)
                #loss = self.criterion(sigm_output, labels)
                loss = self.criterion(logits.view(-1, 2), labels.view(-1))
                #sigm_output = sigm_output.detach().cpu().numpy()

            if len(y_pred) == 0:
                y_pred.append(probs.detach().cpu().numpy())
            else:
                y_pred[0] = np.append(y_pred[0], probs.detach().cpu().numpy(), axis=0)

            # convert to predictions
            #preds = [1 if output > 0.5 else 0 for output in sigm_output]
            #y_pred.extend(preds)

            sum_loss += loss.item()

        y_pred = y_pred[0]
        y_pred = np.argmax(y_pred, axis=1)

        self.model.train()
        return y_pred, sum_loss / len(batches)

# _, USE_CUDA = get_torch_device()
# LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
# FloatTensor = torch.cuda.FLoatTensor if USE_CUDA else torch.FloatTensor
