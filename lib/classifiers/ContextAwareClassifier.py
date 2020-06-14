import torch
from torch import nn
from torch.autograd import Variable
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from torch.optim import lr_scheduler
from torch.utils.data import (DataLoader, SequentialSampler, RandomSampler, TensorDataset)
from lib.evaluate.Eval import my_eval
from transformers import BertModel, BertPreTrainedModel, BertForSequenceClassification
from lib.utils import format_runtime, format_checkpoint_filepath, get_torch_device
import os, time
import numpy as np

from torch.nn import CrossEntropyLoss, NLLLoss, MSELoss, Embedding, Dropout, Linear, Sigmoid, LSTM

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
        self.embedding = Embedding.from_pretrained(self.weights_matrix)
        self.emb_size = weights_matrix.shape[1]

        self.lstm = LSTM(self.input_size, self.hidden_size, num_layers=self.bilstm_layers, bidirectional=True)
        self.num_labels = 2
        self.dropout = Dropout(0.1)
        self.context_naive = context_naive

        if self.context_naive:
            self.classifier = Linear(self.emb_size, 2)
        else:
            self.classifier = Linear(self.hidden_size * 2, 2) #+ self.emb_size

        self.sigm = Sigmoid()

    def forward(self, inputs):
        """
        Forward pass.
        :param input_tensor: batchsize * seq_length
        :param target_idx: batchsize, specifies which token is to be classified
        :return: sigmoid output of size batchsize
        """

        # inputs
        token_ids, token_mask, contexts, positions = inputs
        # shapes and sizes
        batch_size = inputs[0].shape[0]
        sen_len = token_ids.shape[1]
        doc_len = contexts.shape[1]
        seq_len = doc_len

        # init containers for outputs
        rep_dimension = self.emb_size if self.context_naive else self.hidden_size * 2
        sentence_representations = torch.zeros(batch_size, seq_len, rep_dimension, device=self.device)
        #target_sent_reps = torch.zeros(batch_size, rep_dimension, device=self.device)
        target_sent_reps = torch.zeros(batch_size, self.emb_size, device=self.device)

        if self.context_naive:
            target_sent_reps = torch.zeros(batch_size, rep_dimension, device=self.device)
            for item, position in enumerate(positions):
                target_sent_reps[item] = self.embedding(contexts[item, position]).view(1, -1)

        else:
            hidden = self.init_hidden(batch_size)
            for seq_idx in range(contexts.shape[0]):
                embedded_sentence = self.embedding(contexts[:, seq_idx]).view(1, batch_size, -1)
                encoded, hidden = self.lstm(embedded_sentence, hidden)
                sentence_representations[:, seq_idx] = encoded

            for item, position in enumerate(positions):
                target_hid = sentence_representations[item, position].view(1, -1)

                target_roberta = self.embedding(contexts[item, position]).view(1, -1)
                target_sent_reps[item] = target_hid
                # target_sent_reps[item] = torch.cat((target_hid, target_roberta), dim=1)

            # target_sent_reps: bs * hid*2
            #target_sent_reps = torch.cat((target_sent_reps, sentence_representations[:, -1, :]), dim=-1)

        logits = self.classifier(target_sent_reps)
        probs = self.sigm(logits)
        return logits, probs, target_sent_reps

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.bilstm_layers * 2, batch_size, self.hidden_size, device=self.device)
        cell = torch.zeros(self.bilstm_layers * 2, batch_size, self.hidden_size, device=self.device)
        return Variable(hidden), Variable(cell)


class ContextAwareClassifier():
    def __init__(self, emb_dim=768, hid_size=32, layers=1, weights_mat=None, tr_labs=None,
                 b_size=24, cp_dir='models/checkpoints/cam', lr=0.001, start_epoch=0, patience=3,
                 step=1, gamma=0.75, n_eps=10, context_naive=False):
        self.start_epoch = start_epoch
        self.cp_dir = cp_dir
        self.device, self.use_cuda = get_torch_device()

        self.emb_dim = emb_dim
        self.hidden_size = hid_size
        self.batch_size = b_size
        self.criterion = CrossEntropyLoss(weight=torch.tensor([.15, .85], device=self.device))  # could be made to depend on classweight which should be set on input

        # self.criterion = NLLLoss(weight=torch.tensor([.15, .85], device=self.device))  # could be made to depend on classweight which should be set on input
        # set criterion on input
        # n_pos = len([l for l in tr_labs if l == 1])
        # class_weight = 1 - (n_pos / len(tr_labs))
        #self.criterion = nn.BCELoss(weight=torch.tensor([.15, .85], dtype=torch.float, device=self.device))

        self.context_naive = context_naive

        if start_epoch > 0:
            self.model = self.load_model()
        else:
            self.model = ContextAwareModel(input_size=self.emb_dim, hidden_size=self.hidden_size,
                                           bilstm_layers=layers, weights_matrix=weights_mat,
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
        nr_train_instances = len(tr_labs)
        nr_train_batches = int(nr_train_instances / b_size)
        half_tr_bs = int(nr_train_instances/2)
        self.optimizer = AdamW(self.model.parameters(), lr=lr, eps=1e-8)

        # self.scheduler = lr_scheduler.CyclicLR(self.optimizer, base_lr=lr, step_size_up=half_tr_bs,
        #                                       cycle_momentum=False, max_lr=lr * 30)

        num_train_optimization_steps = nr_train_batches * n_eps
        num_train_warmup_steps = int(0.1 * num_train_optimization_steps) #warmup_proportion

        # self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=num_train_warmup_steps,
        #                                                 num_training_steps=num_train_optimization_steps)  # PyTorch scheduler


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
        batch = tuple(t.to(self.device) for t in batch)
        inputs, labels = batch[:-1], batch[-1]

        self.model.zero_grad()
        logits, probs, _ = self.model(inputs)
        loss = self.criterion(logits.view(-1, 2), labels.view(-1))
        loss.backward()

        self.optimizer.step()
        #self.scheduler.step()
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
        embeddings = []
        for step, batch in enumerate(batches):
            batch = tuple(t.to(self.device) for t in batch)
            inputs, labels = batch[:-1], batch[-1]
            token_ids, token_mask, _, _ = inputs

            with torch.no_grad():
                logits, probs, sentence_representation = self.model(inputs)
                loss = self.criterion(logits.view(-1, 2), labels.view(-1))

                embedding = list(sentence_representation.detach().cpu().numpy())
                embeddings.append(embedding)
                #sigm_output  = self.model(ids, documents, positions)
                #sigm_output = sigm_output.detach().cpu().numpy()
                #loss = self.criterion(sigm_output, labels)

            probs = probs.detach().cpu().numpy() #probs.shape: batchsize * num_classes

            if len(y_pred) == 0:
                y_pred = probs
            else:
                y_pred = np.append(y_pred, probs, axis=0)

                # convert to predictions
                # #preds = [1 if output > 0.5 else 0 for output in sigm_output]
                #y_pred.extend(preds)

            sum_loss += loss.item()

        #y_pred = y_pred[0]
        y_pred = np.argmax(y_pred, axis=1)

        self.model.train()
        return y_pred, sum_loss / len(batches), embeddings

# _, USE_CUDA = get_torch_device()
# LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
# FloatTensor = torch.cuda.FLoatTensor if USE_CUDA else torch.FloatTensor
