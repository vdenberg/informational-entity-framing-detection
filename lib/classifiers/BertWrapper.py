from transformers import BertForSequenceClassification, RobertaForSequenceClassification
from torch.nn import Dropout, Linear
from torch.nn import CrossEntropyLoss, MSELoss
import torch
from torch import nn
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import CyclicLR
import os, pickle
import numpy as np
from lib.utils import get_torch_device, to_tensor, to_batches
from torch.nn import CrossEntropyLoss, MSELoss, Embedding, Dropout, Linear, Sigmoid, LSTM
from lib.evaluate.Eval import my_eval

# helpers
class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, my_id, input_ids, input_mask, segment_ids, label_id):
        self.my_id = my_id
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def load_features(fp, batch_size):
    with open(fp, "rb") as f:
        ids, data, labels = to_tensor(pickle.load(f))
    batches = to_batches(data, batch_size=batch_size)
    return ids, batches, labels


class MyBert():
    def __init__(self, start_bert_model, num_labels, device, cache_dir):
        self.start_bert_model = start_bert_model

        self.ROBERTA = True if 'roberta' in self.start_bert_model else False
        self.device = device
        self.sigm = nn.Sigmoid()
        self.loss_fct = CrossEntropyLoss()
        self.num_labels = num_labels
        self.cache_dir = cache_dir

    def init_fresh(self):
        if self.ROBERTA:
            model = RobertaForSequenceClassification.from_pretrained(self.start_bert_model, cache_dir=self.cache_dir,
                                                                     num_labels=self.num_labels, output_hidden_states=False,
                                                                     output_attentions=False)
        else:
            model = BertForSequenceClassification.from_pretrained(self.start_bert_model, cache_dir=self.cache_dir, num_labels=self.num_labels,
                                                                  output_hidden_states=False, output_attentions=False)
        model.to(self.device)
        return model

    def init_model(self, bert_model=None, cache_dir=None, num_labels=2):
        if not bert_model:
            model = self.init_fresh(self.start_bert_model)

        else:
            if self.ROBERTA:
                model = RobertaForSequenceClassification.from_pretrained(bert_model,
                                                                         num_labels=self.num_labels,
                                                                         output_hidden_states=False,
                                                                         output_attentions=False)
            else:
                model = BertForSequenceClassification.from_pretrained(bert_model,
                                                                      num_labels=self.num_labels,
                                                                      output_hidden_states=False,
                                                                      output_attentions=False)

            model.to(self.device)
        return model

    def my_forward(self, model, input_ids=None, attention_mask=None, labels=None, emb_type='regular'):
        if self.ROBERTA:
            outputs = model.roberta(input_ids, attention_mask=attention_mask)
            sequence_output = outputs[0]
            logits = model.classifier(sequence_output)

            emb_output = sequence_output.mean(axis=1)
        else:
            outputs = model.bert(input_ids, attention_mask=attention_mask)
            pooled_output = outputs[1]
            pooled_output = model.dropout(pooled_output)
            logits = model.classifier(pooled_output)

            emb_output = pooled_output

        outputs = (logits, emb_output)

        if labels is not None:
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
            '''   
            TOKEN CLASSIFICATION:     
            if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
            '''
        logits = outputs[1]
        probs = self.sigm(logits)
        outputs = outputs + (probs,)
        return outputs

    def my_predict(self, model, data, return_embeddings=False, output_mode='classification'):
        model.eval()
        preds = []
        embeddings = []
        for step, batch in enumerate(data):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                # print(input_mask)
                outputs = self.my_forward(model, batch[0], batch[1], labels=None)
                logits, emb_output, probs = outputs

            logits = logits.detach().cpu().numpy()
            if output_mode == 'bio_classification':
                preds.extend([list(p) for p in np.argmax(logits, axis=2)])
            elif output_mode == 'classification':
                preds.extend(np.argmax(logits, axis=1))

            if return_embeddings:
                emb_output = list(emb_output[0].detach().cpu().numpy())
                embeddings.append(emb_output)

        model.train()
        return preds, embeddings

    def my_eval(self, model, data, labels, av_loss=None, set_type='dev', name='Basil', output_mode='classification'):
        preds, embeddings = self.my_predict(model, data, output_mode)

        if output_mode == 'bio_classification':
            labels = labels.numpy().flatten()
            preds = np.asarray(preds)
            preds = np.reshape(preds, labels.shape)
        else:
            labels = labels.numpy()

        metrics_dict, metrics_string = my_eval(labels, preds, set_type=set_type, av_loss=av_loss, name=name)

        return metrics_dict, metrics_string



def save_bert_model(model_to_save, model_dir, identifier):
    ''' Save finetuned (finished or intermediate) BERT model to a checkpoint. '''
    output_dir = os.path.join(model_dir, identifier)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
    output_config_file = os.path.join(output_dir, "config.json")

    model_to_save = model_to_save.module if hasattr(model_to_save, 'module') else model_to_save  # Only save the model it-self
    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)


class BertWrapperOld:
    def __init__(self, cp_dir, n_eps, n_train_batches, load_from_path=0,
                 bert_model='bert-base-cased', cache_dir='models/cache', num_labels=2,
                 bert_lr=2e-6, warmup_proportion=0.1, seed_val=None):


        self.warmup_proportion = warmup_proportion
        self.device, self.use_cuda = get_torch_device()
        self.cache_dir = cache_dir
        self.cp_dir = cp_dir
        self.num_labels = num_labels

        self.model = self.load_model(bert_model=bert_model, load_from_path=load_from_path)
        self.model.to(self.device)
        if self.use_cuda:
            self.model.cuda()

        # set criterion, optim and scheduler
        self.criterion = CrossEntropyLoss()
        self.optimizer = AdamW(self.model.parameters(), lr=bert_lr, eps=1e-8)
        num_train_optimization_steps = n_train_batches * n_eps
        num_train_warmup_steps = int(self.warmup_proportion * num_train_optimization_steps)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=num_train_warmup_steps,
                                                         num_training_steps=num_train_optimization_steps)  # PyTorch scheduler
        #stepsize = int(n_train_batches/2)
        #self.scheduler = CyclicLR(self.optimizer, base_lr=bert_lr, max_lr=bert_lr*3,
        #                          step_size_up=stepsize, cycle_momentum=False)

    def train_on_batch(self, batch):
        batch = tuple(t.to(self.device) for t in batch)
        inputs, labels = batch[:-1], batch[-1]
        input_ids, input_mask, _, _ = inputs

        self.model.zero_grad()
        outputs = self.model(input_ids, input_mask, labels=labels)
        (loss), logits, probs, sequence_ouput, pooled_output = outputs
        loss = self.criterion(logits.view(-1, 2), labels.view(-1))
        loss.backward()

        self.optimizer.step()
        self.scheduler.step()
        return loss.item()

    def predict(self, batches):
        self.model.my_eval()

        y_pred = []
        sum_loss = 0
        embeddings = []
        for step, batch in enumerate(batches):
            batch = tuple(t.to(self.device) for t in batch)
            inputs, labels = batch[:-1], batch[-1]
            input_ids, input_mask, _, _ = inputs

            with torch.no_grad():
                outputs = self.model(input_ids, input_mask, labels=None)
                logits, probs, sequence_output, pooled_output = outputs
                loss = self.criterion(logits.view(-1, self.num_labels), labels.view(-1))
                probs = probs.detach().cpu().numpy()

                embedding = list(pooled_output.detach().cpu().numpy())
                embeddings.append(embedding)

            if len(y_pred) == 0:
                y_pred.append(probs)
            else:
                y_pred[0] = np.append(y_pred[0], probs, axis=0)
            sum_loss += loss.item()

        y_pred = np.argmax(y_pred[0], axis=1)
        return y_pred, sum_loss / len(batches), embeddings

    def get_embedding_output(self, batch, emb_type):
        batch = tuple(t.to(self.device) for t in batch)
        _, _, input_ids, input_mask, label_ids = batch

        with torch.no_grad():
            outputs = self.model(input_ids, input_mask, labels=None)
            logits, probs, sequence_output, pooled_output = outputs

            if emb_type == 'avbert':
                return sequence_output.mean(axis=1)

            elif emb_type == 'poolbert':
                return pooled_output

    def get_embeddings(self, batches, emb_type, model_path=''):
        if model_path:
            self.load_model(load_from_path=model_path)

        self.model.my_eval()
        embeddings = []
        for step, batch in enumerate(batches):
            emb_output = self.get_embedding_output(batch, emb_type)

            if self.use_cuda:
                emb_output = list(emb_output.detach().cpu().numpy()) # .detach().cpu() necessary here on gpu
            else:
                emb_output = list(emb_output.numpy())
            embeddings.append(emb_output)
        return embeddings

    def save_model(self, name):
        """
        Save bert model.
        :param model_dir: usually models/bert_for_embed/etc.
        :param name: usually number of current epoch
        """
        model_to_save = self.model

        output_dir = os.path.join(self.cp_dir, name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_model_file = os.path.join(output_dir, "pytorch_model.bin")
        output_config_file = os.path.join(output_dir, "config.json")

        model_to_save = model_to_save.module if hasattr(model_to_save, 'module') else model_to_save  # Only save the model it-self
        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)

    def load_model(self, load_from_path=None, bert_model='bert-base-cased'):
        if not load_from_path:
            return BertForSequenceClassification.from_pretrained(bert_model, cache_dir=self.cache_dir,
                                                                 num_labels=self.num_labels,
                                                                 output_hidden_states=False,
                                                                 output_attentions=False)
        elif load_from_path:
            return BertForSequenceClassification.from_pretrained(load_from_path, num_labels=self.num_labels,
                                                                 output_hidden_states=False,
                                                                 output_attentions=False)
        #elif load_from_ep:
        #    load_dir = os.path.join(self.cp_dir, load_from_ep)
        #    return BertForSequenceClassification.from_pretrained(load_dir, num_labels=self.num_labels,
        #                                                         output_hidden_states=False,
        #                                                         output_attentions=False)

