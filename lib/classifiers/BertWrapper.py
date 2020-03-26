from transformers import BertPreTrainedModel, BertModel
from torch.nn import Dropout, Linear
from torch.nn import CrossEntropyLoss, MSELoss
import torch
from torch import nn
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import CyclicLR
import os
import numpy as np
from lib.utils import get_torch_device


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = Dropout(config.hidden_dropout_prob)
        self.classifier = Linear(config.hidden_size, self.config.num_labels)
        self.sigm = nn.Sigmoid()

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import BertTokenizer, BertForSequenceClassification
        import torch

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)

        loss, logits = outputs[:2]

        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0] # according to pytorch doc for BERTPretrainedModel: (batch_size, sequence_length, hidden_size)
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output) # according to pytorch doc for BERTPretrainedModel: (batch_size, hidden_size)
        logits = self.classifier(pooled_output)
        probs = self.sigm(logits)

        outputs = (logits, probs,) + (sequence_output, pooled_output,) # + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, probs, sequence_ouput, pooled_output, # (hidden_states), (attentions)


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


class BertWrapper:
    def __init__(self, bert_model, cache_dir, cp_dir, num_labels, bert_lr,
                 warmup_proportion, n_train_batches, n_epochs, load_from_ep=0):
        self.warmup_proportion = warmup_proportion
        self.device, self.use_cuda = get_torch_device()
        self.cache_dir = cache_dir
        self.cp_dir = cp_dir
        self.num_labels = num_labels

        self.model = self.load_model(bert_model, load_from_ep)
        self.model.to(self.device)
        if self.use_cuda:
            self.model.cuda()

        # set optim and scheduler
        self.optimizer = AdamW(self.model.parameters(), lr=bert_lr, eps=1e-8)
        num_train_optimization_steps = n_train_batches * n_epochs
        num_train_warmup_steps = int(self.warmup_proportion * num_train_optimization_steps)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=num_train_warmup_steps,
                                                         num_training_steps=num_train_optimization_steps)  # PyTorch scheduler
        #stepsize = int(n_train_batches/2)
        #self.scheduler = CyclicLR(self.optimizer, base_lr=bert_lr, max_lr=bert_lr*3,
        #                          step_size_up=stepsize, cycle_momentum=False)

    def train_on_batch(self, batch):
        self.model.zero_grad()
        batch = tuple(t.to(self.device) for t in batch)

        token_ids, token_masks, tok_seg_ids, _, _, labels, _ = batch

        outputs = self.model(input_ids=token_ids, attention_mask=token_masks, token_type_ids=tok_seg_ids, labels=labels)
        (loss), logits, probs, sequence_ouput, pooled_output = outputs
        loss = outputs[0]

        loss.backward()

        self.optimizer.step()
        self.scheduler.step()
        return loss.item()

    def predict(self, batches):
        self.model.eval()

        preds = []
        sum_loss = 0
        for step, batch in enumerate(batches):
            batch = tuple(t.to(self.device) for t in batch)
            token_ids, token_masks, tok_seg_ids, contexts, labels_fl, labels, positions = batch

            with torch.no_grad():
                (loss), logits, probs, sequence_output, pooled_output = self.model(input_ids=token_ids,
                                                                   attention_mask=token_masks,
                                                                   token_type_ids=tok_seg_ids, labels=labels)
                probs = probs.detach().cpu().numpy()

            if len(preds) == 0:
                preds.append(probs)
            else:
                preds[0] = np.append(preds[0], probs, axis=0)
            sum_loss += loss.item()

        preds = np.argmax(preds[0], axis=1)
        return preds, sum_loss / len(batches)

    def get_embedding_output(self, batch, emb_type):
        batch = tuple(t.to(self.device) for t in batch)
        token_ids, token_masks, tok_seg_ids, contexts, labels_fl, labels, positions = batch

        with torch.no_grad():
            probs, sequence_output, pooled_output = self.model(input_ids=token_ids,
                                                                     attention_mask=token_masks,
                                                                     token_type_ids=tok_seg_ids, labels=None)
            if emb_type == 'avbert':
                return sequence_output.mean(axis=1)

            elif emb_type == 'poolbert':
                return pooled_output

    def get_embeddings(self, batches, emb_type):
        self.model.eval()
        embeddings = []
        for step, batch in enumerate(batches):
            emb_output = self.get_embedding_output(batch, emb_type)

            if self.use_cuda:
                emb_output = list(emb_output[0].detach().cpu().numpy()) # .detach().cpu() necessary here on gpu

            else:
                emb_output = list(emb_output[0].numpy())
            embeddings.append(emb_output)

        return embeddings

    def save_model(self, model_dir, name):
        """
        Save bert model.
        :param model_dir: usually models/bert_for_embed/etc.
        :param name: usually number of current epoch
        """
        model_to_save = self.model

        output_dir = os.path.join(model_dir, name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_model_file = os.path.join(output_dir, "pytorch_model.bin")
        output_config_file = os.path.join(output_dir, "config.json")

        model_to_save = model_to_save.module if hasattr(model_to_save, 'module') else model_to_save  # Only save the model it-self
        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)

    def load_model(self, bert_model, load_from_ep):
        if not load_from_ep:
            return BertForSequenceClassification.from_pretrained(bert_model, cache_dir=self.cache_dir,
                                                                 num_labels=self.num_labels,
                                                                 output_hidden_states=False,
                                                                 output_attentions=False)
        else:
            load_dir = os.path.join(self.cp_dir, load_from_ep)
            return BertForSequenceClassification.from_pretrained(load_dir, num_labels=self.num_labels,
                                                                 output_hidden_states=False,
                                                                 output_attentions=False)

