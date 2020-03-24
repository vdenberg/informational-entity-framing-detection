from transformers import BertPreTrainedModel, BertModel
from torch.nn import Dropout, Linear
from torch.nn import CrossEntropyLoss, MSELoss
import torch
from torch import nn
import numpy as np
from torch.utils.data import (DataLoader, SequentialSampler)
from lib.evaluate.StandardEval import my_eval
import os


# model


class Inferencer():
    def __init__(self, reports_dir, output_mode, logger, device, use_cuda):
        self.device = device
        self.output_mode = output_mode
        self.reports_dir = reports_dir
        self.logger = logger
        self.device = device
        self.use_cuda = use_cuda

    def predict(self, model, batched_data, embedding_type='avbert'):
        model.to(self.device)
        model.eval()

        preds = []
        embeddings = []
        for step, batch in enumerate(batched_data):
            batch = tuple(t.to(self.device) for t in batch)
            token_ids, token_masks, tok_seg_ids, contexts, labels, positions = batch

            with torch.no_grad():
                outputs = model(input_ids, segment_ids, input_mask, labels=None)
                logits, probs, sequence_output, pooled_output, (hidden_states), (attentions) = outputs
                if embedding_type == 'avbert':
                    representation = sequence_output.mean(axis=1) # (batch_size, sequence_length, hidden_size) ->  batch_size, hidden_size)
                elif embedding_type == 'pooled_output':
                    representation = pooled_output  # (batch_size, hidden_size)

            # of last hidden state with size (batch_size, sequence_length, hidden_size)
            # where batch_size=1, sequence_length=95, hidden_size=768)
            # take average of sequence, size (batch_size, hidden_size)

            if self.use_cuda:
                representation = list(representation[0].detach().cpu().numpy()) # .detach().cpu() necessary here on gpu
            else:
                representation = list(representation[0].numpy()) # .detach().cpu() necessary here on gpu
            embeddings.append(representation)

            if len(preds) == 0:
                preds.append(probs.detach().cpu().numpy())
            else:
                preds[0] = np.append(preds[0], probs.detach().cpu().numpy(), axis=0)

        preds = preds[0]
        if self.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif self.output_mode == "regression":
            preds = np.squeeze(preds)

        model.train()
        if return_embeddings:
            return embeddings
        else:
            return preds

    def eval(self, model, data, labels, av_loss=None, name='Basil'):
        preds = self.predict(model, data)
        metrics_dict, metrics_df, metrics_string = my_eval(name, labels.numpy(), preds, av_loss=av_loss)

        if av_loss:
            metrics_df = metrics_df.rename(columns={'average_loss': 'validation_loss', 'f1': 'validation_f1'})
            metrics_df = metrics_df[['tp', 'tn', 'fp', 'fn', 'acc', 'rec', 'prec', 'validation_loss', 'validation_f1']]
        else:
            metrics_df = metrics_df.rename(columns={'f1': 'validation_f1'})
            metrics_df = metrics_df[['tp', 'tn', 'fp', 'fn', 'acc', 'rec', 'prec', 'validation_f1']]

        output_eval_file = os.path.join(self.reports_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            self.logger.info("\n***** Eval results *****")
            self.logger.info(f'\n{metrics_df}')
            #self.logger.info(f'Sample of predictions: {preds[:20]}')
            for key in (metrics_dict.keys()):
                writer.write("%s = %s\n" % (key, str(metrics_dict[key])))

        return metrics_dict, metrics_string


def save_model(model_to_save, model_dir, identifier):
    output_dir = os.path.join(model_dir, identifier)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
    output_config_file = os.path.join(output_dir, "config.json")

    model_to_save = model_to_save.module if hasattr(model_to_save, 'module') else model_to_save  # Only save the model it-self
    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
