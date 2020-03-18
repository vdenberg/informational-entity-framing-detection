from __future__ import absolute_import, division, print_function
import os
import sys
import logging
import pandas as pd
from lib.handle_data.SplitData import Split
import csv

logger = logging.getLogger()
csv.field_size_limit(2147483647) # Increase CSV reader's field limit incase we have long text.


def split_input():
    infp = 'data/cam_input/basil.tsv'
    data = pd.read_csv(infp, index_col=0, sep='\t', header=None).fillna('')
    data.columns = ['sentence', 'bias', 'index']
    data['alpha'] = 'a'
    data['id'] = data.index

    SPL = 'fan'
    spl = Split(data, which=SPL, tst=False)
    folds = spl.apply_split(features=['id', 'index', 'bias', 'alpha', 'sentence'], input_as='huggingface',
                            output_as='huggingface')
    for fold in folds:
        fold['train'].to_csv('data/cam_input/train.tsv', sep='\t', index=False, header=False)
        fold['dev'].to_csv('data/cam_input/dev.tsv', sep='\t', index=False, header=False)
        fold['test'].to_csv('data/cam_input/test.tsv', sep='\t', index=False, header=False)
        # note: data/all.tsv was made by hand
split_input()

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, my_id, index, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.my_id = my_id
        self.index = index
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(cell for cell in line)
                lines.append(line)
            return lines


class BinaryClassificationProcessor(DataProcessor):
    """Processor for binary classification dataset."""

    def get_examples(self, fp, name):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(fp)), name)

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines): #['id', 'index', 'bias', 'alpha', 'sentence']
            guid = "%s-%s" % (set_type, i)
            my_id = line[0]
            index = line[1]
            text_a = line[4]
            label = line[2]
            examples.append(
                InputExample(guid=guid, my_id=my_id, index=my_id, text_a=text_a, text_b=None, label=label))
        return examples


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, my_id, index, input_ids, input_mask, segment_ids, label_id):
        self.my_id = my_id
        self.index = index
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_example_to_feature(example_row):
    # return example_row
    example, label_map, max_seq_length, tokenizer, output_mode = example_row

    tokens_a = tokenizer.tokenize(example.text_a)

    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
    segment_ids = [0] * len(tokens)

    if tokens_b:
        tokens += tokens_b + ["[SEP]"]
        segment_ids += [1] * (len(tokens_b) + 1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    if output_mode == "classification":
        label_id = label_map[example.label]
    elif output_mode == "regression":
        label_id = float(example.label)
    else:
        raise KeyError(output_mode)

    return InputFeatures(my_id=example.my_id,
                         index=example.id,
                         input_ids=input_ids,
                         input_mask=input_mask,
                         segment_ids=segment_ids,
                         label_id=label_id)
