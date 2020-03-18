from typing import Dict
import json
import logging
import numpy as np
from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, ArrayField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def map_vector(vector):
    return np.array([x for x in map(float, vector.strip('[]').split(','))])

@DatasetReader.register("basil_simple")
class BASILReader(DatasetReader):
    """
    Reads a JSON-lines file containing papers from the Semantic Scholar database, and creates a
    dataset suitable for document classification using these papers.

    Parameters
    ----------
    lazy : ``bool`` (optional, default=False)
        Passed to ``DatasetReader``.  If this is ``True``, training will start sooner, but will
        take longer per batch.  This also allows training with datasets that are too large to fit
        in memory.
    """
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line in data_file:
                line = line.strip("\n")
                if not line:
                    continue
                paper_json = json.loads(line)
                current = paper_json['current']
                left = paper_json['right']
                right = paper_json['right']
                label = paper_json['label']
                yield self.text_to_instance(current, left, right, label)

    @overrides
    def text_to_instance(self, current: list, left: list, right: list, label: str = None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        current = map_vector(current)
        current = map_vector(current)
        left = np.array([map_vector(x) for x in left])
        right = np.array([map_vector(x) for x in right])

        current_field = ArrayField(current)
        left_field = ArrayField(left)
        right_field = ArrayField(right)
        fields = {'current': current_field, 'left': left_field, 'right': right_field}
        if label is not None:
            fields['label'] = LabelField(label)
        return Instance(fields)

