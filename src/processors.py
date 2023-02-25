"""Dataset utils for different data settings for GLUE."""

import os
import copy
import logging
import torch
import numpy as np
import time
from filelock import FileLock
import json
import itertools
import random
import transformers
from transformers.data.processors.utils import InputFeatures
from transformers import DataProcessor, InputExample
from transformers.data.processors.glue import *
from transformers.data.metrics import glue_compute_metrics
import dataclasses
from dataclasses import dataclass, asdict
from typing import List, Optional, Union
# from sentence_transformers import SentenceTransformer, util
from copy import deepcopy
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class TextClassificationProcessor(DataProcessor):
    """
    Data processor for text classification datasets (mr, sst-5, subj, trec, cr, mpqa).
    """

    def __init__(self, task_name):
        self.task_name = task_name
        # from googletrans import Translator
        # self.translator = Translator(service_urls=[
        #     'translate.google.cn', ])
    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )
  
    def get_train_examples(self, data_dir):
        """See base class."""
        data_path = os.path.join(data_dir, "train" + ".npy")
        return self._create_examples(np.load(data_path, allow_pickle=True), "train")
        # data_path = os.path.join(data_dir, "train" + ".csv")
        # return self._create_examples(pd.read_csv(data_path, header=None).values.tolist(), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        data_path = os.path.join(data_dir, "dev" + ".npy")
        return self._create_examples(np.load(data_path, allow_pickle=True), "dev")
        # data_path = os.path.join(data_dir, "dev" + ".csv")
        # return self._create_examples(pd.read_csv(data_path, header=None).values.tolist(), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        data_path = os.path.join(data_dir, "test" + ".csv")
        return self._create_examples(pd.read_csv(data_path, header=None).values.tolist(), "test")

    def get_train_false_examples(self, data_dir):
        data_path = os.path.join(data_dir, "train" + "_false.npy")
        return self._create_examples(np.load(data_path, allow_pickle=True), "train")
        # data_path = os.path.join(data_dir, "train" + "_false.csv")
        # return self._create_examples(pd.read_csv(data_path, header=None).values.tolist(), "train")

    def get_labels(self):
        """See base class."""
        if self.task_name == "twi":
            return list(range(2))
        else:
            raise Exception("task_name not supported.")
        
    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            if len(line) <= 3:
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=line[-1], label=int(line[1])))
            else:
                text_b = {'relation': line[2], 'abs_relation': line[3]}
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=int(line[1])))
        return examples
        
def text_classification_metrics(task_name, preds, labels):
    return {"acc": (preds == labels).mean()}

# Add your task to the following mappings

processors_mapping = {
    "twi": TextClassificationProcessor("twi")
}

num_labels_mapping = {
    "twi":2
}

output_modes_mapping = {
    "twi":"classification"
}

# Return a function that takes (task_name, preds, labels) as inputs
compute_metrics_mapping = {
    "twi": text_classification_metrics
}
