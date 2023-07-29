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
# from src.processors import processors_mapping, num_labels_mapping, output_modes_mapping, compute_metrics_mapping, \
#     median_mapping
from transformers.data.processors.utils import InputFeatures
from transformers import DataProcessor, InputExample
import dataclasses
from dataclasses import dataclass
from typing import List, Optional, Union
from sentence_transformers import SentenceTransformer, util
from copy import deepcopy
import pandas as pd
from src.processors import TextClassificationProcessor

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OurInputFeatures(InputFeatures):
    """
    Inherit from Transformers' InputFeatuers.
    """
    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    label: Optional[Union[int, float]] = None
    template_input_ids: List[int] = None
    template_attention_mask: Optional[List[int]] = None
    mask_pos: Optional[List[int]] = None  # Position of the mask token
    comment_pos: Optional[List[int]] = None
    relation: Optional = None
    relation_sent: Optional = None
    abs_relation: Optional = None
    label_word_list: Optional[List[int]] = None  # Label word mapping (dynamic)

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"


def input_example_to_string(example, sep_token):
    if example.text_b is None:
        return example.text_a
    else:
        # Warning: very simple hack here
        return example.text_a + ' ' + sep_token + ' ' + example.text_b


def input_example_to_tuple(example):
    if example.text_b is None:
        if pd.isna(example.text_a) or example.text_a is None:
            return ['']
            logger.warn("Empty input")
        else:
            return [example.text_a, example.label]
    else:
        return [example.text_a, example.text_b]


def tokenize_multipart_input(
        input_text_list,
        context_max_length,
        template_max_length,
        tokenizer,
        position_dataset=False,
        template=None,
        label_word_list=None,
        pooling_sent_limit=None,
        truncate_head=False,
        support_labels=None,
):
    def enc(text):
        return tokenizer.encode(text, add_special_tokens=False)
    relation_dict = input_text_list[1]
    relation = relation_dict['relation']
    abs_relation = relation_dict['abs_relation']
    input_text_list = [input_text_list[0]]
    input_ids = []
    attention_mask = []
    token_type_ids = []  # Only for BERT
    template_attention_mask = []
    template_token_type_ids = []  # Only for BERT
    pooling_sent_limit = pooling_sent_limit
    mask_pos = None
    comment_pos = None# Position of the mask token

    """
    Concatenate all sentences and prompts based on the provided template.
    Template example: '*cls*It was*mask*.*sent_0**<sep>*label_0:*sent_1**<sep>**label_1*:*sent_2**<sep>*'
    *xx* represent variables:
        *cls*: cls_token
        *mask*: mask_token
        *sep*: sep_token
        *sep+*: sep_token, also means +1 for segment id
        *sent_i*: sentence i (input_text_list[i])
        *sent-_i*: same as above, but delete the last token
        *sentl_i*: same as above, but use lower case for the first word
        *sentl-_i*: same as above, but use lower case for the first word and delete the last token
        *+sent_i*: same as above, but add a space before the sentence
        *+sentl_i*: same as above, but add a space before the sentence and use lower case for the first word
        *label_i*: label_word_list[i]
        *label_x*: label depends on the example id (support_labels needed). this is only used in GPT-3's in-context learning

    Use "_" to replace space.
    PAY ATTENTION TO SPACE!! DO NOT leave space before variables, for this will lead to extra space token.
    """
    assert template is not None

    special_token_mapping = {
        'cls': tokenizer.cls_token_id, 'mask': tokenizer.mask_token_id, 'sep': tokenizer.sep_token_id,
        'sep+': tokenizer.sep_token_id,
    }
    template_list = template.split('*')  # Get variable list in the template
    segment_id = 0  # Current segment id. Segment id +1 if encountering sep+.
    template_ids = []
    sent_flag = True
    for part_id, part in enumerate(template_list):
        if part == 'sep+':
            break
        new_tokens = []
        segment_plus_1_flag = False
        if part in special_token_mapping:
            if part == 'cls' and 'T5' in type(tokenizer).__name__:
                # T5 does not have cls token
                continue
            new_tokens.append(special_token_mapping[part])
            if part == 'sep+':
                segment_plus_1_flag = True
        elif part[:6] == 'label_':
            # Note that label_word_list already has extra space, so do not add more space ahead of it.
            label_id = int(part.split('_')[1])
            label_word = label_word_list[label_id]
            new_tokens.append(label_word)
        elif part[:7] == 'labelx_':
            instance_id = int(part.split('_')[1])
            label_id = support_labels[instance_id]
            label_word = label_word_list[label_id]
            new_tokens.append(label_word)
        elif part[:5] == 'sent_':
            sent_id = int(part.split('_')[1])
            new_tokens += enc(input_text_list[sent_id])
        elif part[:6] == '+sent_':
            # Add space
            sent_id = int(part.split('_')[1])
            new_tokens += enc(' ' + input_text_list[sent_id])
        elif part[:6] == 'sent-_':
            # Delete the last token
            sent_id = int(part.split('_')[1])
            new_tokens += enc(input_text_list[sent_id][:-1])
        elif part[:6] == 'sentl_':
            # Lower case the first token
            sent_id = int(part.split('_')[1])
            text = input_text_list[sent_id]
            text = text[:1].lower() + text[1:]
            new_tokens += enc(text)
        elif part[:7] == '+sentl_':
            # Lower case the first token and add space
            sent_id = int(part.split('_')[1])
            text = input_text_list[sent_id]
            text = text[:1].lower() + text[1:]
            new_tokens += enc(' ' + text)
        elif part[:7] == 'sentl-_':
            # Lower case the first token and discard the last token
            sent_id = int(part.split('_')[1])
            text = input_text_list[sent_id]
            text = text[:1].lower() + text[1:]
            new_tokens += enc(text[:-1])
        elif part[:6] == 'sentu_':
            # Upper case the first token
            sent_id = int(part.split('_')[1])
            text = input_text_list[sent_id]
            text = text[:1].upper() + text[1:]
            new_tokens += enc(text)
        elif part[:7] == '+sentu_':
            # Upper case the first token and add space
            sent_id = int(part.split('_')[1])
            text = input_text_list[sent_id]
            text = text[:1].upper() + text[1:]
            new_tokens += enc(' ' + text)
        else:
            # Just natural language prompt
            part = part.replace('_', ' ')
            # handle special case when T5 tokenizer might add an extra space
            if len(part) == 1:
                new_tokens.append(tokenizer._convert_token_to_id(part))
            else:
                new_tokens += enc(part)

        if part[:4] == 'sent' or part[1:5] == 'sent':
            # If this part is the sentence, limit the sentence length
            sent_id = int(part.split('_')[1])
        if part_id > 0 and template_list[part_id - 1][:4] == 'sent':
            sent_flag = False
        if not sent_flag:
            template_ids += new_tokens
            template_attention_mask += [1 for i in range(len(new_tokens))]
            template_token_type_ids += [segment_id for i in range(len(new_tokens))]
        else:
            input_ids += new_tokens
            attention_mask += [1 for i in range(len(new_tokens))]
            token_type_ids += [segment_id for i in range(len(new_tokens))]

        if segment_plus_1_flag:
            segment_id += 1

    relation_sent = None
    abs_matrix = None
    if position_dataset and isinstance(relation,list):
        relation_sent = []
        for i in range(len(relation)):
            if i == pooling_sent_limit:
                break
            if len(relation[i]) < pooling_sent_limit:
                relation_sent.append(relation[i] + [0] * (pooling_sent_limit - len(relation[i])))
                assert isinstance(relation_sent[-1][0], int)
            if len(relation[i]) > pooling_sent_limit:
                relation_sent.append(relation[i][:pooling_sent_limit])
        while len(relation_sent) < pooling_sent_limit:
            relation_sent.append([0] * pooling_sent_limit)
        assert len(relation_sent) == pooling_sent_limit

        from itertools import groupby
        result = [list(g) for k, g in groupby(input_ids, lambda x: x == 2) if not k]
        assert len(result) == len(abs_relation)

        abs_matrix = []
        for i, item in enumerate(abs_relation):
            if i == 0:
                abs_matrix.extend([item+1] * len(result[i]))
            else:
                item_index = item +1
                if item_index > 127:
                    item_index = 127
                abs_matrix.extend([item_index] * (len(result[i]) + 1))
        assert len(input_ids) == len(abs_matrix)

        while len(abs_matrix) < context_max_length +template_max_length:
            abs_matrix.append(0)
        if len(abs_matrix) > context_max_length + template_max_length:
            abs_matrix = abs_matrix[:(context_max_length + template_max_length)]
    while len(input_ids) < context_max_length:
        input_ids.append(tokenizer.pad_token_id)
        attention_mask.append(0)
        token_type_ids.append(0)

    while len(template_ids) < template_max_length:
        template_ids.append(tokenizer.pad_token_id)
        template_attention_mask.append(0)
        template_token_type_ids.append(0)

    # Truncate
    if len(input_ids) > context_max_length:
        if truncate_head:
            input_ids = input_ids[-context_max_length:]
            attention_mask = attention_mask[-context_max_length:]
            token_type_ids = token_type_ids[-context_max_length:]
        else:
            # Default is to truncate the tail
            input_ids = input_ids[:context_max_length]
            attention_mask = attention_mask[:context_max_length]
            token_type_ids = token_type_ids[:context_max_length]


    # Find mask token
    mask_pos = [len(input_ids) + template_ids.index(tokenizer.mask_token_id)]
    # Make sure that the masked position is inside the max_length
    if 2 in input_ids:
        comment_pos = input_ids.index(2)
    else:
        comment_pos = len(input_ids) - 1
    assert mask_pos[0] < context_max_length + template_max_length

    result = {'input_ids': input_ids,
              'template_input_ids': template_ids,
              'attention_mask': attention_mask,
              'template_attention_mask': template_attention_mask,
              # 'relation': final_relation,
              'relation_sent': relation_sent,
              'abs_relation': abs_matrix
              }
    if 'BERT' in type(tokenizer).__name__:
        # Only provide token type ids for BERT
        result['token_type_ids'] = token_type_ids


    result['mask_pos'] = mask_pos
    result['comment_pos'] = comment_pos

    return result


class FewShotDataset(torch.utils.data.Dataset):
    """Few-shot dataset."""

    def __init__(self, args, tokenizer, cache_dir=None, mode="train", use_demo=False, false_flag=False):
        self.args = args
        self.task_name = args.task_name
        self.mode = mode
        # self.dataset_type = args.train_mode
        assert self.mode in ["train", "dev", "test"]
        # assert self.dataset_type in ["base", "time", "tree"]
        self.processor = TextClassificationProcessor(task_name="twi")
        self.tokenizer = tokenizer

        self.false_flag = false_flag
        # If not using demonstrations, use use_demo=True
        self.use_demo = use_demo
        if self.use_demo:
            logger.info("Use demonstrations")
        # assert mode in ["train", "dev", "test"]
        # assert args.train_mode in ["train", "train_order", "train_time", "train_tree"]
        # Get label list and (for prompt) label word list
        self.label_list = self.processor.get_labels()
        self.num_labels = len(self.label_list)
        if args.prompt:
            assert args.mapping is not None
            self.label_to_word = eval(args.mapping)

            for key in self.label_to_word:
                # For RoBERTa/BART/T5, tokenization also considers space, so we use space+word as label words.
                if self.label_to_word[key][0] not in ['<', '[', '.', ',']:
                    # Make sure space+word is in the vocabulary
                    assert len(tokenizer.tokenize(' ' + self.label_to_word[key])) == 1
                    self.label_to_word[key] = tokenizer._convert_token_to_id(
                        tokenizer.tokenize(' ' + self.label_to_word[key])[0])
                else:
                    self.label_to_word[key] = tokenizer._convert_token_to_id(self.label_to_word[key])
                logger.info(
                    "Label {} to word {} ({})".format(key, tokenizer._convert_id_to_token(self.label_to_word[key]),
                                                      self.label_to_word[key]))

            if len(self.label_list) > 1:
                self.label_word_list = [self.label_to_word[str(label)] for label in self.label_list]
            else:
                # Regression task
                # '0' represents low polarity and '1' represents high polarity.
                self.label_word_list = [self.label_to_word[label] for label in ['0', '1']]
        else:
            self.label_to_word = None
            self.label_word_list = None

        # Multiple sampling: when using demonstrations, we sample different combinations of demonstrations during 
        # inference and aggregate the results by averaging the logits. The number of different samples is num_sample.
        # if ("train" in mode) or not self.use_demo:
        # We do not do multiple sampling when not using demonstrations or when it's the training mode
        self.num_sample = 1

        # Load cache
        # Cache name distinguishes mode, task name, tokenizer, and length. So if you change anything beyond these elements, make sure to clear your cache.
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else args.data_dir,
            "cached_{}_{}_{}_{}_{}".format(
                self.mode,
                tokenizer.__class__.__name__,
                str(args.max_seq_length),
                args.task_name,
                self.false_flag
            ),
        )

        logger.info(f"Creating/loading examples from dataset file at {args.data_dir}")

        lock_path = cached_features_file + ".lock"

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            start = time.time()
            self.support_examples, self.query_examples = torch.load(cached_features_file)
            logger.info(
                f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
            )
        else:
            logger.info(f"Creating features from dataset file at {args.data_dir}")

            # The support examples are sourced from the training set.
            if not self.false_flag:
                self.support_examples = self.processor.get_train_examples(args.data_dir)
            else:
                self.support_examples = self.processor.get_train_false_examples(args.data_dir)

            if mode == "dev":
                self.query_examples = self.processor.get_dev_examples(args.data_dir)
            elif mode == "test":
                self.query_examples = self.processor.get_test_examples(args.data_dir)
            else:
                self.query_examples = self.support_examples
            self.support_examples = self.query_examples
            start = time.time()
            torch.save([self.support_examples, self.query_examples], cached_features_file)
            # ^ This seems to take a lot of time so I want to investigate why and how we can improve.
            logger.info(
                "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
            )

        # Size is expanded by num_sample
        self.size = len(self.query_examples) * self.num_sample

        # Prepare examples (especially for using demonstrations)
        support_indices = list(range(len(self.support_examples)))
        self.example_idx = []
        for sample_idx in range(self.num_sample):
            for query_idx in range(len(self.query_examples)):
                # If training, exclude the current example. Else keep all.
                if self.use_demo and args.demo_filter:
                    # Demonstration filtering
                    candidate = [support_idx for support_idx in support_indices
                                 if support_idx != query_idx or mode != "train"]
                    sim_score = []
                    for support_idx in candidate:
                        sim_score.append((support_idx, util.pytorch_cos_sim(self.support_emb[support_idx],
                                                                            self.query_emb[query_idx])))
                    sim_score.sort(key=lambda x: x[1], reverse=True)
                    if self.num_labels == 1:
                        # Regression task
                        limit_each_label = int(len(sim_score) // 2 * args.demo_filter_rate)
                        count_each_label = {'0': 0, '1': 0}
                        context_indices = []

                        if args.debug_mode:
                            print("Query %s: %s" % (
                            self.query_examples[query_idx].label, self.query_examples[query_idx].text_a))  # debug
                        for support_idx, score in sim_score:
                            if count_each_label[
                                '0' if float(self.support_examples[support_idx].label) <= median_mapping[
                                    args.task_name] else '1'] < limit_each_label:
                                count_each_label[
                                    '0' if float(self.support_examples[support_idx].label) <= median_mapping[
                                        args.task_name] else '1'] += 1
                                context_indices.append(support_idx)
                                if args.debug_mode:
                                    print("    %.4f %s | %s" % (score, self.support_examples[support_idx].label,
                                                                self.support_examples[support_idx].text_a))  # debug
                    else:
                        limit_each_label = int(len(sim_score) // self.num_labels * args.demo_filter_rate)
                        count_each_label = {label: 0 for label in self.label_list}
                        context_indices = []

                        if args.debug_mode:
                            print("Query %s: %s" % (
                            self.query_examples[query_idx].label, self.query_examples[query_idx].text_a))  # debug
                        for support_idx, score in sim_score:
                            if count_each_label[self.support_examples[support_idx].label] < limit_each_label:
                                count_each_label[self.support_examples[support_idx].label] += 1
                                context_indices.append(support_idx)
                                if args.debug_mode:
                                    print("    %.4f %s | %s" % (score, self.support_examples[support_idx].label,
                                                                self.support_examples[support_idx].text_a))  # debug
                else:
                    # Using demonstrations without filtering
                    context_indices = [support_idx for support_idx in support_indices
                                       if support_idx != query_idx or mode in ["dev", "test"]]

                # We'll subsample context_indices further later.
                self.example_idx.append((query_idx, context_indices, sample_idx))

        # If it is not training, we pre-process the data; otherwise, we process the data online.
        if mode in ["dev", "test"]:
            self.features = []
            _ = 0
            for query_idx, context_indices, bootstrap_idx in self.example_idx:
                # The input (query) example
                example = self.query_examples[query_idx]
                # The demonstrations
                supports = None
                #supports = self.select_context([self.support_examples[i] for i in context_indices])

                if args.template_list is not None:
                    template = args.template_list[sample_idx % len(args.template_list)]  # Use template in order
                else:
                    template = args.template

                self.features.append(self.convert_fn(
                    example=example,
                    supports=supports,
                    use_demo=self.use_demo,
                    label_list=self.label_list,
                    prompt=args.prompt,
                    template=template,
                    label_word_list=self.label_word_list,
                    verbose=True if _ == 0 else False,
                    mode=self.mode
                ))

                _ += 1
        else:
            self.features = None

    def select_context(self, context_examples):
        """
        Select demonstrations from provided examples.
        """
        max_demo_per_label = 1
        counts = {k: 0 for k in self.label_list}
        if len(self.label_list) == 1:
            # Regression
            counts = {'0': 0, '1': 0}
        selection = []

        if self.args.gpt3_in_context_head or self.args.gpt3_in_context_tail:
            # For GPT-3's in-context learning, we sample gpt3_in_context_num demonstrations randomly. 
            order = np.random.permutation(len(context_examples))
            for i in range(min(self.args.gpt3_in_context_num, len(order))):
                selection.append(context_examples[order[i]])
        else:
            # Our sampling strategy
            order = np.random.permutation(len(context_examples))

            for i in order:
                label = context_examples[i].label
                if len(self.label_list) == 1:
                    # Regression
                    label = '0' if float(label) <= median_mapping[self.args.task_name] else '1'
                if counts[label] < max_demo_per_label:
                    selection.append(context_examples[i])
                    counts[label] += 1
                if sum(counts.values()) == len(counts) * max_demo_per_label:
                    break

            assert len(selection) > 0

        return selection

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        if self.features is None:
            query_idx, context_indices, bootstrap_idx = self.example_idx[i]
            # The input (query) example
            example = self.query_examples[query_idx]
            # The demonstrations
            supports = None
            # supports = self.select_context([self.support_examples[i] for i in context_indices])

            if self.args.template_list is not None:
                template = self.args.template_list[sample_idx % len(self.args.template_list)]
            else:
                template = self.args.template

            features = self.convert_fn(
                example=example,
                supports=supports,
                use_demo=self.use_demo,
                label_list=self.label_list,
                prompt=self.args.prompt,
                template=template,
                label_word_list=self.label_word_list,
                verbose=False,
                mode=self.mode
            )
        else:
            features = self.features[i]

        return features

    def get_labels(self):
        return self.label_list

    def convert_fn(
            self,
            example,
            supports,
            use_demo=False,
            label_list=None,
            prompt=False,
            template=None,
            label_word_list=None,
            verbose=False,
            mode='train'
    ):
        """
        Returns a list of processed "InputFeatures".
        """
        context_max_length = self.args.context_max_length
        template_max_length = self.args.template_max_length
        # Prepare labels
        label_map = {label: i for i, label in enumerate(label_list)}  # Mapping the label names to label ids
        if len(label_list) == 1:
            # Regression
            label_map = {'0': 0, '1': 1}

        # Get example's label id (for training/inference)
        if example.label is None:
            example_label = None
        elif len(label_list) == 1:
            # Regerssion
            example_label = float(example.label)
        else:
            example_label = label_map[example.label]

        # Prepare other features
            # No using demonstrations
        inputs = tokenize_multipart_input(
            input_text_list=input_example_to_tuple(example),
            context_max_length=context_max_length,
            template_max_length=template_max_length,
            tokenizer=self.tokenizer,
            position_dataset=True,
            template=template,
            label_word_list=label_word_list,
            pooling_sent_limit=128,
        )
        features = OurInputFeatures(**inputs, label=example_label)


        if verbose:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("features: %s" % features)
            logger.info("text: %s" % self.tokenizer.decode(features.input_ids))

        return features
