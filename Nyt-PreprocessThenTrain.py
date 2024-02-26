import collections
import logging
import os
import sys
import csv
import glob
from dataclasses import dataclass, field
from typing import Optional

import transformers
import numpy as np
import torch
# from torch.utils.tensorboard import SummaryWriter

from transformers import (BertTokenizerFast, BertModel, Trainer,
                          TrainingArguments, BertConfig, BertLMHeadModel)

from transformers.hf_argparser import HfArgumentParser
from transformers import EvalPrediction, set_seed

from dataprocess.data_processor import UniRelDataProcessor
from dataprocess.dataset import UniRelDataset, UniRelSpanDataset

from model.model_transformers import  UniRelModel
from dataprocess.data_extractor import *
from dataprocess.data_metric import *

max_seq_length = 150

added_token = [f"[unused{i}]" for i in range(1, 17)]
# If use unused to do ablation, should uncomment this
# added_token = [f"[unused{i}]" for i in range(1, 399)]
tokenizer = BertTokenizerFast.from_pretrained(
    "bert-base-cased",
    additional_special_tokens=added_token,
    do_basic_tokenize=False)

# Data pre-processing
DataProcessorType = UniRelDataProcessor
data_processor = DataProcessorType(root="dataset",
                                   tokenizer=tokenizer,
                                   dataset_name="nyt")

train_samples = data_processor.get_train_sample(
        token_len=max_seq_length, data_nums=-1)

print("finish getting train samples")

# metrics
metric_type = unirel_span_metric
predict_metric_type = unirel_span_metric


DatasetType = UniRelSpanDataset(max_length=max_seq_length+2)
ExtractType = unirel_span_extractor  # Extractor triples from the modeled Attention matrix
ModelType = UniRelModel
PredictModelType = UniRelModel
training_args.label_names = LableNamesDict[run_args.test_data_type]