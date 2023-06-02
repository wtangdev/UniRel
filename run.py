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

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DataProcessorDict = {
    "nyt_all_sa": UniRelDataProcessor,
    "unirel_span": UniRelDataProcessor
}

DatasetDict = {
    "nyt_all_sa": UniRelDataset,
    "unirel_span": UniRelSpanDataset 
}

ModelDict = {
    "nyt_all_sa": UniRelModel,
    "unirel_span": UniRelModel
}

PredictModelDict = {
    "nyt_all_sa": UniRelModel,
    "unirel_span": UniRelModel
}

DataMetricDict = {
    "nyt_all_sa": unirel_metric,
    "unirel_span": unirel_span_metric
}

PredictDataMetricDict = {
    "nyt_all_sa": unirel_metric,
    "unirel_span": unirel_span_metric

}

DataExtractDict = {
    "nyt_all_sa": unirel_extractor,
    "unirel_span": unirel_span_extractor

}

LableNamesDict = {
    "nyt_all_sa": [ "tail_label"],
    "unirel_span":["head_label", "tail_label", "span_label"],
}

InputFeature = collections.namedtuple(
    "InputFeature", ["input_ids", "attention_mask", "token_type_ids", "label"])

logger = transformers.utils.logging.get_logger(__name__)

class MyCallback(transformers.TrainerCallback):
    "A callback that prints a message at the beginning of training"

    def on_epoch_begin(self, args, state, control, **kwargs):
        print("Epoch start")


    def on_epoch_end(self, args, state, control, **kwargs):
        print("Epoch end")



@dataclass
class RunArguments:
    """Arguments pretraining to which model/config/tokenizer we are going to continue training, or train from scratch.
    """
    model_dir: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        })
    config_path: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "The configuration file of initialization parameters."
            "If `model_dir` has been set, will read `model_dir/config.json` instead of this path."
        })
    vocab_path: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "The vocabulary for tokenzation."
            "If `model_dir` has been set, will read `model_dir/vocab.txt` instead of this path."
        })
    dataset_dir: str = field(
        metadata={"help": "Directory where data set stores."}, default=None)
    max_seq_length: Optional[int] = field(
        default=100,
        metadata={
            "help":
            "The maximum total input sequence length after tokenization. Longer sequences"
            "will be truncated. Default to the max input length of the model."
        })
    task_name: str = field(metadata={"help": "Task name"},
                           default=None)
    do_test_all_checkpoints: bool = field(
        default=False,
        metadata={"help": "Whether to test all checkpoints by test_data"})
    test_data_type: str = field(
        metadata={"help": "Which data type to test: nyt_all_sa"},
        default=None)
    train_data_nums: int = field(
        metadata={"help": "How much data to train the model."}, default=-1)
    test_data_nums: int = field(metadata={"help": "How much data to test."},
                                default=-1)
    dataset_name: str = field(
        metadata={"help": "The dataset you want to test"}, default=-1)
    threshold: float = field(
        metadata={"help": "The threhold when do classify prediction"},
        default=-1)
    test_data_path: str = field(
        metadata={"help": "Test specific data"},
        default=None)
    checkpoint_dir : str = field(
        metadata={"help": "Test with specififc trained checkpoint"},
        default=None
    )
    is_additional_att: bool = field(
        metadata={"help": "Use additonal attention layer upon BERT"},
        default=False)
    is_separate_ablation: bool = field(
        metadata={"help": "Seperate encode text and predicate to do ablation study"},
        default=False)


if __name__ == '__main__':
    parser = HfArgumentParser((RunArguments, TrainingArguments))
    if len(sys.argv[1]) == 2 and sys.argv[1].endswith(".json"):
        run_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        run_args, training_args = parser.parse_args_into_dataclasses()

    if (os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir) and training_args.do_train
            and not training_args.overwrite_output_dir):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and not empty."
            "Use --overwrite_output_dir to overcome.")

    set_seed(training_args.seed)

    training_args: TrainingArguments
    run_args: RunArguments

    # Initialize configurations and tokenizer.
    added_token = [f"[unused{i}]" for i in range(1, 17)]
    # If use unused to do ablation, should uncomment this
    # added_token = [f"[unused{i}]" for i in range(1, 399)]
    tokenizer = BertTokenizerFast.from_pretrained(
        "bert-base-cased",
        additional_special_tokens=added_token,
        do_basic_tokenize=False)
    transformers.utils.logging.set_verbosity_info()
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    logger.info("Training parameter %s", training_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        +
        f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    # Initialize Dataset-sensitive class/function
    dataset_name = run_args.dataset_name
    DataProcessorType = DataProcessorDict[run_args.test_data_type]
    metric_type = DataMetricDict[run_args.test_data_type]
    predict_metric_type = PredictDataMetricDict[run_args.test_data_type]
    DatasetType = DatasetDict[run_args.test_data_type]
    ExtractType = DataExtractDict[run_args.test_data_type]
    ModelType = ModelDict[run_args.test_data_type]
    PredictModelType = PredictModelDict[run_args.test_data_type]
    training_args.label_names = LableNamesDict[run_args.test_data_type]

    # Load data
    data_processor = DataProcessorType(root=run_args.dataset_dir,
                                       tokenizer=tokenizer,
                                       dataset_name=run_args.dataset_name)
    train_samples = data_processor.get_train_sample(
        token_len=run_args.max_seq_length, data_nums=run_args.train_data_nums)
    dev_samples = data_processor.get_dev_sample(
        token_len=150, data_nums=run_args.test_data_nums)
    
    # For special experiment wants to test on specific testset
    if run_args.test_data_path is not None:
        test_samples = data_processor.get_specific_test_sample(
            data_path=run_args.test_data_path, token_len=150, data_nums=run_args.test_data_nums)
    else:
        test_samples = data_processor.get_test_sample(
            token_len=150, data_nums=run_args.test_data_nums)

    # Train with fixed sentence length of 100
    train_dataset = DatasetType(
        train_samples,
        data_processor,
        tokenizer,
        mode='train',
        ignore_label=-100,
        model_type='bert',
        ngram_dict=None,
        max_length=run_args.max_seq_length + 2,
        predict=False,
        eval_type="train",
    )
    # 150 is big enough for both NYT and WebNLG testset
    dev_dataset = DatasetType(
        dev_samples,
        data_processor,
        tokenizer,
        mode='dev',
        ignore_label=-100,
        model_type='bert',
        ngram_dict=None,
        max_length=150 + 2,
        predict=True,
        eval_type="eval"
    )
    test_dataset = DatasetType(
        test_samples,
        data_processor,
        tokenizer,
        mode='test',
        ignore_label=-100,
        model_type='bert',
        ngram_dict=None,
        max_length=150 + 2,
        predict=True,
        eval_type="test"
    )

    config = BertConfig.from_pretrained(run_args.model_dir,
                                        finetuning_task=run_args.task_name)
    config.threshold = run_args.threshold
    config.num_labels = data_processor.num_labels
    config.num_rels = data_processor.num_rels
    config.is_additional_att = run_args.is_additional_att
    config.is_separate_ablation = run_args.is_separate_ablation
    config.test_data_type = run_args.test_data_type


    model = ModelType(config=config, model_dir=run_args.model_dir)
    model.resize_token_embeddings(len(tokenizer))

   
    if training_args.do_train:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            compute_metrics=metric_type,
        )
        train_result = trainer.train()
        trainer.save_model(
            output_dir=f"{trainer.args.output_dir}/checkpoint-final/")
        output_train_file = os.path.join(training_args.output_dir,
                                         "train_results.txt")
        # trainer.evaluate()
        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as writer:
                logger.info("***** Train Results *****")
                for key, value in sorted(train_result.metrics.items()):
                    logger.info(f"  {key} = {value}")
                    print(f"{key} = {value}", file=writer)

    results = dict()
    if run_args.do_test_all_checkpoints:
        if run_args.checkpoint_dir is None:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(
                    glob.glob(
                        f"{training_args.output_dir}/checkpoint-*/{transformers.file_utils.WEIGHTS_NAME}",
                        recursive=True)))
        else:
            checkpoints = [run_args.checkpoint_dir] 
        logger.info(f"Test the following checkpoints: {checkpoints}")
        best_f1 = 0
        best_checkpoint = None
        # Find best model on devset
        for checkpoint in checkpoints:
            logger.info(checkpoint)
            print(checkpoint)
            output_dir = os.path.join(training_args.output_dir, checkpoint.split("/")[-1])
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)
            global_step = checkpoint.split("-")[1]
            prefix = checkpoint.split(
                "/")[-1] if checkpoint.find("checkpoint") != -1 else ""
            model = PredictModelType.from_pretrained(checkpoint, config=config)
            trainer = Trainer(model=model,
                              args=training_args,
                              eval_dataset=dev_dataset,
                              callbacks=[MyCallback])

            # eval_res = trainer.evaluate(
            #     eval_dataset=dev_dataset, metric_key_prefix="test")
            # result = {f"{k}_{global_step}": v for k, v in eval_res.items()}
            # results.update(result)
            dev_predictions = trainer.predict(dev_dataset)
            p,r,f1 =  ExtractType(tokenizer, dev_dataset, dev_predictions, output_dir)
            if f1 > best_f1:
                best_f1 = f1
                best_checkpoint = checkpoint

        # Do test
        logger.info(f"Best checkpoint at {best_checkpoint} with f1 = {best_f1}")
        model = PredictModelType.from_pretrained(best_checkpoint, config=config)
        trainer = Trainer(model=model,
                            args=training_args,
                            eval_dataset=dev_dataset,
                            callbacks=[MyCallback])

        test_prediction = trainer.predict(test_dataset)
        output_dir = os.path.join(training_args.output_dir, best_checkpoint.split("/")[-1])
        ExtractType(tokenizer, test_dataset, test_prediction, output_dir)
            
    print("Here I am")