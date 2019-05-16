from __future__ import absolute_import, division, print_function

import argparse
import logging
import random
import numpy as np
import torch
import os

from torch.utils.data import (DataLoader,
                             RandomSampler,
                             SequentialSampler,
                             TensorDataset)


import tasks
import bert_models
import tokenization
from optimization import BertAdam


log = logging.getLogger(__file__)
log.setLevel(logging.INFO)


MY_MODEL_NAME = "pytorch_model.bin"
REGRESSION = "regression"
CLASSIFICATION = "classification"


def setupArgs():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--pretrain_dir",
                        default=None,
                        type=str,
                        help="The based pretrained directory where the model be loaded from.")
    parser.add_argument("--vocab_fpath",
                        default=None,
                        type=str,
                        help="the path for the vocabulary file.")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--emb_size',
                        type=int,
                        default=1024,
                        help="vector embedding size for sentences")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    args = parser.parse_args()
    return args


def setupRandomSeed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    return


def checkArgs(args):
    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        if args.do_train:
            msg = "Output directory ({}) already exists and is not empty."
            raise ValueError(msg.format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    return


def getVocabularyPath(args):
    fdir = args.vocab_fpath
    if fdir is None:
        if args.pretrain_dir is None:
            return None

        fdir = os.path.join(args.pretrain_dir, "vocab.txt")
        return fdir
    else:
        return fdir


def getTokenizer(args):
    fpath = getVocabularyPath(args)
    tokenizer = tokenization.getTokenizer(fpath=fpath)
    return tokenizer


def computeSteps(example_num, args):
    a = args.train_batch_size * args.gradient_accumulation_steps
    b = int(example_num / a)
    steps = b * args.num_train_epochs
    return steps


def getModelType(args):
    task_name = args.task_name.lower()
    if task_name not in tasks.processors:
        raise ValueError("Task not found: %s" % (task_name))

    model_type = tasks.output_modes[task_name]
    log.info("model_type: %s" % (model_type))
    return model_type


def getTensorData(features, output_model=CLASSIFICATION):
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    dtype = torch.long
    if output_model == REGRESSION:
        dtype = torch.float
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=dtype)
    log.info("todel: shape=%s, %s" % (all_label_ids.shape, all_label_ids.dtype))

    tensor_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_label_ids)
    return tensor_data, all_label_ids


def getDataProcessor(args):
    #1. get data processor
    task_name = args.task_name.lower()
    if task_name not in tasks.processors:
        raise ValueError("Task not found: %s" % (task_name))
    processor = tasks.processors[task_name]()

    #2. get output_mode: classification Vs. regression
    output_mode = tasks.output_modes[task_name]
    if output_mode not in {CLASSIFICATION, REGRESSION}:
        log.fatal("un-supported output mode: %s" % (output_mode))
        raise ValueError("un-supported output mode: %s" % (output_mode))
    return processor, output_mode


def getDevice(args):
    use_cuda = torch.cuda.is_available() and (not args.no_cuda)

    if use_cuda:
        index = args.local_rank
        if index < 0:
            index = 0
        device = torch.device("cuda", index)
    else:
        device = torch.device("cpu")

    log.info("Using device: %s" % (device))
    return device


def getOptimizer(args, model, train_steps):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    p0 = [p for n, p in param_optimizer if any(nd in n for nd in no_decay)]
    p1 = [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
    optimizer_grouped_parameters = [
        {'params': p1, 'weight_decay': 0.01},
        {'params': p0, 'weight_decay': 0.0}
    ]

    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=train_steps)
    return optimizer


def saveModel(args, model, prefix=None):
    # Only save the model it-self
    model_to_save = model
    if hasattr(model, 'module'):
        model_to_save = model.module

    log.info("saving model to: %s" % (args.output_dir))
    fname = MY_MODEL_NAME
    if prefix:
        fname = "%s.%s" % (prefix, fname)
    output_model_file = os.path.join(args.output_dir, fname)
    torch.save(model_to_save.state_dict(), output_model_file)
    output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
    with open(output_config_file, 'w') as f:
        f.write(model_to_save.config.to_json_string())

    log.info("model is saved: %s" % (args.output_dir))
    return


def getModelConfig(fdir):
    fconfig = os.path.join(fdir, "bert_config.json")
    fweight = os.path.join(fdir, "pytorch_model.bin")

    config = bert_models.BertConfig(fconfig)
    log.info("bert.config:\n%s" % (config))
    return config, fweight


def saveEvalResult(args, result):
    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        log.info("***** Eval results *****")
        for key in sorted(result.keys()):
            log.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return


def printModelSize(model):
    total = 0
    for name, param in model.named_parameters():
        count = 1
        shape = tuple(param.size())
        for x in shape:
            count *= x
        total += count
        log.info("%s, %s, %d" % (name, shape, count))

    name = type(model).__name__
    log.info("model[%s] has %d parameters" % (name, total))
    return


def doLoadModel(model, fpath):
    log.info("begin to load model weights from: %s" % (fpath))
    state_dict = torch.load(fpath, map_location='cpu' if not torch.cuda.is_available() else None)
    # Load from a PyTorch state_dict
    old_keys = []
    new_keys = []
    for key in state_dict.keys():
        new_key = None
        if 'gamma' in key:
            new_key = key.replace('gamma', 'weight')
        if 'beta' in key:
            new_key = key.replace('beta', 'bias')
        if new_key:
            old_keys.append(key)
            new_keys.append(new_key)
    for old_key, new_key in zip(old_keys, new_keys):
        state_dict[new_key] = state_dict.pop(old_key)

    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')
    start_prefix = ''
    if not hasattr(model, 'bert') and any(s.startswith('bert.') for s in state_dict.keys()):
        start_prefix = 'bert.'
    log.info("prefix is : %s" % (start_prefix))
    load(model, prefix=start_prefix)
    if len(missing_keys) > 0:
        log.info("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        log.info("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(error_msgs) > 0:
        raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                           model.__class__.__name__, "\n\t".join(error_msgs)))
    return model


class Counter:
    """count the loss during training"""
    def __init__(self, total_steps, name):
        self.name = name
        self.total_steps = total_steps
        self.nr_steps = 0
        self.reset()
        return

    def update(self, loss, total):
        self.loss += loss
        self.steps += 1
        self.nr_steps += 1
        self.total_num += total
        return

    def reset(self):
        self.loss = 0.0
        self.steps = 1e-3
        self.total_num = 1e-3
        return

    def __str__(self):
        avg = self.loss / self.steps
        stepinfo = "%d/%d" % (int(self.nr_steps), int(self.total_steps))
        msg = "%s %s avg loss: %.4f" % (self.name, stepinfo, avg)
        return msg

    def info(self):
        avg = self.loss / self.steps
        msg = "%s %d steps, %d examples, avg loss: %.4f" % (self.name,
                int(self.steps), int(self.total_num), avg)
        return msg