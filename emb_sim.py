from __future__ import absolute_import, division, print_function


import logging
import torch
import numpy as np

from torch.utils.data import (DataLoader,
                             RandomSampler,
                             SequentialSampler,
                             TensorDataset)

# local imports
import bert_models
import tasks
import tools
from tools import (
    REGRESSION,
    CLASSIFICATION
)

import utils

log = logging.getLogger(__file__)
log.setLevel(logging.INFO)


def loadPreTrainedModel(args):
    fdir = args.pretrain_dir
    config, fweight = tools.getModelConfig(fdir)

    emb_size = args.emb_size
    model = bert_models.BertPairSim(config, emb_size)
    tools.doLoadModel(model, fweight)
    return model


def loadPairSimModel(fdir, emb_size):
    log.info("begin to load BertPairSim model from: %s" % (fdir))
    config, fweight = tools.getModelConfig(fdir)
    model = bert_models.BertPairSim(config, emb_size)

    state_dict = torch.load(fweight)
    model.load_state_dict(state_dict, strict=True)
    log.info("finished loading model from: %s" % (fdir))
    return model


class InputPairFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids1, input_mask1, input_ids2, input_mask2, label_id):
        self.input_ids1 = input_ids1
        self.input_mask1 = input_mask1
        self.input_ids2 = input_ids2
        self.input_mask2 = input_mask2
        self.label_id = label_id


def doTruncate(tokens, max_seq_length):
    if len(tokens) > max_seq_length - 2:
        tokens = tokens[:(max_seq_length - 2)]
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    return tokens


def doPadding(input_ids, max_seq_length):
    input_mask = [1] * len(input_ids)
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    return input_ids, input_mask


def convertLabel(label):
    # return (float(label) - 2.5) / 2.5
    return float(label)


def restoreLabel(label):
    # return (label * 2.5 + 2.5)
    return label


def convertPairFeatures(examples, label_list, max_seq_length,
                        tokenizer, output_mode):
    """Loads a data file into a list of `InputBatch`s.
    exmaples: are tasks.InputExamples;
    """

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            log.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = tokenizer.tokenize(example.text_b)

        # Account for [CLS] and [SEP] with "- 2"
        tokens_a = doTruncate(tokens_a, max_seq_length)
        tokens_b = doTruncate(tokens_b, max_seq_length)

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.

        input_ids1 = tokenizer.convert_tokens_to_ids(tokens_a)
        input_ids2 = tokenizer.convert_tokens_to_ids(tokens_b)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_ids1, input_mask1 = doPadding(input_ids1, max_seq_length)
        input_ids2, input_mask2 = doPadding(input_ids2, max_seq_length)

        assert len(input_ids1) == max_seq_length
        assert len(input_mask1) == max_seq_length
        assert len(input_ids2) == max_seq_length
        assert len(input_mask2) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = convertLabel(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            log.debug("*** Example ***")
            log.debug("guid: %s" % (example.guid))
            log.debug("tokens_a: %s" % " ".join([str(x) for x in tokens_a]))
            log.debug("input_ids1: %s" % " ".join([str(x) for x in input_ids1]))
            log.debug("input_mask1: %s" % " ".join([str(x) for x in input_mask1]))
            log.debug("tokens_b: %s" % " ".join([str(x) for x in tokens_b]))
            log.debug("input_ids2: %s" % " ".join([str(x) for x in input_ids2]))
            log.debug("input_mask2: %s" % " ".join([str(x) for x in input_mask2]))
            log.debug("label: %s (id = %.3f)" % (example.label, label_id))

        features.append(InputPairFeatures(input_ids1=input_ids1,
                              input_mask1=input_mask1,
                              input_ids2=input_ids2,
                              input_mask2=input_mask2,
                              label_id=label_id))
    return features


def getTensorData(features, output_model=CLASSIFICATION):
    input_ids1 = torch.tensor([f.input_ids1 for f in features], dtype=torch.long)
    input_mask1 = torch.tensor([f.input_mask1 for f in features], dtype=torch.long)
    input_ids2 = torch.tensor([f.input_ids2 for f in features], dtype=torch.long)
    input_mask2 = torch.tensor([f.input_mask2 for f in features], dtype=torch.long)
    dtype = torch.long
    if output_model == REGRESSION:
        dtype = torch.float
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=dtype)
    log.info("todel: shape=%s, %s" % (all_label_ids.shape, all_label_ids.dtype))

    tensor_data = TensorDataset(input_ids1, input_mask1, input_ids2, input_mask2,
                                all_label_ids)
    return tensor_data, all_label_ids


def trainEpoch(model, optimizer, loss_fn, device, data_loader,
                index, train_steps, prefix):
    # loss_fn = torch.nn.CosineEmbeddingLoss().to(device)
    counter2 = tools.Counter(train_steps, "Epoch")
    counter3 = tools.Counter(train_steps, "batch")

    model.train()
    total_loss = 0.0
    total_num = 0
    for batch in data_loader:
        batch = tuple(t.to(device) for t in batch)
        input_ids1, input_mask1, input_ids2, input_mask2, label_ids = batch
        logits1 = model(input_ids1, input_mask1)
        logits2 = model(input_ids2, input_mask2)
        loss = loss_fn(logits1, logits2, label_ids)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        vloss = loss.item()
        num = label_ids.size(0)
        counter2.update(vloss, num)
        counter3.update(vloss, num)
        total_loss += vloss
        total_num += num

        index += 1
        if index % 50 == 0:
            log.info("%s %s" % (prefix, counter3))
            counter3.reset()

    log.info("%s %s" % (prefix, counter2))
    return total_loss, total_num, index


def getLossFunction(device):
    cos_fn = torch.nn.CosineSimilarity(dim=1, eps=1e-6).to(device)
    l1_loss = torch.nn.L1Loss(reduction='mean')

    def cosine_loss(v1, v2, label):
        sim_t = cos_fn(v1, v2)
        d = l1_loss(sim_t, label)
        return d
    return cosine_loss


def getTrainData(args):
    # 1. get data
    processor, output_mode = tools.getDataProcessor(args)
    train_examples = processor.get_train_examples(args.data_dir)

    # 2. get lable num
    label_list = processor.get_labels()
    num_labels = len(label_list)

    tokenizer = tools.getTokenizer(args)
    train_features = convertPairFeatures(train_examples, label_list,
        args.max_seq_length, tokenizer, output_mode)

    tensor_data, _ = getTensorData(train_features, output_mode)
    train_sampler = RandomSampler(tensor_data)
    train_dataloader = DataLoader(tensor_data, sampler=train_sampler,
                batch_size=args.train_batch_size)

    # 5. other info
    # compute the total steps
    steps = tools.computeSteps(len(train_examples), args)
    log.info("******** Training Data Information ***********")
    log.info("data_dir: %s" % (args.data_dir))
    log.info("\tNum Examples = %d" % (len(train_examples)))
    log.info("\tBatch Size = %d" % (args.train_batch_size))
    log.info("\tLearning Rate = %s" % (args.learning_rate))
    log.info("\tTotal Steps = %d" % (steps))
    log.info("\tNum labels = %d" % (num_labels))

    return train_dataloader, num_labels, steps


def getEvalData(args):
    #1. get data
    processor, output_mode = tools.getDataProcessor(args)
    eval_examples = processor.get_dev_examples(args.data_dir)

    # 2. get label type num
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # 3. convert tokens to IDs
    tokenizer = tools.getTokenizer(args)
    eval_features = convertPairFeatures(
        eval_examples, label_list, args.max_seq_length, tokenizer, output_mode)

    # 4. convert tokens to tensors
    tensor_data, all_label_ids = getTensorData(eval_features, output_mode)
    eval_sampler = SequentialSampler(tensor_data)
    dataloader = DataLoader(tensor_data, sampler=eval_sampler,
                batch_size=args.eval_batch_size)

    # 5. other info
    # compute the total steps
    steps = int(len(eval_examples) / args.eval_batch_size)
    log.info("******** Testing Data Information ***********")
    log.info("data_dir: %s" % (args.data_dir))
    log.info("\tNum Examples = %d" % (len(eval_examples)))
    log.info("\tBatch Size = %d" % (args.eval_batch_size))
    log.info("\tTotal Steps = %d" % (steps))
    log.info("\tNum labels = %d" % (num_labels))

    return dataloader, num_labels, all_label_ids


def trainIt(args):
    tools.setupRandomSeed(args)
    device = tools.getDevice(args)
    loss_fn = getLossFunction(device)

    dataloader, _, train_steps = getTrainData(args)
    model = loadPreTrainedModel(args)
    model.to(device)

    optimizer = tools.getOptimizer(args, model, train_steps + 1.0)
    log.info("Begin to train the model.")
    model.train()
    # model.eval() # to disable Dropout and BN
    epochs = int(args.num_train_epochs)
    counter1 = tools.Counter(train_steps, "Over-all")
    i = 0
    for e in range(epochs):
        prefix = "[%d/%d]" % (e, epochs)
        log.info("xxxxxxx begin epoch %s xxxxxxxxxx" % (prefix))
        vloss, num, i = trainEpoch(model, optimizer, loss_fn, device,
                                dataloader, i, train_steps, prefix)
        counter1.update(vloss, num)
        if e < epochs - 1:
            tools.saveModel(args, model, "epoch%d" % (e))
        if e > 2 and e < epochs - 1:
            evalIt(args, model)
        log.info("xxxxxxx End epoch %s xxxxxxxxxx" % (prefix))
    log.info("Finished training: %s" % (counter1.info()))
    tools.saveModel(args, model)
    return model


def evalIt(args, model):
    is_classification = (tools.getModelType(args) == CLASSIFICATION)

    # 1. get model and data
    device = tools.getDevice(args)
    if model is None:
        model = loadPairSimModel(args.output_dir, args.emb_size)
    model.eval()
    model.to(device)
    dataloader, _, all_label_ids = getEvalData(args)

    loss_fn = getLossFunction(device)
    cos_fn = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    # 2. do prediction
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    total = int(all_label_ids.size(0) / args.eval_batch_size)

    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        input_ids1, input_mask1, input_ids2, input_mask2, label_ids = batch

        with torch.no_grad():
            logits1 = model(input_ids1, input_mask1)
            logits2 = model(input_ids2, input_mask2)
            sim_t = cos_fn(logits1, logits2)
            loss_t = loss_fn(logits1, logits2, label_ids)
        eval_loss += loss_t.item()
        nb_eval_steps += 1
        if nb_eval_steps % 50 == 0:
            log.info("progress: %d/%d" % (nb_eval_steps, total))
        if nb_eval_steps % 100 == 0:
            d = sim_t - loss_t
            log.info("d.shape=%s, d=%s" % (d.shape, d))

        part = sim_t.detach().cpu().numpy()
        if preds is None:
            preds = part
        else:
            preds = np.append(preds, part, axis=0)

    # 3. compute the metrics
    eval_loss = eval_loss / nb_eval_steps
    log.info("finish evaluating the model: eval_loss=%.5f" % (eval_loss))
    if is_classification:
        preds = np.argmax(preds[0], axis=1)
    else:
        preds = np.squeeze(preds)

    task_name = args.task_name.lower()
    preds = restoreLabel(preds)
    labels = restoreLabel(all_label_ids.numpy())
    log.info("pred.shape=%s, label.shape=%s" % (preds.shape, labels.shape, ))
    log.info("pred=%s\nlabel=%s" % (preds[0:10], labels[0:10]))
    result = tasks.compute_metrics(task_name, preds, labels)
    result['eval_loss'] = eval_loss

    # 4. save the evaluation result
    tools.saveEvalResult(args, result)
    return


def main():
    args = tools.setupArgs()
    tools.checkArgs(args)

    model = None
    if args.do_train:
        log.info("begin to train model for task: %s" % (args.task_name))
        model = trainIt(args)

    if args.do_eval:
        log.info("begin to evaluate model for task: %s" % (args.task_name))
        evalIt(args, model)

    return


if __name__ == "__main__":
    utils.setupLog()
    main()
