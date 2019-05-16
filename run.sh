#!/bin/bash

#1. prepare data 
data_dir=~/dev/data/glue
task_name=STS-B

pretrain_dir=~/dev/nlp/bert/models/bert-base-uncased
vocab_fpath=${pretrain_dir}/vocab.txt

output_dir=./finetune/
options="--data_dir $data_dir/$task_name"
options="$options --task_name $task_name "

options="$options --bert_model bert-base-uncased "
options="$options --do_train "
options="$options --do_eval "
options="$options --pretrain_dir $pretrain_dir "
options="$options --vocab_fpath $vocab_fpath "
options="$options --output_dir $output_dir/$task_name/sim20190515x2"
options="$options --do_lower_case"
options="$options --local_rank 5"
options="$options --learning_rate 3e-5"
options="$options --train_batch_size 32"
options="$options --max_seq_length 64"
options="$options --num_train_epochs 20.0"
options="$options --emb_size 1024"

#2. fine-tune it
python ./emb_sim.py $options