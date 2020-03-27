#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="3"
export BERT_BASE_DIR=models/roberta-magi-finetune
export OUTPUT_DIR=models/magi-finetune-annote-finetune
export DATA_DIR=datasets/annote_spo_data

python predicate_extraction.py\
  --do_train=false \
  --do_eval=true \
  --do_predict=false \
  --data_dir=$DATA_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/config.json \
  --init_checkpoint=$BERT_BASE_DIR/model.ckpt-61557 \
  --max_seq_length=256 \
  --train_batch_size=16 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=$OUTPUT_DIR
