#!/usr/bin/env bash

export SAVED_MODEL_DIR=/home/jiazhuang/workspace/2003_openie_tf2/models/roberta-magi-annote-finetune-saved-model/
export DEVICES=3

nvidia-docker run -p 8501:8501 \
   -v ${SAVED_MODEL_DIR}:/models/tf_debug/ \
   -e NVIDIA_VISIBLE_DEVICES=$DEVICES \
   -e MODEL_NAME=tf_debug  -t tensorflow/serving:1.14.0-gpu
