#!/usr/bin/env bash


#model_path=/home/hyh/projects/benchmark/saved_models/webface_8kstep_lowlba
#mkdir -p ${model_path}
#CUDA_VISIBLE_DEVICES=0 nohup python -u train_webface.py --dataset_path=/home/hyh/projects/benchmark/data/webface --log_dir=${model_path} --softmax_type=a-softmax --cnn_model=a-softmax --margin=10 --batch_size=512 --max_step=16000 --learning_rate=1e-5 --eval_path=/home/hyh/projects/benchmark/data/lfw --base_lambda=1000 >> ${model_path}/webface_train.out &

MODEL_PATH=/home/hyh/projects/benchmark/saved_models/reproduce/webface_16kstep_20klba_newtrainingfile
mkdir -p ${MODEL_PATH}
CUDA_VISIBLE_DEVICES=0 nohup python -u train_webface.py --dataset_path=/home/hyh/projects/benchmark/data/webface --log_dir=${MODEL_PATH} --softmax_type=a-softmax --cnn_model=a-softmax --margin=10 --batch_size=512 --max_step=24000 --learning_rate=1e-4 --eval_path=/home/hyh/projects/benchmark/data/lfw --base_lambda=10000 >> ${MODEL_PATH}/webface_train.out &