#!/bin/bash
#nohup python train.py > /path/to/logs/out.log 2>&1 &
#nohup bash scripts/run_syn_0215.sh >/home/sunhnayu/jupyterlab/XXI/syn-branch_0906/log/ToZXD.log 2>&1 &


NUM_NODES=1
NUM_GPUS_PER_NODE=4
NODE_RANK=0
MASTER_PORT=$(shuf -n 1 -i 10000-65535)

# set -x #关闭调试模式，脚本将不再打印每一行的执行过程。
# torchrun \
#     --nproc_per_node=$NUM_GPUS_PER_NODE --nnodes=$NUM_NODES --node_rank $NODE_RANK --master_addr localhost --master_port $MASTER_PORT \
#     main_0120.py \
#     --output_dir=weight/syn_pretrain_0215/syn_mit_0215 \
#     --gpus=${NUM_GPUS_PER_NODE} \
#     --num_train_epochs=200 \
#     --num_workers=${NUM_GPUS_PER_NODE} \
#     --batch_size=64 \
#     --learning_rate=0.0005 \
#     --weight_decay=0.0001 \
#     --num_warmup_epochs=1 \
#     --use_sub_smiles_prob=1.0 \
#     --use_smiles_prob=0.8 \
#     --task_percent=1.0 \
#     --config_json_path=configs/bart.json \
#     --input_name=sub_smiles \
#     --output_name=smiles \
#     --seed=42 \
#     --do_train \
#     --do_test \
#     --earlystop \
#     --train_folder=Dataset/mit/mit_train_469214.json \
#     --validation_folder=Dataset/mit/mit_val_2370.json \
#     --test_folder=Dataset/mit/mit_test_2370.json
WANDB_MODE=online
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
set -x #关闭调试模式，脚本将不再打印每一行的执行过程。
torchrun \
    --nproc_per_node=$NUM_GPUS_PER_NODE --nnodes=$NUM_NODES --node_rank $NODE_RANK --master_addr localhost --master_port $MASTER_PORT \
    main_0120.py \
    --output_dir=weight/ToZXD \
    --gpus=${NUM_GPUS_PER_NODE} \
    --num_train_epochs=200 \
    --batch_size=32 \
    --learning_rate=0.0001 \
    --weight_decay=0.0001 \
    --num_warmup_epochs=1 \
    --use_sub_smiles_prob=1.0 \
    --use_smiles_prob=0.8 \
    --task_percent=0.5 \
    --config_json_path=configs/bart.json \
    --input_name=sub_smiles \
    --output_name=smiles \
    --seed=42 \
    --do_train \
    --do_test \
    --mode=reverse \
    --num_beams=1 \
    --use_class_prob=0.0\
    --tokenizer_path=tokenizer/tokenizer-smiles-bart \
    --train_folder=/home/sunhnayu/jupyterlab/XXI/syn-branch_0906/Dataset/fromZXD/train.json \
    --validation_folder=/home/sunhnayu/jupyterlab/XXI/syn-branch_0906/Dataset/fromZXD/valid.json \
    --test_folder=/home/sunhnayu/jupyterlab/XXI/syn-branch_0906/Dataset/fromZXD/test.json \
    --model_weight=weight/20220404_bart_3stage_long_1/epoch_199_loss_0.067575.pth