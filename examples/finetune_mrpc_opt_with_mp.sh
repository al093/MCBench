#!/bin/bash

# compress method in [ae, quantize, topk_int, randk_int, topk, randk, topk_feedback, randk_feedback, qr]

WORLD_SIZE=4

DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"

TRAIN_DATA="../glue_data/MRPC/msr_paraphrase_train.txt"
VALID_DATA="../glue_data/MRPC/msr_paraphrase_test.txt"
VOCAB_FILE="../opt-3b-vocab.txt"
PRETRAINED_CHECKPOINT=checkpoints/opt_3b/split_2t_2p
#PRETRAINED_CHECKPOINT=checkpoints/bert_345m/split
#PRETRAINED_CHECKPOINT=checkpoints/bert_345m
CHECKPOINT_PATH=checkpoints/opt_3b_mrpc

python3 -m torch.distributed.launch $DISTRIBUTED_ARGS ../tasks/main.py \
               --tensor-model-parallel-size 2 \
               --pipeline-model-parallel-size 2 \
               --task MRPC \
               --seed 1234 \
               --train-data $TRAIN_DATA \
               --valid-data $VALID_DATA \
               --tokenizer-type OPTTokenizer \
               --vocab-file $VOCAB_FILE \
               --epochs 5 \
               --pretrained-checkpoint $PRETRAINED_CHECKPOINT \
               --num-layers 32 \
               --hidden-size 2560 \
               --num-attention-heads 32 \
               --micro-batch-size 4 \
               --lr 5.0e-5 \
               --lr-warmup-fraction 0.065 \
               --seq-length 512 \
               --max-position-embeddings 2050 \
               --save-interval 500000 \
               --log-interval 10 \
               --eval-interval 100 \
               --eval-iters 50 \
               --weight-decay 1.0e-1 \
               --fp16 \
               --is-pipeline-compress True \
               --pipeline-compress-method ae \
               --pipeline-ae-dim 256 \
               --pipeline-qr-r 50 \
               --pipeline-k 10000 \
               --pipeline-m 50 \
               --pipeline-bits 2 \
               --start-pipeline-compress-rank 0 \
               --is-tensor-compress True \
               --tensor-compress-method ae \
               --tensor-ae-dim 256 \
               --tensor-qr-r 50 \
               --tensor-k 10000 \
               --tensor-m 50 \
               --tensor-bits 8 \
               --start-tensor-compress-layer 16 \

