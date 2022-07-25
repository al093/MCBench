#!/bin/bash

#SBATCH --job-name=ft_mnli    # create a short name for your job
#SBATCH --output=results/4_1_mnli_randk_1000.txt
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=4      # total number of tasks across all nodes
#SBATCH --cpus-per-task=16        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=64G                 # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:4            # number of gpus per node
#SBATCH --time=202:00:00          # total run time limit (HH:MM:SS)

WORLD_SIZE=4

DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"

TRAIN_DATA="../glue_data/MNLI/train.tsv"
VALID_DATA="../glue_data/MNLI/dev_matched.tsv \
            ../glue_data/MNLI/dev_mismatched.tsv"
VOCAB_FILE="../bert-large-cased-vocab.txt"
PRETRAINED_CHECKPOINT=checkpoints/bert_345m/split_16
CHECKPOINT_PATH=checkpoints/bert_345m_mnli

python3 -m torch.distributed.launch $DISTRIBUTED_ARGS ../tasks/main.py \
               --tensor-model-parallel-size 2 \
               --pipeline-model-parallel-size 2 \
               --task MNLI \
               --seed 1234 \
               --train-data $TRAIN_DATA \
               --valid-data $VALID_DATA \
               --tokenizer-type BertWordPieceLowerCase \
               --vocab-file $VOCAB_FILE \
               --epochs 5 \
               --pretrained-checkpoint $PRETRAINED_CHECKPOINT \
               --num-layers 24 \
               --hidden-size 1024 \
               --num-attention-heads 16 \
               --micro-batch-size 16 \
               --lr 5.0e-5 \
               --lr-warmup-fraction 0.065 \
               --seq-length 512 \
               --max-position-embeddings 512 \
               --save-interval 500000 \
               --save $CHECKPOINT_PATH \
               --log-interval 10 \
               --eval-interval 100 \
               --eval-iters 50 \
               --weight-decay 1.0e-1 \
               --fp16 \
               --is-pipeline-compress False \
               --pipeline-compress-method srht \
               --pipeline-ae-dim 1024 \
               --pipeline-qr-r 10 \
               --pipeline-k 10000 \
               --pipeline-m 50 \
               --is-tensor-compress True \
               --tensor-compress-method topk \
               --tensor-ae-dim 100 \
               --tensor-qr-r 10 \
               --tensor-k 10000 \
               --tensor-m 50 \

