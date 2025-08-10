#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --partition=l40s
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=2-
#SBATCH --output=slurm_out/%x_%j.out
#SBATCH --error=slurm_out/%x_%j.err

# ------------------ DINO ------------------
# sbatch scripts/train_HypGCD.cmd cub v1 0.1 1.2 1.0
# sbatch scripts/train_HypGCD.cmd scars v1 0.1 1.2 1.0
# sbatch scripts/train_HypGCD.cmd aircraft v1 0.1 1.2 1.0

# ------------------ DINOv2 ------------------
# sbatch scripts/train_HypGCD.cmd cub v2 0.1 1.2 1.0
# sbatch scripts/train_HypGCD.cmd scars v2 0.1 1.2 1.0
# sbatch scripts/train_HypGCD.cmd aircraft v2 0.1 1.2 1.0


data=$1
dino=$2
c=$3
cr=$4
hmw=$5


date
srun python -m train.train_HypGCD \
            --dataset_name $data \
            --batch_size 128 \
            --grad_from_block 11 \
            --epochs 200 \
            --base_model vit_dino \
            --num_workers 16 \
            --use_ssb_splits 'True' \
            --sup_con_weight 0.35 \
            --weight_decay 5e-5 \
            --contrast_unlabel_only 'False' \
            --transform 'imagenet' \
            --lr 0.1 \
            --c $c \
            --hyper_start_epoch 0 \
            --hyper_end_epoch 200 \
            --eval_funcs 'v2' \
            --cr $cr \
            --hyper_max_weight $hmw \
            --dino $dino > logs/hypgcd_dino${dino}_${data}_hmw${hmw}_c${c}_cr${cr}.txt
date
