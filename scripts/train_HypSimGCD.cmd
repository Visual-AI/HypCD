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
# sbatch scripts/train_HypSimGCD.cmd cub v1 0.1 2.0 0.3
# sbatch scripts/train_HypSimGCD.cmd scars v1 0.1 1.2 0.3
# sbatch scripts/train_HypSimGCD.cmd aircraft v1 0.1 2.3 0.4

# ------------------ DINOv2 ------------------
# sbatch scripts/train_HypSimGCD.cmd cub v2 0.1 1.2 0.4
# sbatch scripts/train_HypSimGCD.cmd scars v2 0.1 1.2 0.35
# sbatch scripts/train_HypSimGCD.cmd aircraft v2 0.1 2.0 0.4

data=$1
dino=$2
hmw=1.0
c=$3
cr=$4
hts=$5

date
srun python -m train.train_HypSimGCD \
    --dataset_name $data \
    --batch_size 128 \
    --grad_from_block 11 \
    --epochs 200 \
    --num_workers 8 \
    --use_ssb_splits \
    --sup_weight 0.35 \
    --weight_decay 5e-5 \
    --transform 'imagenet' \
    --lr 0.1 \
    --warmup_teacher_temp 0.07 \
    --teacher_temp 0.04 \
    --warmup_teacher_temp_epochs 30 \
    --memax_weight 1.0 \
    --exp_name aircraft_simgcd \
    --model_name ${dino} \
    --c $c \
    --cr $cr \
    --hyper_max_weight ${hmw} \
    --hyper_temp_scale ${hts} > logs/hypsimgcd_dino${dino}_${data}_hts${hts}_hmw${hmw}_c${c}_cr${cr}.txt
date
