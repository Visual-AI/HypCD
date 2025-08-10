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
# sbatch scripts/eval_HypGCD.cmd cub v1
# sbatch scripts/eval_HypGCD.cmd scars v1
# sbatch scripts/eval_HypGCD.cmd aircraft v1

# ------------------ DINOv2 ------------------
# sbatch scripts/eval_HypGCD.cmd cub v2
# sbatch scripts/eval_HypGCD.cmd scars v2
# sbatch scripts/eval_HypGCD.cmd aircraft v2

data=$1
dino=$2
c=0.1
cr=1.2
hmw=1.0

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
            --eval_only \
            --eval_model_path hypcd_models/hypgcd/dino${dino}/${data}/model_best.pt \
            --dino $dino > logs/eval_hypgcd_dino${dino}_${data}.txt
date
