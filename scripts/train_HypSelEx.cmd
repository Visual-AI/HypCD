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
# sbatch scripts/train_HypSelEx.cmd cub v1 1.0 1.0 0.1 2.0
# sbatch scripts/train_HypSelEx.cmd scars v1 0.5 1.0 0.1 2.0
# sbatch scripts/train_HypSelEx.cmd aircraft v1 0.5 0.5 0.1 2.0

# ------------------ DINOv2 ------------------
# sbatch scripts/train_HypSelEx.cmd cub v2 1.0 1.0 0.1 1.5
# sbatch scripts/train_HypSelEx.cmd scars v2 0.5 0.5 0.1 2.0
# sbatch scripts/train_HypSelEx.cmd aircraft v2 0.5 0.5 0.05 1.5


data=$1
dino=$2
us=$3
hmw=$4
c=$5
cr=$6
dim=8192

date
srun python -m train.train_HypSelEx \
    --dataset_name $data \
    --batch_size 128 \
    --grad_from_block 10 \
    --epochs 200 \
    --base_model vit_dino \
    --num_workers 4 \
    --use_ssb_splits 'True' \
    --sup_con_weight 0.35 \
    --weight_decay 5e-5 \
    --contrast_unlabel_only 'False' \
    --transform 'imagenet' \
    --lr 0.1 \
    --eval_funcs 'v2' \
    --c $c \
    --cr $cr \
    --dino ${dino} \
    --mlp_out_dim ${dim} \
    --unsupervised_smoothing ${us} \
    --hyper_max_weight ${hmw} > logs/hypselex_dino${dino}_${data}_us${us}_hmw${hmw}_c${c}_cr${cr}.txt
date
