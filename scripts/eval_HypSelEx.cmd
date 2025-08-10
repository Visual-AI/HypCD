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
# sbatch scripts/eval_HypSelEx.cmd cub v1
# sbatch scripts/eval_HypSelEx.cmd scars v1
# sbatch scripts/eval_HypSelEx.cmd aircraft v1

# ------------------ DINOv2 ------------------
# sbatch scripts/eval_HypSelEx.cmd cub v2
# sbatch scripts/eval_HypSelEx.cmd scars v2
# sbatch scripts/eval_HypSelEx.cmd aircraft v2

data=$1
dino=$2
us=1.0
hmw=1.0
c=0.1
cr=1.5

dim=8192

WORK_DIR='/home/ypliu0/projects/HypCD'

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
    --eval_only \
    --eval_model_path hypcd_models/hypselex/dino${dino}/${data}/model_best_train_acc.pt \
    --hyper_max_weight ${hmw} > logs/eval_hypselex_dino${dino}_${data}.txt
date
