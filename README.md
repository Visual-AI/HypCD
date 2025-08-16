#  Hyperbolic Category Discovery (CVPR 2025)


<p align="center">
    <a href="https://arxiv.org/abs/2504.06120"><img src="https://img.shields.io/badge/arXiv-2504.06120-b31b1b"></a>
    <a href="https://visual-ai.github.io/hypcd/"><img src="https://img.shields.io/badge/Project-Website-blue"></a>
    <a href="#jump"><img src="https://img.shields.io/badge/Citation-8A2BE2"></a>
</p>
<p align="center">
	Hyperbolic Category Discovery <br>
  By
  Yuanpei Liu*, 
  Zhenqi He*, and 
  Kai Han.
</p>

![teaser](assets/method.png)

## Prerequisite 🛠️

First, you need to clone the HypCD repository from GitHub. Open the terminal and run the following command:

```
git clone https://github.com/Visual-AI/HypCD.git
cd HypCD
```

We recommend setting up a conda environment for the project:

```
conda create --name=hypcd python=3.8
conda activate hypcd
pip install -r requirements.txt
```

Install the ``kmeans_pytorch``:
```
git clone https://github.com/subhadarship/kmeans_pytorch
cd kmeans_pytorch
pip install --editable .
```
Download the pretrained DINO/DINOv2 weights from their official repository to the ``PRETRAINED_PATH``.

## Running 🏃
### Config

Set paths to datasets, pretrained weights, and log directories in ``config.py``


### Datasets

We use generic object recognition datasets, including CIFAR-10/100 and ImageNet-100:

* [CIFAR-10/100](https://pytorch.org/vision/stable/datasets.html) and [ImageNet-100](https://image-net.org/download.php)

We also use fine-grained benchmarks (CUB, Stanford-cars, FGVC-aircraft). You can find the datasets in:

* [The Semantic Shift Benchmark (SSB)](https://github.com/sgvaze/osr_closed_set_all_you_need#ssb)


### Scripts
We use the slurm system to run the code. The scripts to train and eval each method can be found in the folder `/scripts`. For example, to train and eval the HypSimGCD models.

**Eval the model**
```
sbatch scripts/eval_HypSimGCD.cmd cub v1 0.1 2.0
```

**Train the model**:

```
sbatch scripts/train_HypSimGCD.cmd cub v1 0.1 2.0 0.3
```
Just change the dataset name (``cub``), its corresponding DINO version (``v1``) and the hyperparameters. The detailed parameters used on each dataset can be found in the file ``scripts/eval_HypSimGCD.cmd``. The resulted logs can be found in the directory ``logs``.



## Citing this work
<span id="jump"></span>
If you find this repo useful for your research, please consider citing our paper:

```
@inproceedings{liu2025hyperbolic,
  title={Hyperbolic Category Discovery},
  author={Liu, Yuanpei and He, Zhenqi and Han, Kai},
  booktitle={CVPR},
  year={2025}
}
```