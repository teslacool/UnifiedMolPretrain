# Unified 2D and 3D Pre-Training of Molecular Representations
This repository contains the code for Unified 2D and 3D Pre-Training of Molecular Representations, which is introduced in KDD2022.

## Dataset
We use the [PCQM4Mv2 dataset](https://ogb.stanford.edu/docs/lsc/pcqm4mv2/) for pre-training, which has 3.38M data.

## Requirements and Installation
* PyTorch
* Torch-Geometric

You can build a Docker image with the [Dockerfile](Dockerfile).
To install code and develop it locally
```shell
pip install -e . 
```

## Pre-training
```shell
bash run_training.sh --num-layers 12 --batch-size 128 --enable-tb \
    --node-attn --use-bn --pred-pos-residual --mask-prob 0.25 \
    -c 0,1,2,3 --dist
```

## Finetuning
```shell
bash run_finetune.sh --num-layers 12 --batch-size 128 \
        --dropout 0.3 --dataset ogbg-molpcba \
        --pooler-dropout 0.1 --epochs 50 --seed 42 \
        -m /yourpretrainedmodel \
        --lr 0.0005 --weight-decay 0.01 --grad-norm 1 --prefix molpcba
```

