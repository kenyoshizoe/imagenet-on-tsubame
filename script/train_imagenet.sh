#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=24:00:00
#$ -o log_train_imagenet.txt
#$ -j y

cd /gs/bs/tga-lab_otm/htanimura/imagenet-on-tsubame
~/.local/bin/uv run python main.py --config-name imagenet_1k \
    run_name=default \
    trainer.logger=[wandb] \
    backbone=torchvision_r50 \
    backbone.model.weights="" \
    datamodule.cfg.num_workers=64
