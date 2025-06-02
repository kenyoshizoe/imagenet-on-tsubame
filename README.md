# Imagenet-On-Tsubame
Sample script for training imagenet on TSUBAME4

## Prerequirement
1. install `uv`
```
curl -LsSf https://astral.sh/uv/install.sh | sh
echo "export UV_CACHE_DIR="/gs/bs/tga-xxx/yyy/.cache"" >> ~/.bashrc
source ~/.bashrc
```
2. clone this repo
3. `uv sync`
4. setup imagenet1k-wds from [huggingface](https://huggingface.co/datasets/timm/imagenet-1k-wds)
5. (optional) configure wandb
    1. signup [wandb](https://wandb.ai/)
    2. get key from [here](https://wandb.ai/authorize)
    3. `uv run wandb login`

## run
edit project path in `./script/train_imagenet.sh` and
```
qsub -g tga-xxx ./script/train_imagenet.sh
```
