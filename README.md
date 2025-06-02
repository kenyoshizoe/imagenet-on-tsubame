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

## run
```
qsub -g tga-xxx ./script/train_imagenet.sh
```
