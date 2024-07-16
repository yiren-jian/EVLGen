# Expedited Training of Visual Conditioned Language Generation via Redundancy Reduction

This repo covers implementations of models in: **[Expedited Training of Visual Conditioned Language Generation via Redundancy Reduction]()** accepted at **ACL 2024**. The code is developed based on [LAVIS](https://github.com/salesforce/LAVIS/) project.


## EVLGen Models
For convenience, we put our model files in `lavis/models/blip2_models`:

- [x] `liptome_opt.py`
- [x] `liptome_video_opt.py`
- [x] `liptome_vicuna.py`
- [x] `liptome_video_vicuna.py`

Tomeformer is in `lavis/models/blip2_models/modeling_bert.py`. This file is edited from official HuggingFace implementation with:

- `bipartite_soft_matching()`
- `parse_r()`

from [ToMe](https://github.com/facebookresearch/ToMe). These two functions are used in `BertLayer.forward()`.

Tomeformer is used as the vision-language connector in `liptome_opt.py`:
```python
from .modeling_bert import BertConfig, BertModel

encoder_config = BertConfig.from_pretrained("bert-base-uncased", cache_dir="/mnt/bn/username/cache_dir")
encoder_config.num_hidden_layers = num_layers
encoder_config.r = tome_r
self.VLconnector = BertModel.from_pretrained("bert-base-uncased", config=encoder_config, cache_dir="/mnt/bn/username/cache_dir")
```

The temporal augmented EVA-ViT (used in EVLGen-video) is in `lavis/models`:
- [x] `eva_vit_g.py`

## Installation

```bash
conda create -n lavis python=3.8
conda activate lavis
pip install -e .

pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

## Data Preparation
Please follow instructions from [LAVIS](https://github.com/salesforce/LAVIS/) to download pre-training datasets. CapFilt dataset can be downloaded from [BLIP](https://github.com/salesforce/BLIP).

## Training
We provide example script for training (image and video models):
```bash
python -m torch.distributed.run --nproc_per_node=8 train.py --cfg-path lavis/projects/EVLGen/train/pretrain_opt2.7b.yaml

python -m torch.distributed.run --nproc_per_node=8 train.py --cfg-path lavis/projects/EVLGen_video/train/caption_msrvtt_opt2.7b_ft.yaml

```

## Evaluation
We provide example scripts for evaluation (image and video models):
```bash
python -m torch.distributed.run --nproc_per_node=8 evaluate.py --cfg-path lavis/projects/EVLGen/eval/caption_coco_opt2.7b_eval.yaml
python -m torch.distributed.run --nproc_per_node=8 evaluate.py --cfg-path lavis/projects/EVLGen/eval/gqa_zeroshot_opt2.7b_eval.yaml
python -m torch.distributed.run --nproc_per_node=8 evaluate.py --cfg-path lavis/projects/EVLGen/eval/okvqa_zeroshot_opt2.7b_eval.yaml
python -m torch.distributed.run --nproc_per_node=8 evaluate.py --cfg-path lavis/projects/EVLGen/eval/vqav2_zeroshot_opt2.7b_eval.yaml

python -m torch.distributed.run --nproc_per_node=8 evaluate.py --cfg-path lavis/projects/EVLGen_video/eval/caption_msrvtt_opt2.7b_eval.yaml
```

## Pretrained Models
We provide pretrained models (EVLGen-image based on opt-2.7b and vicuna-7b) in [Dropbox](https://www.dropbox.com/scl/fo/zily1eayhexf6qa85i1qz/ADqocf2e2PMcFYDcJ0Cleu0?rlkey=g3vsom8t6edtkwyr22qfmd3w4&st=oxdq18ds&dl=0).

## Acknowlegements
The code is developed based on [ToMe](https://github.com/facebookresearch/ToMe) and [LAVIS](https://github.com/salesforce/LAVIS/) project.
