# Expediting Vision Language Generation via Redundancy Reduction

This repo covers implementations of models in an ARR submission "Expediting Vision Language Generation via Redundancy Reduction". **The code is developed based on [LAVIS](https://github.com/salesforce/LAVIS/) project**.

## Anoymous GitHub Page
We are not part of LAVIS project and we have removed links, usernames, paths, etc which may reveal our identities.


## EVLGen Models
For convenience, we put our model files in `lavis/models/blip2_models`:

- [x] `liptome_opt.py`
- [x] `liptome_video_opt.py`
- [x] `liptome_vicuna.py`
- [x] `liptome_video_vicuna.py`

The temporal augmented EVA-ViT is in  `lavis/models`:
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
We provide an example script for training:
```
python -m torch.distributed.run --nproc_per_node=8 train.py --cfg-path lavis/projects/EVLGen/train/pretrain_opt2.7b.yaml
python -m torch.distributed.run --nproc_per_node=8 train.py --cfg-path lavis/projects/EVLGen_video/train/caption_msrvtt_opt2.7b_ft.yaml

```

## Evaluation
We provide example scripts for evaluations:
```python -m torch.distributed.run --nproc_per_node=8 evaluate.py --cfg-path lavis/projects/EVLGen/eval/caption_coco_opt2.7b_eval.yaml
python -m torch.distributed.run --nproc_per_node=8 evaluate.py --cfg-path lavis/projects/EVLGen_video/eval/caption_msrvtt_opt2.7b_eval.yaml
```

## Acknowlegements
The code is developed based on [LAVIS](https://github.com/salesforce/LAVIS/) project.

