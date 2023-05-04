# VTP(Volumetric Transformer for Multi-view Multi-person 3D Pose Estimation)

## Reference
This code based on https://github.com/microsoft/voxelpose-pytorch. The 'ORIGINAL' in config file is used to decide which model to use, PRN or VTP.

## Installation
1. Clone this repo, and we'll call the directory that you cloned multiview-multiperson-pose as ${POSE_ROOT}.
2. Install dependencies.

## Data preparation

Download the datasets by following the instructions in [voxelpose](https://github.com/microsoft/voxelpose-pytorch) and extract them under `${POSE_ROOT}/data/

## Training
### CMU Panoptic dataset

```
python run/train_3d.py --cfg configs/panoptic/resnet50/VTP/vtp30_emb256.yaml
```
### Shelf/Campus datasets
```
python run/train_3d.py --cfg configs/shelf/VTP/vtp30_emb256.yaml
python run/train_3d.py --cfg configs/campus/VTP/vtp30_emb256.yaml
```

## Evaluation
### CMU Panoptic dataset

Evaluate the models. It will print evaluation results to the screen./
```
python test/evaluate.py --cfg configs/panoptic/resnet50/VTP/vtp30_emb256.yaml
```
### Shelf/Campus datasets

It will print the PCP results to the screen.
```
python test/evaluate.py --cfg configs/shelf/VTP/vtp30_emb256.yaml
python test/evaluate.py --cfg configs/campus/VTP/vtp30_emb256.yaml
```
