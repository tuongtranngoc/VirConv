# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VirConv implements **Virtual Sparse Convolution for Multimodal 3D Object Detection** (CVPR 2023). It extends [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) with three detector variants trained on the KITTI dataset:

- **VirConv-L**: Lightweight multimodal detector
- **VirConv-T**: Improved detector with transformed refinement scheme
- **VirConv-S**: Semi-supervised extension of VirConv-T using pseudo labels from KITTI odometry data

The key novelty is **Virtual Sparse Convolution**: virtual RGB points (generated via PENet depth completion from camera images) are fused with real LiDAR points and processed through a modified sparse convolution backbone that discards a configurable fraction of virtual voxels per layer.

## Setup

```bash
# Install the pcdet package and compile CUDA extensions
python setup.py develop
```

This compiles five CUDA extensions: `votr_ops_cuda`, `iou3d_nms_cuda`, `roiaware_pool3d_cuda`, `roipoint_pool3d_cuda`, `pointnet2_stack_cuda`, and `pointnet2_batch_cuda`.

## Dataset Preparation

The multimodal models require a `velodyne_depth/` folder alongside the standard KITTI `velodyne/` folders. Generate it with PENet depth completion:

```bash
# 1. Create semi-supervised split from KITTI odometry
cd tools
python3 creat_semi_dataset.py ../data/odometry ../data/kitti/semi

# 2. Generate RGB virtual points using PENet depth model
cd tools/PENet
python3 main.py --detpath ../../data/kitti/training
python3 main.py --detpath ../../data/kitti/testing
python3 main.py --detpath ../../data/kitti/semi  # only for VirConv-S

# 3. Generate dataset info pkl files
python3 -m pcdet.datasets.kitti.kitti_dataset_mm create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
python3 -m pcdet.datasets.kitti.kitti_datasetsemi create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
```

## Training

```bash
cd tools

# Single GPU
python3 train.py --cfg_file cfgs/models/kitti/VirConv-L.yaml

# Multi-GPU (edit CUDA_VISIBLE_DEVICES and nproc_per_node in dist_train.sh first)
sh dist_train.sh
# Logs written to log.txt

# VirConv-S requires a pretrained VirConv-T checkpoint
python3 train.py --cfg_file cfgs/models/kitti/VirConv-S.yaml \
    --pretrained_model ../output/models/kitti/VirConv-T/default/ckpt/checkpoint_epoch_40.pth
```

Key training flags: `--batch_size`, `--epochs`, `--extra_tag` (for output subdirectory), `--ckpt` (resume from checkpoint).

## Evaluation

```bash
cd tools

# Single GPU
python3 test.py --cfg_file cfgs/models/kitti/VirConv-S.yaml --batch_size 1 --ckpt VirConv-S.pth

# Multi-GPU (edit dist_test.sh first)
sh dist_test.sh
# Logs written to log-test.txt
```

Outputs are saved to `../output/<cfg_group>/<model_name>/<extra_tag>/`. Evaluation metric is KITTI 3D AP (R40) for Car.

## Architecture

All three variants follow the **two-stage VoxelRCNN** pipeline defined in `pcdet/models/detectors/detector3d_template.py`:

```
Raw points (LiDAR + virtual RGB)
  → VFE (MeanVFE): per-voxel feature aggregation
  → 3D Backbone (VirConvL8x / VirConvT8x): sparse 3D convolutions with virtual-point discard
  → MapToBEV (HeightCompression): flatten Z axis → BEV feature map
  → 2D Backbone (BaseBEVBackbone): 2D CNN on BEV
  → Dense Head (AnchorHeadSingle): RPN proposals
  → ROI Head (TEDMHead): per-proposal refinement using grid pooling from 3D features
```

**Key implementation files:**

| File | Role |
|------|------|
| [pcdet/models/backbones_3d/spconv_backbone.py](pcdet/models/backbones_3d/spconv_backbone.py) | VirConv backbone — contains `layer_voxel_discard`, `index2uv`, virtual point fusion logic |
| [pcdet/models/detectors/voxel_rcnn.py](pcdet/models/detectors/voxel_rcnn.py) | Top-level model wiring |
| [pcdet/models/roi_heads/ted_head.py](pcdet/models/roi_heads/ted_head.py) | ROI head with dual grid pooling (LiDAR + MM streams) |
| [pcdet/datasets/kitti/kitti_dataset_mm.py](pcdet/datasets/kitti/kitti_dataset_mm.py) | Multimodal KITTI dataset — loads `velodyne_depth` points |
| [pcdet/datasets/kitti/kitti_datasetsemi.py](pcdet/datasets/kitti/kitti_datasetsemi.py) | Semi-supervised dataset with pseudo labels |
| [pcdet/datasets/dataset.py](pcdet/datasets/dataset.py) | Base dataset — input point discard (`INPUT_DISCARD_RATE`) applied here |
| [tools/cfgs/models/kitti/](tools/cfgs/models/kitti/) | YAML configs for VirConv-L, VirConv-T, VirConv-S |

## Configuration System

Configs live in `tools/cfgs/` and use a `_BASE_CONFIG_` inheritance key. The dataset config ([tools/cfgs/dataset_configs/kitti_dataset.yaml](tools/cfgs/dataset_configs/kitti_dataset.yaml)) is shared; model configs override or extend it.

Key multimodal config fields:
- `INPUT_DISCARD_RATE`: fraction of virtual points dropped at input (default 0.8; effectively discards >90% since PENet only saves <50% of RGB points)
- `LAYER_DISCARD_RATE`: fraction of virtual voxels dropped per sparse conv layer (in backbone config)
- `MM_PATH`: subfolder name for virtual point clouds (`velodyne_depth`)
- `LATER_FUSION`: if True, keeps LiDAR and virtual streams separate until ROI head

## CUDA Ops

Custom CUDA extensions are in `pcdet/ops/`. If you modify `.cpp` or `.cu` files, re-run `python setup.py develop` to recompile. The git status shows several `.cpp` files as modified — these may contain local adaptations to the build system.
