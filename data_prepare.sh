#!/bin/bash
conda init
conda activate pointcept
# 设置 ScanNet/ScanNet++ 相关路径
export RAW_SCANNET_DIR="/media/tgm/TgM/dataset/scannet/scannet"
export RAW_SCANNETPP_DIR="/media/tgm/TgM/dataset/scannetpp/semantic"
export S3DIS_DIR="/media/tgm/TgM/dataset/s3dis/data"
export NUSCENES_DIR="/media/tgm/TgM/dataset/nuscene/processed/raw"

export SEMANTIC_KITTI_DIR="/media/tgm/TgM/dataset/semantic_kitti/processed"
export PROCESSED_SCANNETPP_DIR="/media/tgm/TgM/dataset/scannetpp/processed"
export PROCESSED_SCANNET_DIR="/media/tgm/TgM/dataset/scannet/processed"
export PROCESSED_S3DIS_DIR="/media/tgm/TgM/dataset/s3dis/processed"
export PROCESSED_NUSCENES_DIR="/media/tgm/TgM/dataset/nuscene/processed"
# 设置运行参数
export NUM_WORKERS=20
export CODEBASE_DIR="/home/tgm/Project/Pointcept"
export CUDA_VISIBLE_DEVICES="0"
export MAX_SWEEPS=10


# 打印变量验证（可选）
echo "环境变量已临时设置："
echo "CODEBASE_DIR=$CODEBASE_DIR"
echo "PROCESSED_SCANNET_DIR=$PROCESSED_SCANNET_DIR"
echo "PROCESSED_SCANNETPP_DIR=$PROCESSED_SCANNETPP_DIR"
echo "PROCESSED_S3DIS_DIR=$PROCESSED_S3DIS_DIR"
echo "SEMANTIC_KITTI_DIR=$SEMANTIC_KITTI_DIR"
echo "PROCESSED_NUSCENES_DIR=$PROCESSED_NUSCENES_DIR"

# 连接数据集目录
mkdir data
# Scannet
# PROCESSED_SCANNET_DIR: the directory of the processed ScanNet dataset.
ln -s ${PROCESSED_SCANNET_DIR} ${CODEBASE_DIR}/data/scannet
# PROCESSED_SCANNETPP_DIR: the directory of the processed ScanNet dataset.
ln -s ${PROCESSED_SCANNETPP_DIR} ${CODEBASE_DIR}/data/scannetpp
# PROCESSED_S3DIS_DIR: the directory of processed S3DIS dataset.
ln -s ${PROCESSED_S3DIS_DIR} ${CODEBASE_DIR}/data/s3dis
# SEMANTIC_KITTI_DIR: the directory of SemanticKITTI dataset.
ln -s ${SEMANTIC_KITTI_DIR} ${CODEBASE_DIR}/data/semantic_kitti
# PROCESSED_NUSCENES_DIR: the directory of processed nuScenes dataset (output dir).
ln -s ${PROCESSED_NUSCENES_DIR} ${CODEBASE_DIR}/data/nuscenes