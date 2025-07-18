
#!/bin/bash

# 设置运行参数
export NUM_WORKERS=20
export CODEBASE_DIR="$HOME/Pointcept"
export CUDA_VISIBLE_DEVICES="0"
export MAX_SWEEPS=10
export DATA_ROOT="$HOME/data/dataset"

# 设置 ScanNet/ScanNet++ 相关路径
export RAW_SCANNET_DIR="$DATA_ROOT/scannet/scannet"
export RAW_SCANNETPP_DIR="$DATA_ROOT/scannetpp/semantic"
export S3DIS_DIR="$DATA_ROOT/s3dis/data"
export NUSCENES_DIR="$DATA_ROOT/nuscene/processed/raw"

# 设置数据集处理后路径
export SEMANTIC_KITTI_DIR="$DATA_ROOT/semantic_kitti/processed"
export PROCESSED_SCANNETPP_DIR="$DATA_ROOT/scannetpp/processed"
export PROCESSED_SCANNET_DIR="$DATA_ROOT/scannet/processed"
export PROCESSED_S3DIS_DIR="$DATA_ROOT/s3dis/processed"
export PROCESSED_NUSCENES_DIR="$DATA_ROOT/nuscene/processed"

# 打印变量验证（可选）
echo "环境变量已临时设置："
echo "CODEBASE_DIR=$CODEBASE_DIR"
echo "PROCESSED_SCANNET_DIR=$PROCESSED_SCANNET_DIR"
echo "PROCESSED_SCANNETPP_DIR=$PROCESSED_SCANNETPP_DIR"
echo "PROCESSED_S3DIS_DIR=$PROCESSED_S3DIS_DIR"
echo "SEMANTIC_KITTI_DIR=$SEMANTIC_KITTI_DIR"
echo "PROCESSED_NUSCENES_DIR=$PROCESSED_NUSCENES_DIR"
echo "DATA_ROOT=$DATA_ROOT"


# 创建基础目录
mkdir -p ${CODEBASE_DIR}/data

# 检查源目录是否存在，再创建链接
[ -d "${PROCESSED_SCANNET_DIR}" ] && ln -sf "${PROCESSED_SCANNET_DIR}" "${CODEBASE_DIR}/data/scannet"
[ -d "${PROCESSED_SCANNETPP_DIR}" ] && ln -sf "${PROCESSED_SCANNETPP_DIR}" "${CODEBASE_DIR}/data/scannetpp"
[ -d "${PROCESSED_S3DIS_DIR}" ] && ln -sf "${PROCESSED_S3DIS_DIR}" "${CODEBASE_DIR}/data/s3dis"
[ -d "${SEMANTIC_KITTI_DIR}" ] && ln -sf "${SEMANTIC_KITTI_DIR}" "${CODEBASE_DIR}/data/semantic_kitti"
[ -d "${PROCESSED_NUSCENES_DIR}" ] && ln -sf "${PROCESSED_NUSCENES_DIR}" "${CODEBASE_DIR}/data/nuscenes"