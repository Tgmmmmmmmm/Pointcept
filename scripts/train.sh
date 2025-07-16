#!/bin/sh

# 获取当前执行的完整命令
FULL_CMD="sh $0 $*"

cd $(dirname $(dirname "$0")) || exit
ROOT_DIR=$(pwd)
PYTHON=python

TRAIN_CODE=train.py

DATASET=scannet
CONFIG="None"
EXP_NAME=debug
WEIGHT="None"
RESUME=false
NUM_GPU=None
NUM_MACHINE=1
DIST_URL="auto"


while getopts "p:d:c:n:w:g:m:r:" opt; do
  case $opt in
    p)
      PYTHON=$OPTARG
      ;;
    d)
      DATASET=$OPTARG
      ;;
    c)
      CONFIG=$OPTARG
      ;;
    n)
      EXP_NAME=$OPTARG
      ;;
    w)
      WEIGHT=$OPTARG
      ;;
    r)
      RESUME=$OPTARG
      ;;
    g)
      NUM_GPU=$OPTARG
      ;;
    m)
      NUM_MACHINE=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG"
      ;;
  esac
done

if [ "${NUM_GPU}" = 'None' ]
then
  NUM_GPU=`$PYTHON -c 'import torch; print(torch.cuda.device_count())'`
fi

echo "Experiment name: $EXP_NAME"
echo "Python interpreter dir: $PYTHON"
echo "Dataset: $DATASET"
echo "Config: $CONFIG"
echo "GPU Num: $NUM_GPU"
echo "Machine Num: $NUM_MACHINE"

if [ -n "$SLURM_NODELIST" ]; then
  MASTER_HOSTNAME=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
  MASTER_ADDR=$(getent hosts "$MASTER_HOSTNAME" | awk '{ print $1 }')
  MASTER_PORT=$((10000 + 0x$(echo -n "${DATASET}/${EXP_NAME}" | md5sum | cut -c 1-4 | awk '{print $1}') % 20000))
  DIST_URL=tcp://$MASTER_ADDR:$MASTER_PORT
fi

echo "Dist URL: $DIST_URL"

EXP_DIR=exp/${DATASET}/${EXP_NAME}
MODEL_DIR=${EXP_DIR}/model
CODE_DIR=${EXP_DIR}/code
CONFIG_DIR=configs/${DATASET}/${CONFIG}.py
COMMAND_FILE=${EXP_DIR}/command.txt  # 新增：命令记录文件
RUN_LOG="${EXP_DIR}/run_history.log"  # 运行历史记录文件



echo " =========> CREATE EXP DIR <========="
echo "Experiment dir: $ROOT_DIR/$EXP_DIR"
if [ "${RESUME}" = true ] && [ -d "$EXP_DIR" ]
then
  CONFIG_DIR=${EXP_DIR}/config.py
  WEIGHT=$MODEL_DIR/model_last.pth
else
  RESUME=false
  mkdir -p "$MODEL_DIR" "$CODE_DIR"
  cp -r scripts tools pointcept "$CODE_DIR"
fi

echo "Loading config in:" $CONFIG_DIR
export PYTHONPATH=./$CODE_DIR
echo "Running code in: $CODE_DIR"

# 记录本次运行的完整命令和时间戳
mkdir -p "$EXP_DIR"
echo "$(date '+%Y-%m-%d %H:%M:%S') - $FULL_CMD" >> "$RUN_LOG"

echo " =========> RUN TASK <========="
ulimit -n 65536
if [ "${WEIGHT}" = "None" ]
then
    CMD="$PYTHON \"$CODE_DIR\"/tools/$TRAIN_CODE \
    --config-file \"$CONFIG_DIR\" \
    --num-gpus \"$NUM_GPU\" \
    --num-machines \"$NUM_MACHINE\" \
    --machine-rank ${SLURM_NODEID:-0} \
    --dist-url ${DIST_URL} \
    --options save_path=\"$EXP_DIR\""
else
    CMD="$PYTHON \"$CODE_DIR\"/tools/$TRAIN_CODE \
    --config-file \"$CONFIG_DIR\" \
    --num-gpus \"$NUM_GPU\" \
    --num-machines \"$NUM_MACHINE\" \
    --machine-rank ${SLURM_NODEID:-0} \
    --dist-url ${DIST_URL} \
    --options save_path=\"$EXP_DIR\" resume=\"$RESUME\" weight=\"$WEIGHT\""
fi

# 保存命令到文件
echo "Saving command to: $COMMAND_FILE"
echo "$(date '+%Y-%m-%d %H:%M:%S') - $CMD" >> "$COMMAND_FILE"

# 执行命令
eval "$CMD"