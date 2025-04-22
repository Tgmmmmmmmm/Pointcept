#!/bin/bash
while pgrep -f train.sh > /dev/null; do
    echo "脚本正在运行..."
    sleep 300
done
echo "脚本执行完毕，开始执行后续命令..."
# 后续继续执行的命令
# sh scripts/train.sh -d scannet -c semseg-oacnns-v1m1-0-base -n semseg-oacnns-v1m1-0-base -r true
# sh scripts/test.sh -d scannet -n semseg-oacnns-v1m1-0-base -w model_best