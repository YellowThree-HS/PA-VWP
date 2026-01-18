#!/bin/bash
# 重定向后台训练进程的输出到日志文件

# 找到训练进程
TRAIN_PID=$(ps aux | grep "train/train.py" | grep -v grep | awk '{print $2}' | head -1)

if [ -z "$TRAIN_PID" ]; then
    echo "未找到训练进程"
    exit 1
fi

echo "找到训练进程 PID: $TRAIN_PID"

# 找到对应的 bash 脚本进程（父进程）
SCRIPT_PID=$(ps -o ppid= -p $TRAIN_PID | tr -d ' ')
echo "脚本进程 PID: $SCRIPT_PID"

# 创建日志文件
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/train_transunet_base_h100_${TIMESTAMP}.log"

echo "将输出重定向到: $LOG_FILE"
echo ""
echo "注意：当前进程的输出仍会打印到终端"
echo "建议："
echo "  1. 按 Ctrl+C 停止当前输出（不会终止训练）"
echo "  2. 或者新开一个终端窗口"
echo "  3. 或者使用: tail -f $LOG_FILE 查看日志"
