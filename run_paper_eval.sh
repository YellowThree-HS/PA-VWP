#!/bin/bash
#
# 论文评估脚本
# 评估所有模型在测试数据集上的论文指标
#

# 配置
CHECKPOINT_DIR="/DATA/disk0/hs_25/pa/checkpoints"
TEST_DIR="/DATA/disk0/hs_25/pa/all_dataset/test"
OUTPUT_DIR="paper_eval_results"
BATCH_SIZE=32

# 显示帮助
show_help() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --all                评估所有带有 'best' 检查点的模型 (默认行为)"
    echo "  --include_epoch      也评估只有 epoch checkpoint 的模型"
    echo "  --checkpoint PATH    评估单个模型"
    echo "  --test_dir PATH      测试数据目录 (默认: $TEST_DIR)"
    echo "  --batch_size N       批次大小 (默认: $BATCH_SIZE)"
    echo "  --help               显示帮助"
    echo ""
    echo "示例:"
    echo "  $0 --all                  # 只评估有 best 检查点的模型"
    echo "  $0 --all --include_epoch  # 评估所有模型"
    echo "  $0 --checkpoint /DATA/disk0/hs_25/pa/checkpoints/transunet_fusion_tiny_h100"
}

# 激活环境
activate_env() {
    echo "激活 conda 环境..."
    source /home/hs_25/miniconda3/etc/profile.d/conda.sh
    conda activate isaac
}

# 评估所有模型
eval_all() {
    local include_epoch="$1"
    echo "=============================================="
    echo "评估所有模型 (只评估有 best 检查点的模型)"
    echo "检查点目录: $CHECKPOINT_DIR"
    echo "测试数据集: $TEST_DIR"
    echo "=============================================="
    
    local extra_args=""
    if [ "$include_epoch" = "true" ]; then
        extra_args="--include_epoch"
        echo "模式: 包括只有 epoch checkpoint 的模型"
    fi
    
    python evaluate_paper_metrics.py \
        --all \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --test_dir "$TEST_DIR" \
        --batch_size $BATCH_SIZE \
        --output_dir "$OUTPUT_DIR" \
        $extra_args
}

# 评估单个模型
eval_single() {
    local checkpoint="$1"
    echo "="
    echo "评估模型: $checkpoint"
    echo "测试数据集: $TEST_DIR"
    echo "="
    
    python evaluate_paper_metrics.py \
        --checkpoint "$checkpoint" \
        --test_dir "$TEST_DIR" \
        --batch_size $BATCH_SIZE \
        --output_dir "$OUTPUT_DIR"
}

# 主函数
main() {
    cd /home/hs_25/projs/PA-VWP
    
    if [ $# -eq 0 ]; then
        show_help
        exit 0
    fi
    
    local do_all=false
    local include_epoch=false
    local checkpoint=""
    
    # 解析参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            --all)
                do_all=true
                shift
                ;;
            --include_epoch)
                include_epoch=true
                shift
                ;;
            --checkpoint)
                if [ -z "$2" ]; then
                    echo "错误: --checkpoint 需要一个参数"
                    exit 1
                fi
                checkpoint="$2"
                shift 2
                ;;
            --test_dir)
                TEST_DIR="$2"
                shift 2
                ;;
            --batch_size)
                BATCH_SIZE="$2"
                shift 2
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                echo "未知参数: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 执行评估
    if [ "$do_all" = true ]; then
        activate_env
        eval_all "$include_epoch"
    elif [ -n "$checkpoint" ]; then
        activate_env
        eval_single "$checkpoint"
    else
        show_help
    fi
}

main "$@"
