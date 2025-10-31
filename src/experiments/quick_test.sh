#!/bin/bash

# OnCallAgent 快速实验测试脚本
# 用于验证实验环境和基础功能

set -e

ONCALL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."
EXPERIMENTS_DIR="${ONCALL_ROOT}/experiments"

echo "=========================================="
echo "OnCallAgent 快速实验测试"
echo "=========================================="

# 检查环境
echo "[1/4] 检查实验环境..."

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 未安装"
    exit 1
fi

# 检查必要的Python包
echo "检查Python依赖包..."
if ! python3 -c "
try:
    import torch
    import transformers
    from transformers import pipeline
    print('✓ PyTorch和Transformers已安装')
    print(f'  PyTorch版本: {torch.__version__}')
    print(f'  Transformers版本: {transformers.__version__}')
    print(f'  CUDA可用: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'  GPU数量: {torch.cuda.device_count()}')
        print(f'  GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB')
except ImportError as e:
    print(f'❌ 缺失依赖: {e}')
    print('正在安装依赖包...')
    import subprocess
    import sys
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', '${ONCALL_ROOT}/requirements.txt'])
    print('✓ 依赖包安装完成，请重新运行脚本')
    sys.exit(1)
" 2>/dev/null; then
    echo "❌ 依赖检查失败，尝试安装依赖包..."
    pip3 install -r "${ONCALL_ROOT}/requirements.txt"
    echo "✓ 依赖包安装完成，请重新运行脚本"
    exit 1
fi

echo "✓ 环境检查完成"

# 生成小规模测试数据集
echo "[2/4] 生成测试数据集..."

python3 "${EXPERIMENTS_DIR}/prepare_dataset.py" \
    --data_dir "${ONCALL_ROOT}/data" \
    --output_dir "${EXPERIMENTS_DIR}/test_results/dataset" \
    --size 20 \
    --seed 42 \
    --model_name "Qwen/Qwen2.5-7B-Instruct"

echo "✓ 测试数据集生成完成"

# 运行单个baseline测试
echo "[3/4] 运行单个baseline测试..."

python3 "${EXPERIMENTS_DIR}/run_baseline.py" \
    --method "single_vlm" \
    --dataset "${EXPERIMENTS_DIR}/test_results/dataset/test_set.json" \
    --output "${EXPERIMENTS_DIR}/test_results/quick_test_result.json" \
    --config "${ONCALL_ROOT}/config.json" \
    --batch_size 5

echo "✓ Baseline测试完成"

# 运行简单消融实验
echo "[4/4] 运行消融实验测试..."

python3 "${EXPERIMENTS_DIR}/run_ablation.py" \
    --ablation_type "no_rag" \
    --dataset "${EXPERIMENTS_DIR}/test_results/dataset/test_set.json" \
    --output "${EXPERIMENTS_DIR}/test_results/ablation_test_result.json" \
    --config "${ONCALL_ROOT}/config.json" \
    --batch_size 3

echo "✓ 消融实验测试完成"

echo "=========================================="
echo "快速测试完成！"
echo "结果保存在: ${EXPERIMENTS_DIR}/test_results/"
echo "如果测试成功，可以运行完整实验:"
echo "  bash ${EXPERIMENTS_DIR}/run_all_experiments.sh"
echo "=========================================="
