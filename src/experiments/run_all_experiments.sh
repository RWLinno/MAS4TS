#!/bin/bash

# OnCallAgent 完整实验验证脚本
# 运行所有基准测试、消融实验和超参数分析

set -e  # 出错时退出

# ============================================================================
# 实验配置
# ============================================================================

# 基础路径
ONCALL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."
EXPERIMENTS_DIR="${ONCALL_ROOT}/experiments"
RESULTS_DIR="${EXPERIMENTS_DIR}/results"
DATA_DIR="${ONCALL_ROOT}/data"

# 创建结果目录
mkdir -p "${RESULTS_DIR}"/{baselines,ablation,hyperparams,analysis}

# 实验参数
DATASET_SIZE=500
EVAL_BATCH_SIZE=10
MAX_WORKERS=4

# 支持的开源模型列表
MODELS=(
    "Qwen/Qwen2.5-7B-Instruct"
    "Qwen/Qwen2.5-14B-Instruct" 
    "meta-llama/Llama-2-7b-chat-hf"
    "THUDM/chatglm3-6b"
    "baichuan-inc/Baichuan2-7B-Chat"
    "internlm/internlm2-chat-7b"
)

# 默认使用的模型（如果GPU内存不足，可以选择较小的模型）
DEFAULT_MODEL="Qwen/Qwen2.5-7B-Instruct"

# 日志设置
LOG_FILE="${RESULTS_DIR}/experiment_$(date +%Y%m%d_%H%M%S).log"
exec 1> >(tee -a "${LOG_FILE}")
exec 2>&1

echo "=========================================="
echo "OnCallAgent 实验验证开始"
echo "时间: $(date)"
echo "数据集大小: ${DATASET_SIZE}"
echo "日志文件: ${LOG_FILE}"
echo "=========================================="

# ============================================================================
# 1. 数据集准备
# ============================================================================

echo "[1/5] 准备实验数据集..."

python "${EXPERIMENTS_DIR}/prepare_dataset.py" \
    --data_dir "${DATA_DIR}" \
    --output_dir "${RESULTS_DIR}/dataset" \
    --size ${DATASET_SIZE} \
    --seed 42 \
    --model_name "${DEFAULT_MODEL}"

echo "✓ 数据集准备完成"

# ============================================================================
# 2. Baseline对比实验
# ============================================================================

echo "[2/5] 运行Baseline对比实验..."

BASELINES=("gpt4v_direct" "single_vlm" "llm_only" "rag_only" "autogpt_devops" "oncall_agent")

for baseline in "${BASELINES[@]}"; do
    echo "运行 ${baseline} baseline..."
    
    python "${EXPERIMENTS_DIR}/run_baseline.py" \
        --method "${baseline}" \
        --dataset "${RESULTS_DIR}/dataset/test_set.json" \
        --output "${RESULTS_DIR}/baselines/${baseline}_results.json" \
        --config "${ONCALL_ROOT}/config.json" \
        --batch_size ${EVAL_BATCH_SIZE} \
        --max_workers ${MAX_WORKERS}
    
    echo "✓ ${baseline} 完成"
done

echo "✓ Baseline实验完成"

# ============================================================================
# 3. 消融实验
# ============================================================================

echo "[3/5] 运行消融实验..."

# Agent移除实验
ABLATIONS=("no_visual" "no_log" "no_metrics" "no_knowledge" "no_route" "no_rag" "no_mcp")

for ablation in "${ABLATIONS[@]}"; do
    echo "运行消融实验: ${ablation}..."
    
    python "${EXPERIMENTS_DIR}/run_ablation.py" \
        --ablation_type "${ablation}" \
        --dataset "${RESULTS_DIR}/dataset/test_set.json" \
        --output "${RESULTS_DIR}/ablation/${ablation}_results.json" \
        --config "${ONCALL_ROOT}/config.json" \
        --batch_size ${EVAL_BATCH_SIZE}
    
    echo "✓ ${ablation} 完成"
done

# Agent替换实验
REPLACEMENTS=("normal_llm" "simple_route" "fixed_assignment")

for replacement in "${REPLACEMENTS[@]}"; do
    echo "运行替换实验: ${replacement}..."
    
    python "${EXPERIMENTS_DIR}/run_replacement.py" \
        --replacement_type "${replacement}" \
        --dataset "${RESULTS_DIR}/dataset/test_set.json" \
        --output "${RESULTS_DIR}/ablation/${replacement}_results.json" \
        --config "${ONCALL_ROOT}/config.json" \
        --batch_size ${EVAL_BATCH_SIZE}
    
    echo "✓ ${replacement} 完成"
done

echo "✓ 消融实验完成"

# ============================================================================
# 4. 超参数分析
# ============================================================================

echo "[4/5] 运行超参数分析..."

# 询问轮数分析
echo "分析询问轮数影响..."
for max_turns in 1 3 5 10; do
    python "${EXPERIMENTS_DIR}/run_hyperparams.py" \
        --param_name "max_turns" \
        --param_value ${max_turns} \
        --dataset "${RESULTS_DIR}/dataset/test_set.json" \
        --output "${RESULTS_DIR}/hyperparams/max_turns_${max_turns}.json" \
        --config "${ONCALL_ROOT}/config.json"
done

# 最大长度分析
echo "分析最大长度影响..."
for max_length in 256 512 1024 2048; do
    python "${EXPERIMENTS_DIR}/run_hyperparams.py" \
        --param_name "max_length" \
        --param_value ${max_length} \
        --dataset "${RESULTS_DIR}/dataset/test_set.json" \
        --output "${RESULTS_DIR}/hyperparams/max_length_${max_length}.json" \
        --config "${ONCALL_ROOT}/config.json"
done

# 置信度阈值分析
echo "分析置信度阈值影响..."
for confidence in 0.6 0.7 0.8 0.9; do
    python "${EXPERIMENTS_DIR}/run_hyperparams.py" \
        --param_name "confidence_threshold" \
        --param_value ${confidence} \
        --dataset "${RESULTS_DIR}/dataset/test_set.json" \
        --output "${RESULTS_DIR}/hyperparams/confidence_${confidence}.json" \
        --config "${ONCALL_ROOT}/config.json"
done

# 温度参数分析
echo "分析温度参数影响..."
for temperature in 0.1 0.3 0.7 1.0; do
    python "${EXPERIMENTS_DIR}/run_hyperparams.py" \
        --param_name "temperature" \
        --param_value ${temperature} \
        --dataset "${RESULTS_DIR}/dataset/test_set.json" \
        --output "${RESULTS_DIR}/hyperparams/temperature_${temperature}.json" \
        --config "${ONCALL_ROOT}/config.json"
done

# 检索Top-K分析
echo "分析检索Top-K影响..."
for top_k in 3 5 10 20; do
    python "${EXPERIMENTS_DIR}/run_hyperparams.py" \
        --param_name "retrieval_top_k" \
        --param_value ${top_k} \
        --dataset "${RESULTS_DIR}/dataset/test_set.json" \
        --output "${RESULTS_DIR}/hyperparams/top_k_${top_k}.json" \
        --config "${ONCALL_ROOT}/config.json"
done

echo "✓ 超参数分析完成"

# ============================================================================
# 5. 开源模型对比实验
# ============================================================================

echo "[5/6] 运行开源模型对比实验..."

python "${EXPERIMENTS_DIR}/run_model_comparison.py" \
    --dataset "${RESULTS_DIR}/dataset/test_set.json" \
    --output "${RESULTS_DIR}/analysis/model_comparison.json" \
    --config "${ONCALL_ROOT}/config.json" \
    --models "Qwen/Qwen2.5-7B-Instruct" "THUDM/chatglm3-6b" \
    --batch_size 3

echo "✓ 模型对比实验完成"

# ============================================================================
# 6. 结果分析和报告生成
# ============================================================================

echo "[6/6] 生成实验报告..."

python "${EXPERIMENTS_DIR}/analyze_results.py" \
    --results_dir "${RESULTS_DIR}" \
    --output "${RESULTS_DIR}/final_report.html" \
    --dataset_info "${RESULTS_DIR}/dataset/dataset_info.json"

echo "✓ 实验报告生成完成"

# ============================================================================
# 实验完成
# ============================================================================

echo "=========================================="
echo "OnCallAgent 实验验证完成"
echo "时间: $(date)"
echo "结果保存在: ${RESULTS_DIR}"
echo "报告文件: ${RESULTS_DIR}/final_report.html"
echo "日志文件: ${LOG_FILE}"
echo "=========================================="

# 发送通知（可选）
if command -v osascript &> /dev/null; then
    osascript -e 'display notification "OnCallAgent实验验证完成" with title "实验通知"'
fi

# 统计实验用时
end_time=$(date +%s)
if [[ -n "${start_time}" ]]; then
    duration=$((end_time - start_time))
    echo "总用时: $((duration / 3600))小时 $(((duration % 3600) / 60))分钟 $((duration % 60))秒"
fi
