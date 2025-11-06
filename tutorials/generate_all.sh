#!/bin/bash

# MAS4TS 论文图表一键生成脚本
# 使用MAS4TS conda环境

cd /data/sony/VQualA2025/rwl/MAS4TS

PYTHON=/data/sony/anaconda3/envs/MAS4TS/bin/python

echo "======================================================"
echo "MAS4TS - 论文图表生成工具"
echo "======================================================"
echo ""

# 创建输出目录
mkdir -p tutorials/figures
echo "✓ 输出目录已创建"
echo ""

echo "开始生成图表..."
echo "------------------------------------------------------"

# 生成所有图表
declare -a scripts=(
    "fig_comparison.py:方法对比图"
    "fig_framework.py:框架架构图"
    "fig_showcase_forecasting.py:预测任务展示"
    "fig_showcase_classification.py:分类任务展示"
    "fig_showcase_imputation.py:插值任务展示"
    "fig_showcase_anomaly.py:异常检测展示"
    "fig_parameter_study.py:参数敏感性分析"
    "fig_ablation.py:消融实验分析"
    "fig_anchor.py:视觉锚定过程"
    "fig_efficiency_study.py:效率研究对比图"
)

success=0
failed=0
total=${#scripts[@]}

for item in "${scripts[@]}"; do
    IFS=':' read -r script desc <<< "$item"
    
    ((count=success+failed+1))
    echo ""
    echo "[$count/$total] 生成 $desc..."
    
    if $PYTHON tutorials/$script 2>&1 | grep -q "✓"; then
        echo "  ✓ 成功"
        ((success++))
    else
        echo "  ✗ 失败"
        ((failed++))
    fi
done

echo ""
echo "======================================================"
echo "图表生成完成！"
echo "  成功: $success/$total"
echo "  失败: $failed/$total"
echo "======================================================"
echo ""

# 列出生成的文件
if [ $success -gt 0 ]; then
    echo "生成的图表："
    echo "------------------------------------------------------"
    ls -1 tutorials/*.png 2>/dev/null | head -20
    echo ""
fi

echo "所有图表已保存到 tutorials/ 目录"
echo "======================================================"

