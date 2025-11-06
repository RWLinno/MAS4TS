#!/bin/bash

# MAS4TS 论文图表一键生成脚本
# 生成所有可视化图表

echo "======================================================"
echo "MAS4TS - 论文图表生成工具"
echo "======================================================"
echo ""

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "错误: 未找到Python"
    exit 1
fi

echo "Python版本:"
python --version
echo ""

# 创建输出目录
mkdir -p tutorials/figures
echo "✓ 输出目录已创建: tutorials/figures/"
echo ""

# 图表列表
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
)

echo "开始生成图表..."
echo "------------------------------------------------------"

success=0
failed=0
total=${#scripts[@]}

for item in "${scripts[@]}"; do
    IFS=':' read -r script desc <<< "$item"
    
    ((count=success+failed+1))
    echo ""
    echo "[$count/$total] 生成 $desc ($script)..."
    
    if [ -f "tutorials/$script" ]; then
        if python "tutorials/$script" 2>&1; then
            echo "  ✓ 成功生成"
            ((success++))
        else
            echo "  ✗ 生成失败"
            ((failed++))
        fi
    else
        echo "  ✗ 文件不存在: tutorials/$script"
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

if [ $success -gt 0 ]; then
    echo "生成的图表文件："
    echo "------------------------------------------------------"
    ls -lh tutorials/*.png 2>/dev/null | awk '{print "  📊 " $9 " (" $5 ")"}'
    ls -lh tutorials/*.pdf 2>/dev/null | awk '{print "  📄 " $9 " (" $5 ")"}'
    echo ""
fi

echo "所有图表已保存到 tutorials/ 目录"
echo "======================================================"

exit 0

