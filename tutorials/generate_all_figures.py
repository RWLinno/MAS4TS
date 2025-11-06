"""
生成所有论文图表的主脚本
一键运行生成所有可视化图表
"""

import os
import sys

# 确保matplotlib使用Agg后端（无GUI）
import matplotlib
matplotlib.use('Agg')

print("="*60)
print("MAS4TS - 论文图表生成工具")
print("="*60)
print()

# 创建输出目录
os.makedirs('tutorials/figures', exist_ok=True)
print("✓ Output directory created: tutorials/figures/")
print()

# 图表生成脚本列表
figure_scripts = [
    ('fig_comparison.py', '方法对比图'),
    ('fig_framework.py', '框架架构图'),
    ('fig_showcase_forecasting.py', '预测任务展示'),
    ('fig_showcase_classification.py', '分类任务展示'),
    ('fig_showcase_imputation.py', '插值任务展示'),
    ('fig_showcase_anomaly.py', '异常检测展示'),
    ('fig_parameter_study.py', '参数敏感性分析'),
    ('fig_ablation.py', '消融实验分析'),
    ('fig_anchor.py', '视觉锚定过程'),
    ('fig_efficiency_study.py', '效率研究对比图')
]

print("开始生成图表...")
print("-"*60)

success_count = 0
failed_count = 0

for script_name, description in figure_scripts:
    try:
        print(f"\n[{success_count + failed_count + 1}/{len(figure_scripts)}] 生成 {description} ({script_name})...")
        
        # 执行脚本
        script_path = f'tutorials/{script_name}'
        if os.path.exists(script_path):
            exec(open(script_path).read())
            success_count += 1
            print(f"  ✓ 成功生成")
        else:
            print(f"  ✗ 文件不存在: {script_path}")
            failed_count += 1
            
    except Exception as e:
        print(f"  ✗ 生成失败: {e}")
        failed_count += 1

print()
print("="*60)
print(f"图表生成完成！")
print(f"  成功: {success_count}/{len(figure_scripts)}")
print(f"  失败: {failed_count}/{len(figure_scripts)}")
print("="*60)
print()

print("生成的图表文件：")
print("-"*60)
figure_files = [
    'comparison_methods.png/pdf',
    'framework.png/pdf',
    'showcase_forecasting.png/pdf',
    'showcase_classification.png/pdf',
    'showcase_imputation.png/pdf',
    'showcase_anomaly.png/pdf',
    'parameter_study.png/pdf',
    'ablation_study.png/pdf',
    'visual_anchoring.png/pdf',
    'efficiency_study.png/pdf'
]

for fig_file in figure_files:
    print(f"  📊 tutorials/{fig_file}")

print()
print("所有图表已保存到 tutorials/ 目录")
print("="*60)

