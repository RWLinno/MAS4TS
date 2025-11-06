# MAS4TS 论文图表生成指南

## 概述

本目录包含用于生成MAS4TS论文和宣传材料所需的所有可视化图表的Python脚本。

---

## 图表清单

### 1. 方法对比图 (`fig_comparison.py`)

**用途**: 论文Method部分，展示MAS4TS与现有方法的区别

**内容**:
- (a) Time-Series Specific Methods架构
- (b) Pre-trained LMs for TS架构  
- (c) MAS4TS多智能体架构
- 能力雷达图对比
- 性能柱状图对比

**输出**: `comparison_methods.png/pdf`

**关键亮点**:
- 清晰展示三类方法的架构差异
- 多维度能力对比（准确性、可解释性、泛化性等）
- 实验性能对比

---

### 2. 框架架构图 (`fig_framework.py`)

**用途**: 论文Method部分，系统整体架构

**内容**:
- 完整的4-Agent协作流程
- Manager Agent中央调度
- 数据流向和处理阶段
- 下游应用场景展示

**输出**: `framework.png/pdf`

**关键亮点**:
- 清晰的分层结构
- 多智能体协作机制
- 统一框架支持多任务

---

### 3. 任务展示图 (`fig_showcase_*.py`)

#### 3.1 长期预测 (`fig_showcase_forecasting.py`)
- 不同预测长度的结果
- 多个数据集的表现
- 置信区间可视化

**输出**: `showcase_forecasting.png/pdf`

#### 3.2 分类任务 (`fig_showcase_classification.py`)
- 不同类别的时序样本
- 混淆矩阵
- 准确率和F1分数对比

**输出**: `showcase_classification.png/pdf`

#### 3.3 插值任务 (`fig_showcase_imputation.py`)
- 三种缺失模式（随机、块状、突发）
- 不同方法的填补结果
- MSE对比

**输出**: `showcase_imputation.png/pdf`

#### 3.4 异常检测 (`fig_showcase_anomaly.py`)
- 三种异常类型（点、上下文、集体）
- 检测效果可视化
- Precision/Recall/F1对比

**输出**: `showcase_anomaly.png/pdf`

---

### 4. 参数敏感性分析 (`fig_parameter_study.py`)

**用途**: 论文Experiments部分，参数调优

**内容**:
- (a) Top-K特征选择的影响
- (b) VLM温度参数的影响
- (c) LLM Ensemble大小的影响
- (d) 置信水平的影响
- (e) Batch并行效率
- (f) 锚点策略对比

**输出**: `parameter_study.png/pdf`

**关键亮点**:
- 6个关键参数的敏感性分析
- 最优参数配置建议
- 性能-效率权衡分析

---

### 5. 消融实验 (`fig_ablation.py`)

**用途**: 论文Experiments部分，验证各组件贡献

**内容**:
- (a) 逐步添加组件的性能提升
- (b) VLM模型选择的影响
- (c) LLM模型选择的影响
- (d) 融合策略对比
- (e) 组件重要性分析（SHAP值风格）

**输出**: `ablation_study.png/pdf`

**关键亮点**:
- 证明每个组件的必要性
- 模型选择的影响
- 协作机制的贡献

---

### 6. 视觉锚定过程 (`fig_anchor.py`)

**用途**: 论文Method部分，核心创新展示

**内容**:
- (a) 原始时序数据
- (b) VLM分析的可视化图像
- (c) 生成的锚点和置信区间
- (d) 有无锚点的预测对比

**输出**: `visual_anchoring.png/pdf`

**关键亮点**:
- 展示核心创新：视觉-语义锚定
- VLM的作用可视化
- 锚点对预测质量的提升

---

## 使用方法

### 方法1：生成所有图表

```bash
cd /data/sony/VQualA2025/rwl/MAS4TS
python tutorials/generate_all_figures.py
```

### 方法2：单独生成某个图表

```bash
python tutorials/fig_comparison.py
python tutorials/fig_framework.py
python tutorials/fig_showcase_forecasting.py
# ... 等等
```

### 方法3：在Python中导入

```python
from tutorials.fig_comparison import create_comparison_figure
from tutorials.fig_framework import create_framework_figure

# 生成特定图表
create_comparison_figure()
create_framework_figure()
```

---

## 输出文件

所有图表会生成两种格式：
- **PNG格式** (300 DPI): 用于PPT、网页等
- **PDF格式** (矢量图): 用于论文投稿

保存位置：`tutorials/` 目录

---

## 自定义设置

### 修改图表尺寸

在各个脚本中找到：
```python
fig = plt.figure(figsize=(16, 10))  # (width, height) in inches
```

### 修改DPI（分辨率）

```python
plt.savefig('output.png', dpi=300)  # 提高到400或600
```

### 修改颜色方案

在脚本开头定义的颜色变量：
```python
colors = ['#2E86AB', '#F77F00', '#06A77D', '#D62828']
```

### 修改字体

```python
plt.rcParams['font.family'] = 'DejaVu Sans'  # 或 'Arial', 'Times New Roman'
plt.rcParams['font.size'] = 10
```

---

## 依赖包

确保安装以下Python包：

```bash
pip install matplotlib numpy scipy
```

已包含在项目的`requirements.txt`中。

---

## 图表规格

### 论文投稿标准

- **格式**: PDF（矢量图）
- **DPI**: 300-600
- **尺寸**: 
  - 单栏图: 3.5 inches 宽
  - 双栏图: 7 inches 宽
  - 全页图: 7 inches 宽 × 9 inches 高

### PPT/海报使用

- **格式**: PNG
- **DPI**: 300
- **尺寸**: 根据需要调整

---

## 常见问题

### Q1: 图表显示中文乱码？
**A**: 安装中文字体：
```bash
sudo apt-get install fonts-wqy-zenhei
```
然后修改脚本：
```python
plt.rcParams['font.family'] = 'WenQuanYi Zen Hei'
```

### Q2: 生成的图表太大或太小？
**A**: 调整figsize参数：
```python
fig = plt.figure(figsize=(width, height))
```

### Q3: 需要修改数据？
**A**: 每个脚本中的数据都是示例数据，可以：
- 直接修改脚本中的数值
- 或从实验结果文件加载真实数据

---

## 论文使用建议

### Introduction
- 使用 `framework.png` 展示系统概览

### Method
- 使用 `framework.png` 详细说明架构
- 使用 `visual_anchoring.png` 解释核心创新
- 使用 `comparison_methods.png` 对比与现有方法

### Experiments
- 使用 `showcase_*.png` 展示各任务效果
- 使用 `parameter_study.png` 说明参数选择
- 使用 `ablation_study.png` 验证设计选择

### Conclusion
- 使用 `comparison_methods.png` 中的性能对比图

---

## 更新日志

- **2025-11-05**: 创建所有基础图表脚本
- **v1.0**: 初始版本，支持9种图表类型

---

## 联系方式

如需修改图表或添加新图表，请参考现有脚本的代码结构。

**Happy Publishing! 📊🎓**

