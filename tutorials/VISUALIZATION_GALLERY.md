# MAS4TS 图表画廊

本文档展示所有生成的图表及其用途。

---

## 📊 已生成的图表 (9组，18个文件)

### 1. 方法对比图 ✅

**文件**: 
- `comparison_methods.png` (710KB, 300 DPI)
- `comparison_methods.pdf` (47KB, 矢量图)

**内容**:
- 三种方法架构对比
- 能力雷达图（6维度）
- 性能柱状图（5数据集）
- 平均提升12.3%

**论文位置**: Introduction, Method
**LaTeX引用**: `\ref{fig:comparison}`

---

### 2. 框架架构图 ✅

**文件**:
- `framework.png` (732KB)
- `framework.pdf` (53KB)

**内容**:
- 4个Agent协作流程
- Manager协调机制
- 5个下游应用
- 3个核心创新标注

**论文位置**: Method (主要架构)
**LaTeX引用**: `\ref{fig:framework}`

---

### 3. 长短期预测展示 ✅

**文件**:
- `showcase_forecasting.png` (1.7MB)
- `showcase_forecasting.pdf` (80KB)

**内容**:
- 6个预测场景（ETTh1/ETTm1/Weather × 96/192步）
- 真实值 vs MAS4TS vs Baseline
- 95%置信区间
- MSE和改进百分比

**论文位置**: Experiments (主要结果)
**LaTeX引用**: `\ref{fig:forecasting}`

---

### 4. 分类任务展示 ✅

**文件**:
- `showcase_classification.png` (979KB)
- `showcase_classification.pdf` (51KB)

**内容**:
- 不同类别时序样本
- 4×4混淆矩阵
- 准确率对比（94.5%）
- Per-class F1分数

**论文位置**: Experiments
**LaTeX引用**: `\ref{fig:classification}`

---

### 5. 插值任务展示 ✅

**文件**:
- `showcase_imputation.png` (1.6MB)
- `showcase_imputation.pdf` (82KB)

**内容**:
- 3种缺失模式（随机、块状、突发）
- 方法对比（Linear/Mean/MAS4TS）
- MSE性能对比
- 缺失区域可视化

**论文位置**: Experiments
**LaTeX引用**: `\ref{fig:imputation}`

---

### 6. 异常检测展示 ✅

**文件**:
- `showcase_anomaly.png` (927KB)
- `showcase_anomaly.pdf` (53KB)

**内容**:
- 3种异常类型（点、上下文、集体）
- 检测结果（TP/FP标注）
- Precision/Recall/F1对比
- 方法性能对比

**论文位置**: Experiments
**LaTeX引用**: `\ref{fig:anomaly}`

---

### 7. 参数敏感性分析 ✅

**文件**:
- `parameter_study.png` (819KB)
- `parameter_study.pdf` (43KB)

**内容**:
- 6个关键参数的敏感性曲线
- (a) Top-K特征选择 → 最优K=10
- (b) VLM温度 → 最优T=0.3
- (c) LLM Ensemble → 最优N=3
- (d) 置信水平 vs 区间质量
- (e) Batch并行效率
- (f) 锚点策略对比

**论文位置**: Experiments (参数选择)
**LaTeX引用**: `\ref{fig:parameters}`

---

### 8. 消融实验分析 ✅

**文件**:
- `ablation_study.png` (678KB)
- `ablation_study.pdf` (52KB)

**内容**:
- 逐步添加组件的性能提升
- VLM模型选择（5个模型对比）
- LLM模型选择（6个模型对比）
- 融合策略对比（5种策略）
- 组件重要性（SHAP值风格）

**论文位置**: Experiments (验证设计)
**LaTeX引用**: `\ref{fig:ablation}`

---

### 9. 视觉锚定过程 ✅

**文件**:
- `visual_anchoring.png` (1.1MB)
- `visual_anchoring.pdf` (54KB)

**内容**:
- (a) 原始时序数据
- (b) VLM分析的纯视觉图
- (c) 生成的锚点和置信区间
- (d) 有无锚点的预测对比

**论文位置**: Method (核心创新)
**LaTeX引用**: `\ref{fig:anchor}`

---

## 🎨 图表质量

### 分辨率
- **PNG**: 300 DPI（适合打印和投稿）
- **PDF**: 矢量图（无损缩放）

### 尺寸
- 大部分图表: 16×10或16×12 inches
- 适合双栏论文的全页宽度

### 颜色方案
- 主色调：蓝、红、绿、紫、橙
- 符合色盲友好原则
- 打印效果良好

---

## 🔄 更新图表

### 使用真实实验数据

1. 修改脚本中的数据数组：
```python
# 在fig_comparison.py中
mas4ts_mse = [0.352, 0.338, 0.441, ...]  # 改为实际值
```

2. 重新运行脚本：
```bash
/data/sony/anaconda3/envs/MAS4TS/bin/python tutorials/fig_comparison.py
```

### 批量更新

```bash
# 修改所有脚本中的数据
# 然后运行
./tutorials/generate_all.sh
```

---

## 📐 LaTeX集成

### 图片导入模板

```latex
% 单栏图
\begin{figure}[t]
    \centering
    \includegraphics[width=\columnwidth]{tutorials/framework.pdf}
    \caption{MAS4TS framework architecture.}
    \label{fig:framework}
\end{figure}

% 双栏图
\begin{figure*}[t]
    \centering
    \includegraphics[width=\textwidth]{tutorials/comparison_methods.pdf}
    \caption{Method comparison...}
    \label{fig:comparison}
\end{figure*}
```

### 子图引用

```latex
...as shown in Figure~\ref{fig:comparison}(a)...
...the VLM analysis (Figure~\ref{fig:anchor}(b))...
```

---

## 💡 提示和技巧

### 调整图表尺寸
```python
fig, axes = plt.subplots(2, 3, figsize=(width, height))
```

### 修改颜色
在脚本开头查找颜色定义：
```python
colors = ['#2E86AB', '#F77F00', ...]
```

### 提高分辨率
```python
plt.savefig('output.png', dpi=400)  # 从300提高到400
```

### 去除emoji警告
将emoji字符替换为文字：
```python
# 修改前
ax.text(x, y, '🖼️  VLM')

# 修改后  
ax.text(x, y, '[VLM]')
```

---

## 🎯 论文投稿建议

### 图表数量控制

**会议论文** (页数限制8-10页):
- 推荐5-6张图
- 必须: Framework, Comparison, Main Results
- 可选: Ablation或Parameter Study

**期刊论文** (页数限制较宽松):
- 推荐7-9张图
- 全部包含，展示完整工作

### 图表顺序建议

1. Introduction: Comparison (架构对比)
2. Method: Framework → Visual Anchoring
3. Experiments: Forecasting → Ablation → Parameter Study
4. 补充: 其他任务展示

---

## 📦 打包发布

### 论文投稿包

```bash
# 创建投稿文件夹
mkdir paper_submission
cp tutorials/*.pdf paper_submission/

# 压缩
tar -czf mas4ts_figures.tar.gz paper_submission/
```

### 宣传材料

```bash
# 使用高分辨率PNG
cp tutorials/*.png presentation/
```

---

**生成完成！所有图表ready for使用！** 🎊📊

查看图表：
```bash
cd /data/sony/VQualA2025/rwl/MAS4TS/tutorials
ls -lh *.png *.pdf
```

