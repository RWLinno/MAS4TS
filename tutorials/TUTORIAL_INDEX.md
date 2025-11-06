# MAS4TS 教程和可视化索引

## 📚 目录概览

本目录包含MAS4TS系统的所有教程、文档和可视化脚本。

---

## 🎨 论文图表生成脚本

### 核心图表（必需）

| 脚本文件 | 描述 | 输出文件 | 论文位置 |
|---------|------|---------|---------|
| `fig_comparison.py` | 方法对比图 | `comparison_methods.png/pdf` | Method, Introduction |
| `fig_framework.py` | 框架架构图 | `framework.png/pdf` | Method |
| `fig_anchor.py` | 视觉锚定过程 | `visual_anchoring.png/pdf` | Method (核心创新) |

### 实验结果图表

| 脚本文件 | 描述 | 输出文件 | 论文位置 |
|---------|------|---------|---------|
| `fig_showcase_forecasting.py` | 长短期预测展示 | `showcase_forecasting.png/pdf` | Experiments |
| `fig_showcase_classification.py` | 分类任务展示 | `showcase_classification.png/pdf` | Experiments |
| `fig_showcase_imputation.py` | 插值任务展示 | `showcase_imputation.png/pdf` | Experiments |
| `fig_showcase_anomaly.py` | 异常检测展示 | `showcase_anomaly.png/pdf` | Experiments |

### 分析图表

| 脚本文件 | 描述 | 输出文件 | 论文位置 |
|---------|------|---------|---------|
| `fig_parameter_study.py` | 参数敏感性分析 | `parameter_study.png/pdf` | Experiments |
| `fig_ablation.py` | 消融实验分析 | `ablation_study.png/pdf` | Experiments |

---

## 🚀 快速开始

### 一键生成所有图表

**方法1: 使用Shell脚本**
```bash
cd /data/sony/VQualA2025/rwl/MAS4TS
./tutorials/RUN_ALL.sh
```

**方法2: 使用Python脚本**
```bash
cd /data/sony/VQualA2025/rwl/MAS4TS
python tutorials/generate_all_figures.py
```

### 单独生成某个图表

```bash
python tutorials/fig_comparison.py
python tutorials/fig_framework.py
python tutorials/fig_anchor.py
```

---

## 📖 文档清单

### 系统文档

| 文档文件 | 描述 | 目标读者 |
|---------|------|---------|
| `QUICK_START.md` | 快速入门指南 | 新用户 |
| `MULTI_AGENT_IMPLEMENTATION.md` | 多智能体实现细节 | 开发者 |
| `BUG_FIX_SUMMARY.md` | Bug修复记录 | 维护者 |

### 配置文档

| 文档文件 | 描述 |
|---------|------|
| `CONFIG_GUIDE.md` | 配置使用指南 |
| `CONFIG_STRUCTURE.md` | 配置结构说明 |

### 图表文档

| 文档文件 | 描述 |
|---------|------|
| `README_FIGURES.md` | 图表生成详细指南 |
| `TUTORIAL_INDEX.md` | 本文件 |

---

## 🎯 论文写作建议

### Introduction部分

**建议图表**:
1. `framework.png` - 系统概览
2. `comparison_methods.png` - 与现有方法对比

**重点**:
- 强调多智能体协作
- 突出视觉-语义锚定创新

### Method部分

**建议图表**:
1. `framework.png` - 完整架构
2. `visual_anchoring.png` - 核心创新机制
3. `comparison_methods.png` (a-c部分) - 架构对比

**重点**:
- 详细说明4个Agent的职责
- 解释VLM和LLM如何协作
- 展示数据流和决策过程

### Experiments部分

**建议图表**:
1. 所有`showcase_*.png` - 不同任务的效果
2. `parameter_study.png` - 参数选择依据
3. `ablation_study.png` - 组件有效性验证
4. `comparison_methods.png` (性能对比部分) - 整体性能

**重点**:
- 多任务性能优势
- 参数设置合理性
- 各组件的贡献

### Conclusion部分

**建议图表**:
1. `comparison_methods.png` (雷达图) - 多维度优势总结

---

## 🎨 图表样式指南

### 颜色方案

**主要颜色**:
- 蓝色系 `#2E86AB`, `#89CFF0` - 数据、真实值
- 红色系 `#D62828`, `#FF6B6B` - MAS4TS结果
- 绿色系 `#06A77D`, `#C7F9CC` - Visual Anchor
- 紫色系 `#7209B7`, `#C084FC` - LLM相关
- 橙色系 `#F77F00`, `#FFD6A5` - Data Analyzer

**Agent颜色编码**:
- Data Analyzer: 橙色 `#F77F00`
- Visual Anchor: 绿色 `#06A77D`
- Numeric Adapter: 蓝色 `#4361EE`
- Task Executor: 紫色 `#9D4EDD`
- Manager: 红色 `#D62828`

### 字体设置

```python
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 9
```

### 图表尺寸

- 全页图: `figsize=(16, 10)` 或 `(16, 12)`
- 子图多的: `figsize=(14, 10)`
- 简单对比: `figsize=(12, 6)`

---

## 📊 图表质量标准

### 分辨率

- **PNG**: 300 DPI (论文投稿标准)
- **PDF**: 矢量图（无损缩放）

### 元素要求

- ✅ 清晰的标题
- ✅ 轴标签和单位
- ✅ 图例（位置合理）
- ✅ 网格线（alpha=0.3）
- ✅ 性能指标标注
- ✅ 颜色对比明显
- ✅ 字号适中（8-12pt）

---

## 🔧 自定义修改

### 修改实验数据

如果你有真实的实验结果，可以这样修改：

```python
# 在fig_comparison.py中
# 修改这些数值为你的实验结果
ts_specific_mse = [0.42, 0.38, 0.52, 0.45, 0.58]  # 改为实际值
pretrained_lm_mse = [0.39, 0.36, 0.48, 0.43, 0.54]
mas4ts_mse = [0.35, 0.32, 0.44, 0.39, 0.49]
```

### 添加新的数据集

```python
datasets = ['ETTh1', 'ETTm1', 'Weather', 'YourDataset']  # 添加新数据集
# 对应添加数据
```

### 修改配色

在脚本开头修改颜色变量即可。

---

## 📝 论文LaTeX引用示例

```latex
\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{figures/framework.pdf}
    \caption{The MAS4TS framework architecture showing the four-agent collaborative process.}
    \label{fig:framework}
\end{figure}

\begin{figure*}[t]
    \centering
    \includegraphics[width=\textwidth]{figures/comparison_methods.pdf}
    \caption{Comparison between MAS4TS and existing methods: (a) Time-series specific methods, (b) Pre-trained LMs for TS, (c) MAS4TS multi-agent system.}
    \label{fig:comparison}
\end{figure*}
```

---

## ✅ 检查清单

生成图表前确认：

- [ ] 已安装matplotlib, numpy等依赖
- [ ] 数据值已更新为实际实验结果
- [ ] 颜色方案符合期刊要求
- [ ] 图表尺寸适合论文版面
- [ ] 所有文字可读（字号合适）
- [ ] PDF格式正常（用于投稿）
- [ ] PNG格式清晰（用于演示）

---

## 🎓 论文投稿建议

### 图表数量

**建议最多使用**:
- 主要方法图: 1-2张（framework, comparison）
- 实验结果: 2-3张（选择最有代表性的任务）
- 分析图表: 1-2张（ablation或parameter study）

**总计**: 5-7张图表

### 图表顺序

1. Framework（系统概览）
2. Visual Anchoring（核心创新）
3. Performance Comparison（整体性能）
4. Task-specific Results（选1-2个任务详细展示）
5. Ablation Study（验证设计）

### 图表说明（Caption）要点

- 简洁明了地说明图表内容
- 指出关键观察点
- 解释缩写和符号
- 标注子图(a), (b), (c)等

---

## 📞 技术支持

如遇到问题：
1. 检查Python环境和依赖包
2. 查看脚本中的注释说明
3. 参考`README_FIGURES.md`详细指南

---

**最后更新**: 2025-11-05
**版本**: v1.0

