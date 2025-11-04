# Abstract

时序分析对于xx/xx等领域至关重要。最近基于预训练大语言模型的工作相较于专用时序模型得到了更强的泛化和上下文能力。然而依然面临下面挑战 1）效率低下; 2) 任务单一。To address these challenges, we proposed MAS4TS, utilizing multi-agent collaboration for general time-series tasks, including classification, forecasting, imputation and anomaly detection.  Specially, 1) Instead of token利用率低下且推理缓慢的单一模型 , 我们部署多个任务专用模型并且并发推理的方式，并由统一的maneger进行调度和决策。2）我们设计了视觉锚定agent观察时序数据图表，生成历史和未来数据的“锚点”，并且生成语义先验。3 )运用锚点+时序数据+语义的形式，由numerologic_adapter得到精确的数值推理。Comprehensive experiments in xxx数据集表明我们的方法在xx任务上均取得了SOTA成绩，同时兼顾强泛化能力和高推理效率。

# Introduction

(第一段介绍Time-series analysis对实际应用场景的意义)

(第二段介绍传统方法到DeepLearning方法的过渡和现在使用LLM的先进工作，然而对LLM对于时序来说存在xxx的challenge)

(第三段讲Multi-Agents的发展对xxxx领域的significance，然而对时序工作来讲，存在xxx的gap)

challenges:

(第四段讲我们的工作弥补了这个gap和解决了以上困难，具体来讲我们先…)

(然后总结我们的贡献)

```jsx
可能存在的一些challenges:
**Challenge1: The Efficiency-Semantics Trade-off.**

LLM方法有强大的语义理解能力（可以理解任务描述、领域知识），但计算成本极高。专用时序模型计算高效，但缺乏语义理解，无法处理文本描述的任务或利用领域知识。现有的一些LLM4TS工作(如xx和xx罗列…) 然而Token使用效率低，推理慢，难以部署。❌

✅我们使用多个轻量化Agent，并行推理，共同决策，集成专用时序任务的工具正确。

**Challenge2: Task Generalization with Tool-Augmented Reasoning**

时序分析任务如Few-shot/Zero-shot需要强大的泛化能力，目前时序专用模型如xx and xx。虽然这些专用模型使用较低成本就能获得很好的效果，但仅限于单个任务或bench，难以泛化到多领域多任务。❌

✅我们采用动态调度Agent(决策核心)+任务增强Agent(工具调用)的思路，让多智能体协作模式主动适应多个时序任务。
```

# Related Work

### Pre-trained LMs for Time-series

### Multi-Agent System

# Methodology

### Visual Anchoring

重点讲我们怎么把历史时序转为图像然后用VLM进行区间和预测点的锚定，以及为什么要这样做

### Numerical Reasoning

如何通过锚点和多模态信息(预测图像区域/数据集特征描述和源时序数据)进行数值推导，利用语义信息的预训练模型如何高效编码数值信息，为什么要这样做。

### Multi-Agent Collaboration

重点讲多智能体之间如何协作和做决策，她们各自定义的动机以及如何优化具体任务。

# Experiments

## Setup

dataset

baselines

metrics

implementation details

## Results

### Classification

### Imputation

### Forecasting

## Ablation Study

## Efficiency

# Conclusion

xx