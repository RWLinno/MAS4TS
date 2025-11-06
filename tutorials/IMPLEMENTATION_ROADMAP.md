# MAS4TS 实现路线图

## ✅ 第一阶段：核心架构（已完成）

### 1.1 模型入口
- ✅ `models/MAS4TS.py`: 真正的Multi-Agent系统调用
- ✅ 不再是简单神经网络，而是协调4个Agent

### 1.2 执行流程
- ✅ `src/agents/manager_agent.py`: 定义完整的4-Agent执行计划
- ✅ Stage 1: Data Analyzer
- ✅ Stage 2: Visual Anchor  
- ✅ Stage 3: Numerical Adapter
- ✅ Stage 4: Task Executor

### 1.3 基础设施
- ✅ Agent基类和消息传递
- ✅ 数据预处理Pipeline
- ✅ 配置管理系统

## 🚧 第二阶段：Agent实现（进行中）

### 2.1 Data Analyzer Agent

**当前状态**: 基础功能已实现

**需要添加**:
```python
# 文件: src/agents/data_analyzer.py

# ✅ 已有: 统计特征提取
# ✅ 已有: 趋势分析
# ✅ 已有: 异常检测

# 🚧 需要添加:
def _generate_plot(self, data, batch_idx):
    """生成时序plot图并保存"""
    
def _generate_statistics_text(self, features):
    """生成统计描述文本"""
```

**输出**:
- `data_features`: Dict[str, torch.Tensor]
- `plot_path`: str (./visualizations/data_analysis/batch_X.png)
- `statistics_text`: str

### 2.2 Visual Anchor Agent

**当前状态**: 基础规则方法已实现

**需要添加**:
```python
# 文件: src/agents/visual_anchor.py

# ✅ 已有: 规则based锚点生成
# ✅ 已有: 置信区间计算

# 🚧 需要添加:
def _init_local_vlm(self):
    """初始化Qwen-VL模型"""
    
async def _call_vlm(self, image_path, statistics_text):
    """调用VLM分析时序图像"""
    
async def _call_eas_vlm(self, image_path, prompt):
    """调用EAS在线VLM服务"""
```

**VLM Prompt示例**:
```
Analyze this time series plot.

Statistics:
- Mean: 0.5234
- Trend: increasing
- Volatility: medium

Predict the next 96 steps:
1. Expected value range
2. Confidence level
3. Key anchor points

Output as JSON.
```

**输出**:
- `visual_anchors`: Dict (含range, confidence, anchor_points)
- `anchor_image_path`: str (保存带标注的图)

### 2.3 Numerical Adapter Agent

**当前状态**: 简单融合已实现

**需要添加**:
```python
# 文件: src/agents/numerologic_adapter.py

# ✅ 已有: 多模态特征融合
# ✅ 已有: 注意力机制

# 🚧 需要添加:
async def _parallel_llm_inference(self, prompt, num_models=3):
    """并发调用3个LLM模型"""
    
async def _call_single_llm(self, model_name, prompt):
    """调用单个LLM"""
    
def _ensemble_predictions(self, results):
    """ensemble多个LLM的结果"""
```

**LLM Prompt示例**:
```
Task: Numerical reasoning for time series forecasting

Visual Analysis:
- Range: [0.45, 0.62]
- Trend: increasing
- Anchors: [t=10, t=30, t=50]

Data Statistics:
- Mean: 0.5234
- Std: 0.0823
- Slope: 0.0012

Refine predictions with numerical reasoning.
Output JSON with predictions and confidence.
```

**输出**:
- `numerical_predictions`: torch.Tensor or Dict
- `confidence_intervals`: torch.Tensor

### 2.4 Task Executor Agent

**当前状态**: 基础实现完成

**需要确保**:
- ✅ 接收所有上游Agent的输出
- ✅ 根据任务类型执行
- ✅ 应用约束条件

## 🎯 第三阶段：VLM/LLM集成

### 3.1 本地部署选项

**Qwen-VL (视觉)**:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-VL-Chat",
    device_map="auto",
    trust_remote_code=True
).eval()
```

**Qwen-7B/14B/72B (文本)**:
```python
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-14B-Chat",
    device_map="auto"
).eval()
```

### 3.2 EAS在线服务选项

创建 `src/utils/eas_client.py`:
```python
class EASClient:
    def __init__(self, endpoint, token):
        self.endpoint = endpoint
        self.token = token
    
    async def call_vlm(self, image_base64, prompt):
        """调用VLM服务"""
        
    async def call_llm(self, prompt, model_name):
        """调用LLM服务"""
```

## 🔧 第四阶段：完善与优化

### 4.1 可视化系统
- ✅ 目录结构已定义
- 🚧 实现保存逻辑
- 🚧 添加结果对比图

### 4.2 配置系统
在 `run.py` 添加参数:
```bash
--use_vlm              # 使用VLM
--use_eas              # 使用EAS服务
--vlm_model            # VLM模型名称
--num_llm_models       # 并发LLM数量
--eas_endpoint         # EAS服务地址
--eas_token            # EAS认证token
```

### 4.3 性能优化
- 🚧 Agent结果缓存
- 🚧 并发执行优化
- 🚧 显存管理

## 📈 第五阶段：实验与论文

### 5.1 实验脚本
- ✅ 基础测试脚本已完成
- 🚧 VLM/LLM实验脚本
- 🚧 消融实验脚本

### 5.2 论文写作
- ✅ Abstract完成
- ✅ Introduction完成
- ✅ Related Work完成
- ✅ Methodology完成
- 🚧 Experiments需要更新实际结果
- ✅ Conclusion完成

## 🎬 快速开始

### 当前可以运行的命令

**测试基础功能**（不使用VLM/LLM）:
```bash
cd /data/sony/VQualA2025/rwl/MAS4TS
bash src/scripts/test_all_tasks.sh
```
✅ 这个可以立即运行，使用rule-based方法

**运行完整实验**（需要实现VLM/LLM）:
```bash
python run.py \
  --model MAS4TS \
  --task_name long_term_forecast \
  --use_vlm \
  --vlm_model Qwen/Qwen-VL-Chat \
  --data ETTh1
```
🚧 需要先实现VLM集成

## 📋 待办事项优先级

### P0 - 核心功能（必须）
1. [ ] Data Analyzer: 实现`_generate_plot()`
2. [ ] Visual Anchor: 集成Qwen-VL（至少rule-based版本）
3. [ ] Numerical Adapter: 实现基础LLM调用
4. [ ] 端到端测试：验证4个Agent流程

### P1 - 增强功能（重要）
5. [ ] 并发LLM推理（3个模型ensemble）
6. [ ] EAS在线服务支持
7. [ ] 可视化结果保存
8. [ ] 性能优化和缓存

### P2 - 可选功能（Nice to have）
9. [ ] 更多数据集支持
10. [ ] 更多VLM模型选项
11. [ ] Web UI界面
12. [ ] 实验结果dashboard

## 🚀 预期效果

完整实现后，运行一次预测会产生：

```
./visualizations/
├── data_analysis/
│   ├── batch_0_analysis.png        ← Data Analyzer输出
│   └── batch_0_statistics.txt
├── visual_anchors/
│   ├── batch_0_anchors.png         ← Visual Anchor输出（带标注）
│   └── batch_0_vlm_response.json
├── numerical_reasoning/
│   ├── batch_0_llm_ensemble.json   ← Numerical Adapter输出
│   └── batch_0_confidence.txt
└── final_results/
    ├── batch_0_predictions.png     ← 最终结果对比
    └── batch_0_metrics.json
```

**日志输出示例**:
```
[ManagerAgent] Starting 4-Agent pipeline
[Stage 1] Data Analysis & Visualization
  → Generated plot: ./visualizations/data_analysis/batch_0.png
  → Statistics: Mean=0.52, Trend=increasing
[Stage 2] Visual Anchoring with VLM
  → VLM analyzed image
  → Anchors: [0.45, 0.62], Confidence=0.89
[Stage 3] Numerical Reasoning with LLM Ensemble
  → Parallel LLM calls: 3 models
  → Ensemble predictions generated
[Stage 4] Task Execution
  → Final predictions: shape=[32, 96, 7]
✓ Pipeline completed in 2.3s
```

## 💡 关键设计决策

1. **为什么4个Agent？**
   - 专业分工，各司其职
   - 可解释性强，每步可视化
   - 灵活替换和优化

2. **为什么使用VLM？**
   - 时序图像包含丰富的模式信息
   - VLM能识别人类难以量化的特征
   - 提供语义先验指导数值推理

3. **为什么并发LLM？**
   - 不同规模模型有不同优势
   - Ensemble提升准确性
   - 并发执行保持高效率

4. **为什么不用纯神经网络？**
   - Multi-Agent更可解释
   - 可以利用预训练的VLM/LLM
   - Few-shot和Zero-shot能力更强

## 📞 总结

**现状**: 架构完整，基础功能可运行，VLM/LLM集成待实现

**下一步**: 实现Data Analyzer可视化 → Visual Anchor VLM → Numerical Adapter LLM

**时间估计**: 2-3天完成核心功能，1周完成全部功能

