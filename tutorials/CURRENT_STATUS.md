# MAS4TS 当前实现状态

## ✅ 已完成

### 1. 核心架构
- ✅ `models/MAS4TS.py`: 真正调用Multi-Agent系统，不是简单的神经网络
- ✅ `src/agents/manager_agent.py`: 完整的4-Agent执行计划
- ✅ `src/agents/base_agent_ts.py`: Agent基类
- ✅ `src/base/processor.py`: 数据预处理器

### 2. Multi-Agent流程（已规划）

```
输入数据 (x_enc)
    ↓
[Stage 1] Data Analyzer Agent
    → 分析数据趋势、统计信息
    → 生成plot图 (保存到 ./visualizations/data_analysis/)
    → 输出: data_features, plot_path, statistics_text
    ↓
[Stage 2] Visual Anchor Agent
    → 读取plot图
    → 调用VLM (Qwen-VL) 分析图像
    → 生成锚点和置信区间
    → 保存带标注的图片
    → 输出: visual_anchors, anchor_image_path
    ↓
[Stage 3] Numerical Adapter Agent
    → 使用LLM进行数值推理
    → 并发调用3个模型 (qwen-7b, qwen-14b, qwen-72b)
    → ensemble结果
    → 输出: numerical_predictions, confidence_intervals
    ↓
[Stage 4] Task Executor Agent
    → 整合所有信息
    → 根据任务类型输出最终结果
    → 输出: final_predictions
    ↓
最终输出
```

## 🚧 需要完成的实现

### 1. Data Analyzer Agent - 添加可视化功能

**文件**: `src/agents/data_analyzer.py`

**需要添加**:
```python
import matplotlib.pyplot as plt
import os
from pathlib import Path

async def process(self, input_data):
    # ... 现有代码 ...
    
    # 添加可视化
    if task == 'full_analysis_with_plot':
        plot_path = self._generate_plot(data, batch_idx=0)
        statistics_text = self._generate_statistics_text(features)
        result['plot_path'] = plot_path
        result['statistics_text'] = statistics_text

def _generate_plot(self, data, batch_idx=0):
    """生成时序plot图"""
    save_dir = Path('./visualizations/data_analysis/')
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f'batch_{batch_idx}_analysis.png'
    
    fig, ax = plt.subplots(figsize=(12, 6))
    data_np = data[0].cpu().numpy()  # 取第一个样本
    for i in range(min(3, data.shape[2])):
        ax.plot(data_np[:, i], label=f'Feature {i}', linewidth=2)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Value')
    ax.set_title('Time Series Data Analysis')
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return str(save_path)

def _generate_statistics_text(self, features):
    """生成统计描述"""
    text = "Time Series Statistics:\n"
    text += f"- Mean: {features['mean'].mean().item():.4f}\n"
    text += f"- Std: {features['std'].mean().item():.4f}\n"
    text += f"- Min: {features['min'].mean().item():.4f}\n"
    text += f"- Max: {features['max'].mean().item():.4f}\n"
    text += f"- Trend Slope: {features['trend'].mean().item():.6f}\n"
    return text
```

### 2. Visual Anchor Agent - 集成VLM

**文件**: `src/agents/visual_anchor.py`

**需要添加**:
```python
from PIL import Image
import json

# 在__init__中添加VLM配置
def __init__(self, config):
    super().__init__("VisualAnchorAgent", config)
    self.use_vlm = config.get('use_vlm', False)
    self.use_eas = config.get('use_eas', False)
    self.vlm_model_name = config.get('vlm_model', 'qwen-vl')
    
    if self.use_vlm and not self.use_eas:
        self._init_local_vlm()

def _init_local_vlm(self):
    """初始化本地VLM模型"""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.vlm_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen-VL-Chat",
            device_map="auto",
            trust_remote_code=True
        ).eval()
        self.vlm_tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen-VL-Chat",
            trust_remote_code=True
        )
        self.log_info("VLM model loaded successfully")
    except Exception as e:
        self.log_error(f"Failed to load VLM: {e}")
        self.use_vlm = False

async def _extract_semantic_priors(self, plot_path, statistics_text, task):
    """使用VLM提取语义先验"""
    if not self.use_vlm:
        return self._extract_rule_based_priors(data, task)
    
    prompt = f"""Analyze this time series plot for {task} task.

{statistics_text}

Based on the plot, provide:
1. Trend direction (increasing/decreasing/stable)
2. Volatility level (low/medium/high)
3. Confidence interval width (narrow/medium/wide)
4. Key anchor points for future predictions

Format as JSON."""

    if self.use_eas:
        response = await self._call_eas_vlm(plot_path, prompt)
    else:
        image = Image.open(plot_path)
        query = self.vlm_tokenizer.from_list_format([
            {'image': plot_path},
            {'text': prompt}
        ])
        response, _ = self.vlm_model.chat(
            self.vlm_tokenizer, 
            query=query, 
            history=None
        )
    
    return self._parse_vlm_response(response)
```

### 3. Numerical Adapter Agent - 并发LLM推理

**文件**: `src/agents/numerologic_adapter.py`

**需要添加**:
```python
import asyncio

async def process(self, input_data):
    # ... 现有代码 ...
    
    if input_data.get('use_parallel_llm', False):
        num_models = input_data.get('num_llm_models', 3)
        predictions = await self._parallel_llm_inference(
            visual_anchors, 
            data_features,
            num_models
        )
        result['numerical_predictions'] = predictions

async def _parallel_llm_inference(self, anchors, features, num_models=3):
    """并发调用多个LLM"""
    prompt = self._build_numerical_reasoning_prompt(anchors, features)
    
    tasks = []
    models = ['qwen-7b', 'qwen-14b', 'qwen-72b'][:num_models]
    
    for model_name in models:
        tasks.append(self._call_single_llm(model_name, prompt))
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 过滤错误并ensemble
    valid_results = [r for r in results if not isinstance(r, Exception)]
    if valid_results:
        return self._ensemble_predictions(valid_results)
    else:
        # Fallback
        return self._rule_based_prediction(anchors, features)

def _build_numerical_reasoning_prompt(self, anchors, features):
    """构建LLM推理prompt"""
    prompt = f"""Task: Time series numerical reasoning

Visual Anchors:
- Expected range: [{anchors['lower_bound']}, {anchors['upper_bound']}]
- Trend direction: {anchors.get('trend_direction', 'unknown')}
- Anchor points: {anchors.get('key_points', [])}

Data Features:
- Mean: {features['mean']}
- Std: {features['std']}
- Trend slope: {features['trend']}

Perform numerical reasoning to:
1. Refine the prediction range
2. Identify specific anchor values
3. Provide confidence scores

Output JSON format:
{{
  "predictions": [value1, value2, ...],
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}}"""
    return prompt
```

### 4. Task Executor Agent - 整合所有输出

**文件**: `src/agents/task_executor.py`

**已基本完成**，需要确保能接收并使用：
- `numerical_predictions` from Numerical Adapter
- `visual_anchors` from Visual Anchor
- `confidence_intervals` from Numerical Adapter

## 📝 Prompt设计（已完成）

详见 `MULTI_AGENT_IMPLEMENTATION.md`

## 🎯 运行方式

### 简单模式（不使用VLM/LLM）
```bash
bash src/scripts/test_all_tasks.sh
```
- 使用rule-based方法替代VLM/LLM
- 适合快速测试

### 完整模式（使用本地VLM）
```bash
python run.py \
  --model MAS4TS \
  --task_name long_term_forecast \
  --use_vlm \
  --vlm_model Qwen/Qwen-VL-Chat \
  --data ETTh1 \
  --seq_len 96 \
  --pred_len 96
```

### 完整模式（使用EAS在线服务）
```bash
python run.py \
  --model MAS4TS \
  --use_vlm \
  --use_eas \
  --eas_url https://your-endpoint.com \
  ...
```

## 📊 可视化输出

所有中间结果会保存到：
```
./visualizations/
├── data_analysis/
│   ├── batch_0_analysis.png       # Data Analyzer输出
│   └── batch_0_statistics.txt
├── visual_anchors/
│   ├── batch_0_anchors.png        # Visual Anchor输出
│   └── batch_0_anchors.json
└── numerical_reasoning/
    └── batch_0_predictions.json   # Numerical Adapter输出
```

## 🔧 下一步工作

1. **实现Data Analyzer可视化** (30分钟)
   - 添加`_generate_plot()`方法
   - 添加`_generate_statistics_text()`方法

2. **实现Visual Anchor VLM集成** (1小时)
   - 添加本地Qwen-VL加载
   - 添加EAS客户端
   - 实现prompt调用

3. **实现Numerical Adapter并发LLM** (1小时)
   - 并发调用多个模型
   - Ensemble策略
   - Fallback机制

4. **测试完整流程** (30分钟)
   - 运行测试脚本
   - 检查可视化输出
   - 验证结果准确性

## 💡 设计理念

**为什么使用Multi-Agent而不是单一模型？**

1. **专业分工**: 每个Agent专注于一个子任务
2. **可解释性**: 每个步骤的中间结果可视化
3. **灵活性**: 可以单独替换/优化任何Agent
4. **并发性**: Numerical Adapter并发调用3个LLM，提速3倍
5. **准确性**: VLM+LLM ensemble提升预测精度

**核心创新点**:
- 🎯 **Visual Anchoring**: 将时序转为图像，用VLM理解模式
- 🧮 **Numerical Reasoning**: 用LLM ensemble进行精确数值推理
- 🤝 **Agent Collaboration**: 4个Agent协作，不是简单pipeline

## 📚 相关文档

- `MULTI_AGENT_IMPLEMENTATION.md` - 详细实现指南
- `QUICK_START.md` - 快速开始
- `BUG_FIX_SUMMARY.md` - Bug修复记录
- `README.md` - 项目总览

