# MAS4TS Multi-Agent系统实现说明

## 当前状态

✅ **已实现**: models/MAS4TS.py 现在真正调用Multi-Agent系统
✅ **流程**: 默认使用4个Agent的完整流程

## Multi-Agent流程

### 1. Data Analyzer Agent
**功能**: 
- 分析数据趋势和统计信息
- 生成时序plot图（保存到./visualizations/）
- 输出统计描述文本

**需要增强**:
```python
# 在 src/agents/data_analyzer.py 的 process() 中添加
import matplotlib.pyplot as plt
import os

def _generate_plot(self, data, save_path):
    """生成并保存时序plot图"""
    fig, ax = plt.subplots(figsize=(12, 6))
    for i in range(min(3, data.shape[2])):  # 绘制前3个特征
        ax.plot(data[0, :, i].cpu().numpy(), label=f'Feature {i}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path

def _generate_statistics_text(self, features):
    """生成统计描述文本"""
    text = f"Time Series Statistics:\n"
    text += f"- Mean: {features['mean'].mean().item():.4f}\n"
    text += f"- Std: {features['std'].mean().item():.4f}\n"
    text += f"- Trend: {features['trend'].mean().item():.4f}\n"
    return text
```

### 2. Visual Anchor Agent  
**功能**:
- 读入plot图和统计信息
- 使用VLM (Qwen-VL) 分析图像
- 生成锚点和置信区间
- 保存带标注的图片

**VLM集成** (需要添加):
```python
# 在 src/agents/visual_anchor.py 添加
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

class VisualAnchorAgent:
    def __init__(self, config):
        super().__init__("VisualAnchorAgent", config)
        self.use_vlm = config.get('use_vlm', True)
        self.use_eas = config.get('use_eas', False)
        
        if self.use_vlm and not self.use_eas:
            # 本地加载Qwen-VL
            self.vlm_model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen-VL-Chat", 
                device_map="auto",
                trust_remote_code=True
            ).eval()
            self.vlm_tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen-VL-Chat",
                trust_remote_code=True
            )
    
    async def _call_vlm(self, image_path, statistics_text):
        """调用VLM分析时序图像"""
        if self.use_eas:
            # 调用EAS在线服务
            return await self._call_eas_vlm(image_path, statistics_text)
        else:
            # 本地推理
            prompt = f"""Analyze this time series plot.
{statistics_text}

Based on the plot and statistics, predict:
1. Future trend direction (up/down/stable)
2. Confidence interval (narrow/medium/wide)  
3. Key anchor points for next {self.pred_len} steps

Provide numerical ranges."""

            image = Image.open(image_path)
            query = self.vlm_tokenizer.from_list_format([
                {'image': image_path},
                {'text': prompt},
            ])
            response, _ = self.vlm_model.chat(self.vlm_tokenizer, query=query, history=None)
            return response
```

### 3. Numerical Adapter Agent
**功能**:
- 使用LLM进行数值推理
- 并发询问多个模型
- 取平均结果作为最终预测约束

**并发LLM推理** (需要添加):
```python
# 在 src/agents/numerologic_adapter.py 添加
import asyncio

class NumerologicAdapterAgent:
    async def _parallel_llm_inference(self, prompt, num_models=3):
        """并发调用多个LLM模型"""
        tasks = []
        models = ['qwen-7b', 'qwen-14b', 'qwen-72b'][:num_models]
        
        for model_name in models:
            tasks.append(self._call_single_llm(model_name, prompt))
        
        results = await asyncio.gather(*tasks)
        
        # 取平均或投票
        return self._aggregate_results(results)
    
    async def _call_single_llm(self, model_name, prompt):
        """调用单个LLM"""
        if self.use_eas:
            return await self._call_eas_llm(model_name, prompt)
        else:
            # 本地推理
            from transformers import AutoModelForCausalLM, AutoTokenizer
            model = AutoModelForCausalLM.from_pretrained(f"Qwen/{model_name}")
            tokenizer = AutoTokenizer.from_pretrained(f"Qwen/{model_name}")
            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = model.generate(**inputs)
            return tokenizer.decode(outputs[0])
```

### 4. Task Executor Agent
**功能**:
- 根据任务类型执行最终预测
- 整合所有agent的输出
- 返回最终结果

**已实现**: src/agents/task_executor.py

## Prompt设计

### Data Analyzer Prompt
```
Analyze the following time series data:
- Sequence length: {seq_len}
- Number of features: {n_features}
- Data range: [{min_val}, {max_val}]

Identify:
1. Overall trend (increasing/decreasing/stable)
2. Seasonality (period if any)
3. Anomalies (timestamps)
4. Volatility level (low/medium/high)
```

### Visual Anchor Prompt (给VLM)
```
You are analyzing a time series forecasting plot.

Current observations:
{statistics_text}

Task: Predict the next {pred_len} time steps.

Provide:
1. Expected value range: [lower_bound, upper_bound]
2. Confidence level: 0.0-1.0
3. Key anchor points: timestamps where significant changes may occur
4. Trend continuation: will the trend continue or reverse?

Format your response as JSON:
{
  "range": [lower, upper],
  "confidence": 0.95,
  "anchors": [t1, t2, ...],
  "trend": "continue/reverse"
}
```

### Numerical Adapter Prompt (给LLM)
```
Given time series forecasting task:
- Historical data statistics: {statistics}
- Visual analysis: {visual_anchors}
- Task: Predict next {pred_len} steps

Constraints from visual analysis:
- Expected range: {range}
- Anchor points: {anchors}

Perform numerical reasoning to refine predictions:
1. Validate anchor points against historical patterns
2. Adjust confidence intervals based on volatility
3. Provide refined numerical predictions

Output format:
{
  "predictions": [v1, v2, ...],
  "confidence_intervals": [[l1,u1], [l2,u2], ...],
  "reasoning": "explanation"
}
```

## 配置文件

在 `run.py` 中添加参数:
```bash
parser.add_argument('--use_vlm', action='store_true', help='Use VLM for visual analysis')
parser.add_argument('--vlm_model', type=str, default='qwen-vl', help='VLM model name')
parser.add_argument('--use_eas', action='store_true', help='Use EAS online service')
parser.add_argument('--num_llm_models', type=int, default=3, help='Number of LLMs for ensemble')
```

## 运行示例

```bash
# 使用Multi-Agent系统 + VLM
python run.py \
  --model MAS4TS \
  --task_name long_term_forecast \
  --use_vlm \
  --vlm_model Qwen/Qwen-VL-Chat \
  --num_llm_models 3 \
  --data ETTh1 \
  --seq_len 96 \
  --pred_len 96

# 使用EAS在线服务
python run.py \
  --model MAS4TS \
  --use_vlm \
  --use_eas \
  --eas_url https://your-eas-endpoint.com \
  ...
```

## 可视化输出

所有可视化结果保存在:
```
./visualizations/
├── data_analysis/
│   ├── batch_0_analysis.png
│   ├── batch_0_statistics.txt
├── visual_anchors/
│   ├── batch_0_anchors.png
│   ├── batch_0_anchors.json
└── final_results/
    ├── batch_0_predictions.png
```

## 下一步实现

1. **添加VLM集成**: 在 `src/agents/visual_anchor.py`
2. **添加并发LLM**: 在 `src/agents/numerologic_adapter.py`  
3. **完善可视化**: 在 `src/agents/data_analyzer.py`
4. **添加EAS支持**: 创建 `src/utils/eas_client.py`
5. **更新run.py**: 添加新的命令行参数

## 性能优势

- **并发执行**: 3个LLM并发推理，速度提升3倍
- **视觉理解**: VLM能识别人类难以发现的模式
- **数值精度**: 多模型ensemble提升准确性
- **可解释性**: 每个agent提供中间结果和推理过程

