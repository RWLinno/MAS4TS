# MAS4TS 配置指南

## 当前配置状态

### 统一EAS配置
目前所有agent都使用同一个EAS端点：
- **模型**: Qwen3-VL-235B-A22B-Instruct-BF16
- **地址**: http://1054059136692489.cn-shanghai.pai-eas.aliyuncs.com/api/predict/qwen3_vl_235b_a22b_instruct_bf16
- **Token**: NmQ0ZWIzMzA1MjdjMjQ2N2EyYjQ3YTEzYTViMGFhYjk4YjU4MGNjZg==

---

## 配置结构说明

### 1. 全局统一配置 (eas_unified)

```json
{
  "eas_unified": {
    "comment": "当前统一使用的EAS配置 - 可以为每个agent单独覆盖",
    "base_url": "...",
    "token": "...",
    "model_name": "...",
    "timeout": 30,
    "max_retries": 3
  }
}
```

这个配置作为参考，方便后续：
- 快速查看当前统一使用的EAS信息
- 批量更换EAS端点时的模板

### 2. Agent专属配置

每个agent都有自己的`eas_config`，可以独立配置：

#### Visual Anchor (VLM)
```json
{
  "agents": {
    "visual_anchor": {
      "use_vlm": false,  // 是否启用VLM
      "use_eas": false,  // 是否使用EAS
      "model_name": "Qwen3-VL-235B-A22B-Instruct-BF16",
      "max_tokens": 1024,
      "temperature": 0.3,
      "eas_config": {
        "base_url": "...",
        "token": "...",
        "timeout": 30,
        "max_retries": 3
      }
    }
  }
}
```

#### Numerologic Adapter (LLM Ensemble)
```json
{
  "agents": {
    "numerologic_adapter": {
      "use_llm": false,  // 是否启用LLM
      "use_eas": false,  // 是否使用EAS
      "num_llm_models": 3,
      "llm_ensemble": [
        {
          "model_name": "Qwen3-VL-235B-A22B-Instruct-BF16",
          "max_tokens": 512,
          "temperature": 0.5,
          "eas_config": {
            "base_url": "...",
            "token": "...",
            "timeout": 30
          }
        }
        // ... 可以配置多个不同的LLM模型
      ]
    }
  }
}
```

---

## 使用方式

### 方式1: 使用配置文件（当前）

直接修改 `src/config.json`：

1. **启用VLM/LLM**:
   ```json
   "visual_anchor": {
     "use_vlm": true,
     "use_eas": true
   }
   ```

2. **使用统一EAS**: 所有agent的`eas_config`已配置为同一地址

3. **单独配置某个agent**: 修改对应agent的`eas_config`

### 方式2: 使用环境变量

环境变量会覆盖配置文件：

```bash
# VLM专用
export VLM_EAS_BASE_URL="http://your-vlm-endpoint"
export VLM_EAS_TOKEN="your-vlm-token"

# LLM专用
export LLM_EAS_BASE_URL="http://your-llm-endpoint"
export LLM_EAS_TOKEN="your-llm-token"

# 模型专用（优先级最高）
export QWEN3_VL_235B_A22B_INSTRUCT_BF16_EAS_BASE_URL="..."
export QWEN3_VL_235B_A22B_INSTRUCT_BF16_EAS_TOKEN="..."
```

**优先级**: 模型专用环境变量 > 通用环境变量 > config.json

---

## 切换到多EAS配置

### 场景1: Visual Anchor用VLM，LLM用文本模型

1. 修改 `visual_anchor.eas_config`:
   ```json
   {
     "base_url": "http://vlm-endpoint",
     "token": "vlm-token"
   }
   ```

2. 修改 `numerologic_adapter.llm_ensemble`:
   ```json
   {
     "model_name": "Qwen/Qwen2.5-7B-Instruct",
     "eas_config": {
       "base_url": "http://llm-7b-endpoint",
       "token": "llm-7b-token"
     }
   }
   ```

### 场景2: LLM Ensemble使用多个不同模型

```json
{
  "llm_ensemble": [
    {
      "model_name": "Qwen2.5-7B",
      "temperature": 0.5,
      "eas_config": {
        "base_url": "http://7b-endpoint",
        "token": "token-7b"
      }
    },
    {
      "model_name": "Qwen2.5-14B",
      "temperature": 0.4,
      "eas_config": {
        "base_url": "http://14b-endpoint",
        "token": "token-14b"
      }
    },
    {
      "model_name": "Qwen2.5-72B",
      "temperature": 0.3,
      "eas_config": {
        "base_url": "http://72b-endpoint",
        "token": "token-72b"
      }
    }
  ]
}
```

---

## 测试建议

### 1. 不启用VLM/LLM（当前设置）
```bash
python run.py --model MAS4TS --data ETTh1 --task_name long_term_forecast
```

系统会使用基于规则的方法，不调用EAS。

### 2. 启用VLM进行视觉分析
```json
"visual_anchor": {
  "use_vlm": true,
  "use_eas": true
}
```

然后运行：
```bash
python run.py --model MAS4TS --data ETTh1 --task_name long_term_forecast
```

### 3. 启用LLM Ensemble
```json
"numerologic_adapter": {
  "use_llm": true,
  "use_eas": true
}
```

### 4. Debug模式
```bash
CUDA_LAUNCH_BLOCKING=1 python run.py --model MAS4TS --data ETTh1
```

---

## 配置检查清单

在启用EAS之前，确保：

- [ ] `eas_config.base_url` 正确填写
- [ ] `eas_config.token` 正确填写
- [ ] `use_eas` 设置为 `true`
- [ ] 对应的agent启用标志（`use_vlm`/`use_llm`）设置为 `true`
- [ ] 网络可以访问EAS端点
- [ ] Token有效且有足够权限

---

## 常见问题

### Q1: 如何暂时禁用VLM/LLM？
**A**: 设置 `use_vlm: false` 或 `use_llm: false`，系统会使用基于规则的方法。

### Q2: 为什么配置了EAS但没有调用？
**A**: 检查：
1. `use_eas` 是否为 `true`
2. `use_vlm` 或 `use_llm` 是否为 `true`
3. `multi_agent_system.use_eas` 是否为 `true`（如果需要全局启用）

### Q3: 如何批量更换EAS端点？
**A**: 
1. 更新 `eas_unified` 作为参考
2. 使用脚本批量替换所有agent的 `eas_config`
3. 或者使用环境变量统一覆盖

### Q4: 如何验证EAS配置是否正确？
**A**: 运行时查看日志：
```
[VisualAnchorAgent] VLM EAS client initialized: http://...
[NumerologicAdapterAgent] LLM EAS client initialized for Qwen3-VL-...: http://...
```

---

## 性能调优

### Token分配
- **Visual Anchor (VLM)**: 1024 tokens - 用于视觉推理
- **LLM Ensemble**: 512 tokens - 用于数值推理

### Temperature设置
- **VLM**: 0.3 - 更确定的视觉分析
- **LLM**: 0.3-0.5 - 根据模型大小调整
  - 大模型(72B): 0.3
  - 中等模型(14B): 0.4
  - 小模型(7B): 0.5

### Timeout配置
- **默认**: 30秒
- **建议**: 根据网络状况和模型大小调整
  - 本地/内网: 10-20秒
  - 公网: 30-60秒
  - 大模型: 60-120秒

---

## 版本历史

- **v1.1** (2025-11-05): 统一EAS配置，支持每个agent独立配置
- **v1.0** (2025-11-05): 初始版本

---

**配置完成！** 现在可以开始测试了。🎯

