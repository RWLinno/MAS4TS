# MAS4TS 配置文件结构说明

## 概述

配置文件采用分层结构，将全局配置和agent专属配置分离，每个配置项放在最合适的位置。

---

## 配置文件结构

```json
{
  "global_config": {
    // 全局配置
    "use_parallel_agents": true,
    "save_visualizations": true,
    "parallel_execution": {
      // 全局并行执行配置
      "enable_batch_splitting": true,
      "enable_batch_parallel": true,
      "max_parallel_batches": 4,
      "num_splits": 2,
      "enable_concurrent_llm": true,
      "max_concurrent_requests": 5
    }
  },
  
  "directories": {
    // 全局目录配置
  },
  
  "logging": {
    // 全局日志配置
  },
  
  "agents_config": {
    "data_analyzer": {
      // DataAnalyzerAgent专属配置
      "data_processing": {
        // 数据处理配置（属于DataAnalyzer）
      }
    },
    "visual_anchor": {
      // VisualAnchorAgent专属配置
      "visualization": {
        // 可视化配置（属于VisualAnchor）
      }
    },
    "numerologic_adapter": {
      // NumerologicAdapterAgent专属配置
    },
    "task_executor": {
      // TaskExecutorAgent专属配置
    }
  }
}
```

---

## 详细说明

### 1. global_config (全局配置)

全局配置影响整个系统的行为。

```json
{
  "global_config": {
    "use_parallel_agents": true,        // 是否启用多agent并行
    "save_visualizations": true,        // 是否保存可视化结果
    "parallel_execution": {
      "enable_batch_splitting": true,   // 是否启用batch分割
      "enable_batch_parallel": true,    // 是否启用batch级并行
      "max_parallel_batches": 4,        // 最大并行batch数
      "num_splits": 2,                  // batch分割数量
      "enable_concurrent_llm": true,    // 是否启用LLM并发
      "max_concurrent_requests": 5      // 最大并发请求数
    }
  }
}
```

**使用位置**: `ManagerAgent` 读取并应用到整个多agent系统的调度

---

### 2. directories (目录配置)

全局目录路径配置。

```json
{
  "directories": {
    "vis_save_dir": "./visualizations/",
    "data_analysis_dir": "./visualizations/data_analysis/",
    "visual_anchors_dir": "./visualizations/visual_anchors/",
    "checkpoints_dir": "./checkpoints/",
    "results_dir": "./results/",
    "log_dir": "./logs/"
  }
}
```

**使用位置**: 所有agents和系统组件

---

### 3. logging (日志配置)

全局日志配置。

```json
{
  "logging": {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "to_file": true
  }
}
```

**使用位置**: 整个系统的日志系统

---

### 4. agents_config.data_analyzer (数据分析Agent)

DataAnalyzerAgent专属配置，包含数据处理相关的所有参数。

```json
{
  "agents_config": {
    "data_analyzer": {
      "enabled": true,
      "use_differencing": false,
      "smooth_window": 5,
      "anomaly_threshold": 3.0,
      "seasonal_period": 12,
      "top_k_features": 10,
      "feature_selection_method": "covariance",
      "data_processing": {
        "use_norm": true,              // 数据归一化
        "clip_predictions": false,     // 是否裁剪预测值
        "clip_min": -1e6,
        "clip_max": 1e6,
        "handle_missing": true          // 处理缺失值
      }
    }
  }
}
```

**为什么data_processing在这里？**
- 数据处理是DataAnalyzerAgent的核心职责
- 归一化、裁剪、缺失值处理都由DataAnalyzer执行
- BatchProcessor也会读取这些配置（通过data_analyzer.data_processing）

**使用位置**: 
- `DataAnalyzerAgent.__init__()` - 读取所有配置
- `BatchProcessor._load_data_processing_config()` - 读取data_processing部分

---

### 5. agents_config.visual_anchor (视觉锚定Agent)

VisualAnchorAgent专属配置，包含可视化相关的所有参数。

```json
{
  "agents_config": {
    "visual_anchor": {
      "enabled": true,
      "use_vlm": true,
      "use_eas": false,
      "model_name": "Qwen3-VL-235B-A22B-Instruct-BF16",
      "anchor_strategy": "confidence_interval",
      "confidence_level": 0.95,
      "max_tokens": 1024,
      "temperature": 0.3,
      "eas_config": {
        "base_url": "...",
        "token": "...",
        "timeout": 30,
        "max_retries": 3
      },
      "visualization": {
        "fig_size": [12, 6],           // 图片尺寸
        "dpi": 150,                     // 分辨率
        "plot_style": "default",        // 绘图风格
        "remove_plot_text": true,       // 是否移除文本（VLM视觉分析）
        "show_grid": true,              // 显示网格
        "show_legend": false,           // 显示图例
        "line_width": 2,                // 线宽
        "fill_alpha": 0.3,              // 填充透明度
        "color_scheme": "default"       // 配色方案
      }
    }
  }
}
```

**为什么visualization在这里？**
- VisualAnchorAgent负责生成和分析时序可视化图
- 所有绘图参数都与视觉分析直接相关
- DataAnalyzer虽然也生成plot，但它调用的是VisualAnchor的可视化功能

**使用位置**: 
- `VisualAnchorAgent.__init__()` - 读取所有配置
- `DataAnalyzerAgent._generate_plot()` - 也可以读取这些可视化参数

---

### 6. agents_config.numerologic_adapter (数值逻辑适配器Agent)

NumerologicAdapterAgent专属配置。

```json
{
  "agents_config": {
    "numerologic_adapter": {
      "enabled": true,
      "use_llm": false,
      "use_eas": false,
      "hidden_dim": 128,
      "num_layers": 2,
      "fusion_strategy": "attention",
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
        // ... 更多LLM模型
      ]
    }
  }
}
```

**使用位置**: `NumerologicAdapterAgent.__init__()`

---

### 7. agents_config.task_executor (任务执行Agent)

TaskExecutorAgent专属配置。

```json
{
  "agents_config": {
    "task_executor": {
      "enabled": true,
      "default_model": "DLinear",
      "model_config": {
        "d_model": 512,
        "n_heads": 8,
        "e_layers": 2,
        "d_layers": 1,
        "d_ff": 2048,
        "dropout": 0.1
      }
    }
  }
}
```

**使用位置**: `TaskExecutorAgent.__init__()`

---

## 配置读取机制

### 优先级

1. **传入的config参数** (最高优先级)
2. **环境变量**
3. **config.json文件**
4. **代码默认值** (最低优先级)

### 读取流程

每个Agent的`__init__()`方法中：

```python
def __init__(self, config: Dict[str, Any]):
    super().__init__("AgentName", config)
    
    # 1. 从config.json读取
    self._load_config_from_file()
    
    # 2. 允许传入的config覆盖
    self.param = config.get('param', self.param)
    
    # 3. 初始化其他组件
    ...
```

---

## 配置修改指南

### 修改全局并行配置

修改 `global_config.parallel_execution`:

```json
{
  "global_config": {
    "parallel_execution": {
      "max_parallel_batches": 8  // 增加并行数
    }
  }
}
```

### 修改数据处理配置

修改 `agents_config.data_analyzer.data_processing`:

```json
{
  "agents_config": {
    "data_analyzer": {
      "data_processing": {
        "use_norm": false,  // 关闭归一化
        "clip_predictions": true  // 启用裁剪
      }
    }
  }
}
```

### 修改可视化配置

修改 `agents_config.visual_anchor.visualization`:

```json
{
  "agents_config": {
    "visual_anchor": {
      "visualization": {
        "dpi": 300,  // 提高分辨率
        "remove_plot_text": false  // 显示文本标签
      }
    }
  }
}
```

### 启用VLM/LLM

```json
{
  "agents_config": {
    "visual_anchor": {
      "use_vlm": true,
      "use_eas": true
    },
    "numerologic_adapter": {
      "use_llm": true,
      "use_eas": true
    }
  }
}
```

---

## 代码实现

### DataAnalyzerAgent

```python
def _load_config_from_file(self):
    config_path = Path(__file__).parent.parent / 'config.json'
    with open(config_path) as f:
        json_config = json.load(f)
        agent_config = json_config['agents_config']['data_analyzer']
        
        # 读取基础配置
        self.top_k_features = agent_config['top_k_features']
        
        # 读取data_processing配置
        data_proc = agent_config['data_processing']
        self.use_norm = data_proc['use_norm']
        self.clip_predictions = data_proc['clip_predictions']
```

### VisualAnchorAgent

```python
def _load_config_from_file(self):
    config_path = Path(__file__).parent.parent / 'config.json'
    with open(config_path) as f:
        json_config = json.load(f)
        agent_config = json_config['agents_config']['visual_anchor']
        
        # 读取基础配置
        self.use_vlm = agent_config['use_vlm']
        
        # 读取visualization配置
        vis_config = agent_config['visualization']
        self.fig_size = tuple(vis_config['fig_size'])
        self.dpi = vis_config['dpi']
        self.remove_plot_text = vis_config['remove_plot_text']
```

### ManagerAgent

```python
def _load_parallel_config(self):
    config_path = Path(__file__).parent.parent / 'config.json'
    with open(config_path) as f:
        json_config = json.load(f)
        global_config = json_config['global_config']
        
        # 读取全局parallel配置
        parallel_config = global_config['parallel_execution']
        self.enable_batch_parallel = parallel_config['enable_batch_parallel']
        self.max_parallel_batches = parallel_config['max_parallel_batches']
```

---

## 优势

1. **清晰的职责分离**: 每个配置项都在最合适的位置
2. **易于维护**: 修改某个agent的配置，只需找到对应的agents_config
3. **灵活性**: 支持环境变量和传入参数覆盖
4. **可扩展**: 添加新agent时，只需在agents_config中添加新section

---

## 版本历史

- **v2.0** (2025-11-05): 优化配置结构，分离全局和agent专属配置
- **v1.1** (2025-11-05): 统一EAS配置
- **v1.0** (2025-11-05): 初始版本

