# Multi-Agent System for Time Series Analysis(MAS4TS)


## Quick Start
```
# 1. clone repo
git clone https://github.com/RWLinno/MAS4TS.git
cd MAS4TS

# 2. virtual environment
conda create -n MAS4TS python==3.12
conda activate MAS4TS
pip install -r requirements.txt

# 3. download data and pre-trained model
gdown https://drive.google.com/uc?id=1pmXvqWsfUeXWCMz5fqsP8WLKXR5jxY8z
unzip all_datasets.zip # 放到./dataset下面

# 4. config setting
cd src
cp config.example.json config.json
```

## Structure
The Repo is built upon Time-series-Library. We thank the authors for their great work. The core structure of our multi-agent system is shown as below.
```
src/
├── model.py                     # 模型入口
├── config.example.json          # 参数设定模版
├── base/                        
│   ├── unified_manager.py       # 统一配置管理预训练模型，并发推理
│   ├── processor.py             # 负责处理单个batch
├── agents/                      # 存放各个agent
│   ├── manager_agent.py           # 负责调度任务和决策
│   ├── data_analyzer.py         # 负责数据分析和处理，缺失异常值处理/差分/分段平滑
│   ├── visual_anchor.py         # 可视化时序数据，做一个置信区间和预测值的锚定
│   ├── numerologic_adapter.py   # 结合数值推理，得到精细化结果
│   ├── knowledge_retriever.py   # 支持维护向量库和近似近邻检索
│   └── task_executor.py         # 负责各个下游任务，异常检测，分类，外部变量预测，差值，长/短期预测
├── utils/                       # 实用工具
│   ├── embedding.py             # 数据编码
│   ├── logger.py                # 日志打印
│   ├── config_loader.py         
│   └── ...
├── demo/...                     # 提供一个用户使用的网页可视化demo
├── tools/...                    # 存放agent需要的工具调用
└── scripts/...                  # 训练脚本文件