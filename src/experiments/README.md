# OnCallAgent 实验验证方案

## 实验设计概述

基于真实OnCall场景数据进行完整的实验验证，包括对比实验、消融实验和超参数分析。

## 1. Setup部分：统一实验设置

### 1.1 Benchmark构建
- **基础数据**: 基于`data/documents/engineering/autoops/`中的真实OnCall问题
- **问答对构造**: 使用GPT-4从checklist和问题文档中生成问答对
- **多模态数据**: 包含文本查询、图像分析（oncall_case.jpg）、文档检索

### 1.2 Baseline方法
1. **GPT-4V Direct**: 直接使用GPT-4V处理所有查询
2. **Single-VLM**: 单一Qwen2.5-VL-7B模型处理
3. **LLM-Only**: 仅使用文本LLM（不处理图像）
4. **RAG-Only**: 传统RAG系统（无多智能体协作）
5. **AutoGPT-DevOps**: AutoGPT适配DevOps场景
6. **OnCallAgent**: 完整多智能体系统

### 1.3 评估指标
- **准确率**: 回答正确性（专家评估）
- **响应时间**: 查询到回答的时间
- **完整性评分**: 回答覆盖度（1-5分）
- **多模态利用率**: 多种数据类型的有效使用
- **置信度分数**: 系统对答案的信心度

## 2. 实验数据集构造

### 2.1 OnCall场景问答数据集
基于真实文档构造500个测试样例：
- **纯文本查询** (200个): 基于common_problems.md等
- **图像查询** (150个): 包含截图分析的问题
- **文档检索查询** (100个): 需要@文档引用的问题
- **复合查询** (50个): 多模态复合问题

### 2.2 数据构造脚本
使用大模型基于现有文档自动生成问答对，确保：
- 问题的真实性和多样性
- 标准答案的准确性
- 多种难度级别的覆盖

## 3. 消融实验设计

### 3.1 Agent移除实验
- **No Visual Agent**: 移除视觉分析智能体
- **No Log Agent**: 移除日志分析智能体  
- **No Metrics Agent**: 移除指标分析智能体
- **No Knowledge Agent**: 移除知识检索智能体
- **No Route Agent**: 移除路由智能体，随机分配
- **No RAG**: 移除检索增强生成
- **No MCP**: 移除工具调用能力

### 3.2 Agent替换实验
- **Normal LLM**: 用普通LLM替换专业Agent
- **Single Route**: 用简单规则替换智能路由
- **Fixed Assignment**: 固定Agent分配策略

## 4. 超参数分析

### 4.1 核心超参数
- **询问轮数** (max_turns): [1, 3, 5, 10]
- **最大长度** (max_length): [256, 512, 1024, 2048]
- **置信度阈值** (confidence_threshold): [0.6, 0.7, 0.8, 0.9]
- **温度参数** (temperature): [0.1, 0.3, 0.7, 1.0]
- **检索Top-K** (retrieval_top_k): [3, 5, 10, 20]

### 4.2 Agent特定参数
- **路由策略**: rule_based vs learned_routing
- **模型选择**: 不同规模的VLM/LLM对比
- **并行度**: 串行 vs 并行Agent执行

## 5. 实验脚本设计

统一的bash脚本管理所有实验运行。
