#!/usr/bin/env python3
"""
OnCallAgent Agent替换实验
测试用普通LLM替换专业Agent的效果
"""

import json
import time
import argparse
import asyncio
from pathlib import Path
from typing import Dict, List, Any
import sys
import os
import copy

# 添加OnCallAgent到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.oncall_agent.main import OnCallAgent

class ReplacementExperiment:
    """Agent替换实验执行器"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.base_config = json.load(f)
    
    def create_replacement_config(self, replacement_type: str) -> Dict[str, Any]:
        """根据替换类型创建修改后的配置"""
        config = copy.deepcopy(self.base_config)
        
        if replacement_type == "normal_llm":
            # 用普通LLM替换所有专业Agent
            for agent_name in config["agents"]:
                if agent_name != "route_agent":
                    config["agents"][agent_name]["model_name"] = "gpt-3.5-turbo"
                    config["agents"][agent_name]["specialized_prompt"] = False
            
        elif replacement_type == "simple_route":
            # 用简单规则替换智能路由
            config["agents"]["route_agent"]["routing_strategy"] = "simple_rules"
            config["agents"]["route_agent"]["use_ml_routing"] = False
            
        elif replacement_type == "fixed_assignment":
            # 固定Agent分配策略
            config["agents"]["route_agent"]["routing_strategy"] = "fixed"
            config["fixed_assignment"] = {
                "text": "knowledge_agent",
                "image": "visual_analysis_agent", 
                "retrieval": "retriever_agent",
                "default": "comprehensive_agent"
            }
            
        else:
            raise ValueError(f"未知的替换类型: {replacement_type}")
        
        return config
    
    async def evaluate_query(self, query_data: Dict[str, Any], replacement_config: Dict[str, Any]) -> Dict[str, Any]:
        """使用替换配置评估单个查询"""
        start_time = time.time()
        
        try:
            # 使用修改后的配置初始化OnCallAgent
            oncall_agent = OnCallAgent(replacement_config)
            
            # 准备查询上下文
            query_context = {
                "query": query_data["question"],
                "image": None,
                "context": {"keywords": query_data.get("keywords", [])},
                "model": replacement_config.get("default_model", "Qwen/Qwen2.5-VL-7B-Instruct"),
                "type": "online"
            }
            
            # 处理查询
            result = await oncall_agent.process_query(query_context)
            
            response = result.get("answer", "无响应")
            confidence = result.get("confidence", 0.0)
            
        except Exception as e:
            response = f"处理失败: {str(e)}"
            confidence = 0.0
        
        return {
            "query_id": query_data.get("id", "unknown"),
            "response": response,
            "response_time": time.time() - start_time,
            "confidence": confidence,
            "ground_truth": query_data.get("answer", ""),
            "difficulty": query_data.get("difficulty", "medium"),
            "type": query_data.get("type", "text"),
            "source_doc": query_data.get("source_doc", "")
        }
    
    async def run_replacement_experiment(self, dataset_path: str, replacement_type: str, 
                                       batch_size: int = 10) -> List[Dict[str, Any]]:
        """运行替换实验"""
        print(f"开始替换实验: {replacement_type}")
        
        # 创建替换配置
        replacement_config = self.create_replacement_config(replacement_type)
        
        # 加载数据集
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        print(f"数据集大小: {len(dataset)}")
        
        results = []
        
        # 批量处理
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            print(f"处理批次 {i // batch_size + 1}/{(len(dataset) + batch_size - 1) // batch_size}")
            
            # 并发处理批次
            batch_tasks = [
                self.evaluate_query(query_data, replacement_config)
                for query_data in batch
            ]
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # 处理结果
            for result in batch_results:
                if isinstance(result, Exception):
                    print(f"评估失败: {result}")
                    continue
                results.append(result)
        
        print(f"✓ 替换实验 {replacement_type} 完成，共 {len(results)} 个结果")
        return results

async def main():
    parser = argparse.ArgumentParser(description="运行OnCallAgent替换实验")
    parser.add_argument("--replacement_type", required=True, choices=[
        "normal_llm", "simple_route", "fixed_assignment"
    ], help="替换实验类型")
    parser.add_argument("--dataset", required=True, help="数据集文件路径")
    parser.add_argument("--output", required=True, help="结果输出文件路径")
    parser.add_argument("--config", required=True, help="基础配置文件路径")
    parser.add_argument("--batch_size", type=int, default=10, help="批处理大小")
    
    args = parser.parse_args()
    
    # 运行替换实验
    experiment = ReplacementExperiment(args.config)
    results = await experiment.run_replacement_experiment(
        args.dataset, args.replacement_type, args.batch_size
    )
    
    # 计算指标（重用消融实验的指标计算函数）
    from run_ablation import calculate_ablation_metrics
    metrics = calculate_ablation_metrics(results)
    
    # 保存结果
    output_data = {
        "replacement_type": args.replacement_type,
        "metrics": metrics,
        "detailed_results": results,
        "experiment_time": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 替换实验结果已保存到 {args.output}")

if __name__ == "__main__":
    asyncio.run(main())
