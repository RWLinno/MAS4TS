#!/usr/bin/env python3
"""
OnCallAgent 超参数分析实验
测试不同超参数设置对系统性能的影响
"""

import json
import time
import argparse
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Union
import sys
import os
import copy

# 添加OnCallAgent到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.oncall_agent.main import OnCallAgent

class HyperparameterExperiment:
    """超参数实验执行器"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.base_config = json.load(f)
    
    def create_hyperparam_config(self, param_name: str, param_value: Union[int, float, str]) -> Dict[str, Any]:
        """根据超参数创建修改后的配置"""
        config = copy.deepcopy(self.base_config)
        
        if param_name == "max_turns":
            # 修改最大对话轮数
            config["max_turns"] = param_value
            
        elif param_name == "max_length":
            # 修改最大生成长度
            for agent_name in config["agents"]:
                if "max_tokens" in config["agents"][agent_name]:
                    config["agents"][agent_name]["max_tokens"] = param_value
            
        elif param_name == "confidence_threshold":
            # 修改置信度阈值
            for agent_name in config["agents"]:
                if "confidence_threshold" in config["agents"][agent_name]:
                    config["agents"][agent_name]["confidence_threshold"] = param_value
            
        elif param_name == "temperature":
            # 修改温度参数
            for agent_name in config["agents"]:
                if "temperature" in config["agents"][agent_name]:
                    config["agents"][agent_name]["temperature"] = param_value
            
        elif param_name == "retrieval_top_k":
            # 修改检索Top-K
            if "rag" in config:
                config["rag"]["top_k"] = param_value
            if "retrieval" in config:
                config["retrieval"]["top_k"] = param_value
            
        elif param_name == "routing_strategy":
            # 修改路由策略
            config["agents"]["route_agent"]["routing_strategy"] = param_value
            
        elif param_name == "model_type":
            # 修改模型类型
            for agent_name in config["agents"]:
                if "model_name" in config["agents"][agent_name]:
                    config["agents"][agent_name]["model_name"] = param_value
            
        elif param_name == "parallel_execution":
            # 修改并行执行设置
            config["parallel_execution"] = param_value
            
        else:
            raise ValueError(f"未知的超参数: {param_name}")
        
        return config
    
    async def evaluate_query(self, query_data: Dict[str, Any], hyperparam_config: Dict[str, Any]) -> Dict[str, Any]:
        """使用超参数配置评估单个查询"""
        start_time = time.time()
        
        try:
            # 使用修改后的配置初始化OnCallAgent
            oncall_agent = OnCallAgent(hyperparam_config)
            
            # 准备查询上下文
            query_context = {
                "query": query_data["question"],
                "image": None,  # 实际实现中需要处理图像
                "context": {"keywords": query_data.get("keywords", [])},
                "model": hyperparam_config.get("default_model", "Qwen/Qwen2.5-VL-7B-Instruct"),
                "type": "online"
            }
            
            # 处理查询
            result = await oncall_agent.process_query(query_context)
            
            response = result.get("answer", "无响应")
            confidence = result.get("confidence", 0.0)
            metadata = result.get("metadata", {})
            
        except Exception as e:
            response = f"处理失败: {str(e)}"
            confidence = 0.0
            metadata = {}
        
        return {
            "query_id": query_data.get("id", "unknown"),
            "response": response,
            "response_time": time.time() - start_time,
            "confidence": confidence,
            "metadata": metadata,
            "ground_truth": query_data.get("answer", ""),
            "difficulty": query_data.get("difficulty", "medium"),
            "type": query_data.get("type", "text"),
            "source_doc": query_data.get("source_doc", "")
        }
    
    async def run_hyperparam_experiment(self, dataset_path: str, param_name: str, 
                                      param_value: Union[int, float, str], 
                                      batch_size: int = 10) -> List[Dict[str, Any]]:
        """运行超参数实验"""
        print(f"开始超参数实验: {param_name}={param_value}")
        
        # 创建超参数配置
        hyperparam_config = self.create_hyperparam_config(param_name, param_value)
        
        # 加载数据集
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        # 为了快速实验，只使用部分数据
        if len(dataset) > 100:
            dataset = dataset[:100]
        
        print(f"数据集大小: {len(dataset)}")
        
        results = []
        
        # 批量处理
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            print(f"处理批次 {i // batch_size + 1}/{(len(dataset) + batch_size - 1) // batch_size}")
            
            # 并发处理批次
            batch_tasks = [
                self.evaluate_query(query_data, hyperparam_config)
                for query_data in batch
            ]
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # 处理结果
            for result in batch_results:
                if isinstance(result, Exception):
                    print(f"评估失败: {result}")
                    continue
                results.append(result)
        
        print(f"✓ 超参数实验 {param_name}={param_value} 完成，共 {len(results)} 个结果")
        return results

def calculate_hyperparam_metrics(results: List[Dict[str, Any]], param_name: str, param_value: Union[int, float, str]) -> Dict[str, Any]:
    """计算超参数实验指标"""
    if not results:
        return {}
    
    total_queries = len(results)
    avg_response_time = sum(r["response_time"] for r in results) / total_queries
    avg_confidence = sum(r["confidence"] for r in results) / total_queries
    success_rate = len([r for r in results if r["confidence"] > 0.5]) / total_queries
    
    # 响应时间分布
    response_times = [r["response_time"] for r in results]
    response_times.sort()
    
    # 置信度分布
    confidence_distribution = {
        "low (0-0.3)": len([r for r in results if 0 <= r["confidence"] < 0.3]),
        "medium (0.3-0.7)": len([r for r in results if 0.3 <= r["confidence"] < 0.7]),
        "high (0.7-1.0)": len([r for r in results if 0.7 <= r["confidence"] <= 1.0])
    }
    
    # 按难度分析
    difficulty_analysis = {}
    for difficulty in ["easy", "medium", "hard"]:
        difficulty_results = [r for r in results if r["difficulty"] == difficulty]
        if difficulty_results:
            difficulty_analysis[difficulty] = {
                "count": len(difficulty_results),
                "avg_confidence": sum(r["confidence"] for r in difficulty_results) / len(difficulty_results),
                "avg_response_time": sum(r["response_time"] for r in difficulty_results) / len(difficulty_results),
                "success_rate": len([r for r in difficulty_results if r["confidence"] > 0.5]) / len(difficulty_results)
            }
    
    # 特定超参数的分析
    param_specific_metrics = {}
    
    if param_name == "max_turns":
        # 分析对话轮数的影响
        param_specific_metrics["turns_impact"] = {
            "avg_confidence_improvement": avg_confidence,
            "complex_query_handling": len([r for r in results if r["type"] == "multimodal" and r["confidence"] > 0.7])
        }
    
    elif param_name == "temperature":
        # 分析温度对创造性和一致性的影响
        param_specific_metrics["temperature_impact"] = {
            "response_diversity": len(set(r["response"][:50] for r in results)) / len(results),
            "confidence_variance": sum((r["confidence"] - avg_confidence) ** 2 for r in results) / len(results)
        }
    
    elif param_name == "retrieval_top_k":
        # 分析检索数量的影响
        retrieval_queries = [r for r in results if r["type"] in ["retrieval", "multimodal"]]
        if retrieval_queries:
            param_specific_metrics["retrieval_impact"] = {
                "retrieval_success_rate": len([r for r in retrieval_queries if r["confidence"] > 0.7]) / len(retrieval_queries),
                "avg_retrieval_time": sum(r["response_time"] for r in retrieval_queries) / len(retrieval_queries)
            }
    
    return {
        "param_name": param_name,
        "param_value": param_value,
        "total_queries": total_queries,
        "avg_response_time": avg_response_time,
        "avg_confidence": avg_confidence,
        "success_rate": success_rate,
        "response_time_percentiles": {
            "p50": response_times[len(response_times) // 2],
            "p90": response_times[int(len(response_times) * 0.9)],
            "p95": response_times[int(len(response_times) * 0.95)]
        },
        "confidence_distribution": confidence_distribution,
        "difficulty_analysis": difficulty_analysis,
        "param_specific_metrics": param_specific_metrics
    }

async def main():
    parser = argparse.ArgumentParser(description="运行OnCallAgent超参数实验")
    parser.add_argument("--param_name", required=True, choices=[
        "max_turns", "max_length", "confidence_threshold", "temperature", 
        "retrieval_top_k", "routing_strategy", "model_type", "parallel_execution"
    ], help="超参数名称")
    parser.add_argument("--param_value", required=True, help="超参数值")
    parser.add_argument("--dataset", required=True, help="数据集文件路径")
    parser.add_argument("--output", required=True, help="结果输出文件路径")
    parser.add_argument("--config", required=True, help="基础配置文件路径")
    parser.add_argument("--batch_size", type=int, default=5, help="批处理大小")
    
    args = parser.parse_args()
    
    # 类型转换
    param_value = args.param_value
    if args.param_name in ["max_turns", "max_length", "retrieval_top_k"]:
        param_value = int(param_value)
    elif args.param_name in ["confidence_threshold", "temperature"]:
        param_value = float(param_value)
    elif args.param_name == "parallel_execution":
        param_value = param_value.lower() == "true"
    
    # 运行超参数实验
    experiment = HyperparameterExperiment(args.config)
    results = await experiment.run_hyperparam_experiment(
        args.dataset, args.param_name, param_value, args.batch_size
    )
    
    # 计算指标
    metrics = calculate_hyperparam_metrics(results, args.param_name, param_value)
    
    # 保存结果
    output_data = {
        "hyperparameter": args.param_name,
        "value": param_value,
        "metrics": metrics,
        "detailed_results": results,
        "experiment_time": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 超参数实验结果已保存到 {args.output}")
    print(f"  超参数: {args.param_name}={param_value}")
    print(f"  平均置信度: {metrics['avg_confidence']:.3f}")
    print(f"  成功率: {metrics['success_rate']:.1%}")
    print(f"  平均响应时间: {metrics['avg_response_time']:.2f}秒")

if __name__ == "__main__":
    asyncio.run(main())
