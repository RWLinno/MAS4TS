#!/usr/bin/env python3
"""
OnCallAgent 开源模型对比实验
测试不同开源LLM/VLM作为Agent backend的效果
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
from model_configs import list_supported_models, get_model_config

class ModelComparisonExperiment:
    """开源模型对比实验"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.base_config = json.load(f)
    
    def create_model_config(self, model_name: str) -> Dict[str, Any]:
        """为指定模型创建配置"""
        config = copy.deepcopy(self.base_config)
        
        # 更新所有Agent使用指定模型
        for agent_name in config["agents"]:
            if "model_name" in config["agents"][agent_name]:
                config["agents"][agent_name]["model_name"] = model_name
        
        # 设置默认模型
        config["default_model"] = model_name
        
        # 根据模型调整生成参数
        model_config = get_model_config(model_name)
        for agent_name in config["agents"]:
            if "max_tokens" in config["agents"][agent_name]:
                config["agents"][agent_name]["max_tokens"] = model_config.generation_params.get("max_new_tokens", 512)
            if "temperature" in config["agents"][agent_name]:
                config["agents"][agent_name]["temperature"] = model_config.generation_params.get("temperature", 0.7)
        
        return config
    
    async def evaluate_query_with_model(self, query_data: Dict[str, Any], model_name: str) -> Dict[str, Any]:
        """使用指定模型评估查询"""
        start_time = time.time()
        
        try:
            # 创建模型特定配置
            model_config = self.create_model_config(model_name)
            
            # 初始化OnCallAgent
            oncall_agent = OnCallAgent(model_config)
            
            # 准备查询上下文
            query_context = {
                "query": query_data["question"],
                "image": None,  # 实际实现中需要处理图像
                "context": {"keywords": query_data.get("keywords", [])},
                "model": model_name,
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
            metadata = {"error": str(e)}
        
        return {
            "query_id": query_data.get("id", "unknown"),
            "model_name": model_name,
            "response": response,
            "response_time": time.time() - start_time,
            "confidence": confidence,
            "metadata": metadata,
            "ground_truth": query_data.get("answer", ""),
            "difficulty": query_data.get("difficulty", "medium"),
            "type": query_data.get("type", "text"),
            "source_doc": query_data.get("source_doc", "")
        }
    
    async def run_model_comparison(self, dataset_path: str, models: List[str], 
                                 batch_size: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """运行多模型对比实验"""
        print(f"开始模型对比实验，测试 {len(models)} 个模型")
        
        # 加载数据集
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        # 为了快速实验，只使用部分数据
        if len(dataset) > 100:
            dataset = dataset[:100]
        
        print(f"数据集大小: {len(dataset)}")
        
        all_results = {}
        
        for model_name in models:
            print(f"\n正在测试模型: {model_name}")
            model_results = []
            
            # 批量处理
            for i in range(0, len(dataset), batch_size):
                batch = dataset[i:i + batch_size]
                print(f"  批次 {i // batch_size + 1}/{(len(dataset) + batch_size - 1) // batch_size}")
                
                # 串行处理（避免GPU内存不足）
                for query_data in batch:
                    try:
                        result = await self.evaluate_query_with_model(query_data, model_name)
                        model_results.append(result)
                    except Exception as e:
                        print(f"  查询处理失败: {e}")
                        continue
            
            all_results[model_name] = model_results
            print(f"✓ 模型 {model_name} 完成，共 {len(model_results)} 个结果")
        
        return all_results

def calculate_model_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """计算模型性能指标"""
    if not results:
        return {}
    
    total_queries = len(results)
    avg_response_time = sum(r["response_time"] for r in results) / total_queries
    avg_confidence = sum(r["confidence"] for r in results) / total_queries
    success_rate = len([r for r in results if r["confidence"] > 0.5]) / total_queries
    
    # GPU内存使用情况（如果有的话）
    gpu_memory_info = {}
    if results[0]["metadata"].get("gpu_memory"):
        gpu_memory_info = {
            "peak_memory_mb": max(r["metadata"].get("gpu_memory", {}).get("peak", 0) for r in results),
            "avg_memory_mb": sum(r["metadata"].get("gpu_memory", {}).get("avg", 0) for r in results) / total_queries
        }
    
    # 按难度分析
    difficulty_analysis = {}
    for difficulty in ["easy", "medium", "hard"]:
        difficulty_results = [r for r in results if r["difficulty"] == difficulty]
        if difficulty_results:
            difficulty_analysis[difficulty] = {
                "count": len(difficulty_results),
                "avg_confidence": sum(r["confidence"] for r in difficulty_results) / len(difficulty_results),
                "success_rate": len([r for r in difficulty_results if r["confidence"] > 0.5]) / len(difficulty_results)
            }
    
    return {
        "total_queries": total_queries,
        "avg_response_time": avg_response_time,
        "avg_confidence": avg_confidence,
        "success_rate": success_rate,
        "difficulty_analysis": difficulty_analysis,
        "gpu_memory_info": gpu_memory_info
    }

async def main():
    parser = argparse.ArgumentParser(description="运行OnCallAgent模型对比实验")
    parser.add_argument("--dataset", required=True, help="数据集文件路径")
    parser.add_argument("--output", required=True, help="结果输出文件路径")
    parser.add_argument("--config", required=True, help="基础配置文件路径")
    parser.add_argument("--models", nargs='+', help="要测试的模型列表", 
                       default=["Qwen/Qwen2.5-7B-Instruct", "THUDM/chatglm3-6b"])
    parser.add_argument("--batch_size", type=int, default=5, help="批处理大小")
    
    args = parser.parse_args()
    
    # 验证模型是否支持
    supported_models = list_supported_models()
    valid_models = [m for m in args.models if m in supported_models]
    
    if not valid_models:
        print(f"错误: 没有找到支持的模型。支持的模型列表:")
        for model in supported_models:
            print(f"  - {model}")
        return
    
    print(f"将测试以下模型: {valid_models}")
    
    # 运行模型对比实验
    experiment = ModelComparisonExperiment(args.config)
    all_results = await experiment.run_model_comparison(
        args.dataset, valid_models, args.batch_size
    )
    
    # 计算每个模型的指标
    model_metrics = {}
    for model_name, results in all_results.items():
        model_metrics[model_name] = calculate_model_metrics(results)
    
    # 保存结果
    output_data = {
        "experiment_type": "model_comparison",
        "tested_models": valid_models,
        "model_metrics": model_metrics,
        "detailed_results": all_results,
        "experiment_time": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 模型对比实验结果已保存到 {args.output}")
    
    # 输出性能排名
    print("\n📊 模型性能排名:")
    model_ranking = sorted(model_metrics.items(), 
                          key=lambda x: x[1].get("avg_confidence", 0), reverse=True)
    
    for i, (model_name, metrics) in enumerate(model_ranking, 1):
        print(f"  {i}. {model_name}")
        print(f"     准确率: {metrics['avg_confidence']:.3f}")
        print(f"     响应时间: {metrics['avg_response_time']:.2f}s")
        print(f"     成功率: {metrics['success_rate']:.1%}")

if __name__ == "__main__":
    asyncio.run(main())
