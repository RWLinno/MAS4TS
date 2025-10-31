#!/usr/bin/env python3
"""
OnCallAgent 消融实验脚本
测试移除不同Agent组件对系统性能的影响
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

class AblationExperiment:
    """消融实验执行器"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.base_config = json.load(f)
    
    def create_ablation_config(self, ablation_type: str) -> Dict[str, Any]:
        """根据消融类型创建修改后的配置"""
        config = copy.deepcopy(self.base_config)
        
        if ablation_type == "no_visual":
            # 禁用视觉分析智能体
            config["agents"]["visual_analysis_agent"]["enabled"] = False
            
        elif ablation_type == "no_log":
            # 禁用日志分析智能体
            config["agents"]["log_analysis_agent"]["enabled"] = False
            
        elif ablation_type == "no_metrics":
            # 禁用指标分析智能体
            config["agents"]["metrics_analysis_agent"]["enabled"] = False
            
        elif ablation_type == "no_knowledge":
            # 禁用知识检索智能体
            config["agents"]["knowledge_agent"]["enabled"] = False
            
        elif ablation_type == "no_route":
            # 禁用智能路由，使用随机分配
            config["agents"]["route_agent"]["routing_strategy"] = "random"
            config["agents"]["route_agent"]["confidence_threshold"] = 0.0
            
        elif ablation_type == "no_rag":
            # 禁用RAG功能
            for agent_name in config["agents"]:
                if "use_rag" in config["agents"][agent_name]:
                    config["agents"][agent_name]["use_rag"] = False
            config["rag"]["enabled"] = False
            
        elif ablation_type == "no_mcp":
            # 禁用MCP工具调用
            config["mcp"]["enabled"] = False
            config["tools"]["enabled"] = False
            
        else:
            raise ValueError(f"未知的消融类型: {ablation_type}")
        
        return config
    
    async def evaluate_query(self, query_data: Dict[str, Any], ablation_config: Dict[str, Any]) -> Dict[str, Any]:
        """使用消融配置评估单个查询"""
        start_time = time.time()
        
        try:
            # 使用修改后的配置初始化OnCallAgent
            oncall_agent = OnCallAgent(ablation_config)
            
            # 准备查询上下文
            query_context = {
                "query": query_data["question"],
                "image": None,  # 实际实现中需要处理图像
                "context": {"keywords": query_data.get("keywords", [])},
                "model": ablation_config.get("default_model", "Qwen/Qwen2.5-VL-7B-Instruct"),
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
    
    async def run_ablation_experiment(self, dataset_path: str, ablation_type: str, 
                                    batch_size: int = 10) -> List[Dict[str, Any]]:
        """运行消融实验"""
        print(f"开始消融实验: {ablation_type}")
        
        # 创建消融配置
        ablation_config = self.create_ablation_config(ablation_type)
        
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
                self.evaluate_query(query_data, ablation_config)
                for query_data in batch
            ]
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # 处理结果
            for result in batch_results:
                if isinstance(result, Exception):
                    print(f"评估失败: {result}")
                    continue
                results.append(result)
        
        print(f"✓ 消融实验 {ablation_type} 完成，共 {len(results)} 个结果")
        return results

def calculate_ablation_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """计算消融实验指标"""
    if not results:
        return {}
    
    total_queries = len(results)
    avg_response_time = sum(r["response_time"] for r in results) / total_queries
    avg_confidence = sum(r["confidence"] for r in results) / total_queries
    success_rate = len([r for r in results if r["confidence"] > 0.5]) / total_queries
    
    # 按难度分组统计
    difficulty_breakdown = {}
    for difficulty in ["easy", "medium", "hard"]:
        difficulty_results = [r for r in results if r["difficulty"] == difficulty]
        if difficulty_results:
            difficulty_breakdown[difficulty] = {
                "count": len(difficulty_results),
                "avg_confidence": sum(r["confidence"] for r in difficulty_results) / len(difficulty_results),
                "success_rate": len([r for r in difficulty_results if r["confidence"] > 0.5]) / len(difficulty_results)
            }
    
    # 按类型分组统计
    type_breakdown = {}
    for query_type in ["text", "image", "retrieval", "multimodal"]:
        type_results = [r for r in results if r["type"] == query_type]
        if type_results:
            type_breakdown[query_type] = {
                "count": len(type_results),
                "avg_confidence": sum(r["confidence"] for r in type_results) / len(type_results),
                "success_rate": len([r for r in type_results if r["confidence"] > 0.5]) / len(type_results)
            }
    
    return {
        "total_queries": total_queries,
        "avg_response_time": avg_response_time,
        "avg_confidence": avg_confidence,
        "success_rate": success_rate,
        "difficulty_breakdown": difficulty_breakdown,
        "type_breakdown": type_breakdown
    }

async def main():
    parser = argparse.ArgumentParser(description="运行OnCallAgent消融实验")
    parser.add_argument("--ablation_type", required=True, choices=[
        "no_visual", "no_log", "no_metrics", "no_knowledge", 
        "no_route", "no_rag", "no_mcp"
    ], help="消融实验类型")
    parser.add_argument("--dataset", required=True, help="数据集文件路径")
    parser.add_argument("--output", required=True, help="结果输出文件路径")
    parser.add_argument("--config", required=True, help="基础配置文件路径")
    parser.add_argument("--batch_size", type=int, default=10, help="批处理大小")
    
    args = parser.parse_args()
    
    # 运行消融实验
    experiment = AblationExperiment(args.config)
    results = await experiment.run_ablation_experiment(
        args.dataset, args.ablation_type, args.batch_size
    )
    
    # 计算指标
    metrics = calculate_ablation_metrics(results)
    
    # 保存结果
    output_data = {
        "ablation_type": args.ablation_type,
        "metrics": metrics,
        "detailed_results": results,
        "experiment_time": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 消融实验结果已保存到 {args.output}")
    print(f"  消融类型: {args.ablation_type}")
    print(f"  平均置信度: {metrics['avg_confidence']:.3f}")
    print(f"  成功率: {metrics['success_rate']:.1%}")
    print(f"  平均响应时间: {metrics['avg_response_time']:.2f}秒")

if __name__ == "__main__":
    asyncio.run(main())
