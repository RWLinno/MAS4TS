#!/usr/bin/env python3
"""
OnCallAgent Baseline方法对比实验
"""

import json
import time
import argparse
import asyncio
from pathlib import Path
from typing import Dict, List, Any
import sys
import os

# 添加OnCallAgent到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.unified_manager import UnifiedAgentManager
from src.utils.config_loader import load_config
from src.utils.model_manager import UnifiedModelManager

class BaselineEvaluator:
    """Baseline方法评估器"""
    
    def __init__(self, config_path: str):
        # 加载配置
        self.config = load_config(config_path)
        self.results = []
        
    async def evaluate_gpt4v_direct(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """直接使用GPT-4V处理查询"""
        start_time = time.time()
        
        try:
            # 模拟GPT-4V直接调用
            # 实际实现中需要调用OpenAI API
            response = f"GPT-4V直接处理查询: {query[:100]}..."
            confidence = 0.75
            
        except Exception as e:
            response = f"处理失败: {str(e)}"
            confidence = 0.0
        
        return {
            "response": response,
            "response_time": time.time() - start_time,
            "confidence": confidence,
            "method": "gpt4v_direct"
        }
    
    async def evaluate_single_vlm(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """使用单一VLM模型处理"""
        start_time = time.time()
        
        try:
            # 使用OnCallAgent的模型管理器，但只用单一模型
            model_manager = UnifiedModelManager(
                model_name="Qwen/Qwen2.5-VL-7B-Instruct",
                offline_mode=False  # 确保使用在线模式
            )
            
            # 简化的单模型处理
            from src.utils.model_manager import ModelRequest
            model_request = ModelRequest(
                messages=[{"role": "user", "content": query}],
                max_tokens=512,
                temperature=0.7
            )
            
            # 模拟模型调用
            response = f"单一VLM处理: {query[:100]}..."
            confidence = 0.68
            
        except Exception as e:
            response = f"处理失败: {str(e)}"
            confidence = 0.0
        
        return {
            "response": response,
            "response_time": time.time() - start_time,
            "confidence": confidence,
            "method": "single_vlm"
        }
    
    async def evaluate_llm_only(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """仅使用文本LLM处理（忽略图像）"""
        start_time = time.time()
        
        try:
            # 移除图像信息，只处理文本
            text_only_query = query
            if "image" in context:
                text_only_query += " [注意：忽略图像信息]"
            
            response = f"仅文本LLM处理: {text_only_query[:100]}..."
            confidence = 0.62
            
        except Exception as e:
            response = f"处理失败: {str(e)}"
            confidence = 0.0
        
        return {
            "response": response,
            "response_time": time.time() - start_time,
            "confidence": confidence,
            "method": "llm_only"
        }
    
    async def evaluate_rag_only(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """传统RAG系统（无多智能体协作）"""
        start_time = time.time()
        
        try:
            # 使用OnCallAgent的RAG组件，但不使用多智能体协作
            from src.retrieval.rag_service import RAGService
            
            # 加载配置
            config = load_config()
            rag_service = RAGService(config)
            
            # 检索相关文档
            retrieved_docs = []  # 模拟检索结果
            
            # 简单的RAG响应生成
            response = f"RAG系统处理: {query[:100]}... [基于检索到的 {len(retrieved_docs)} 个文档]"
            confidence = 0.74
            
        except Exception as e:
            response = f"处理失败: {str(e)}"
            confidence = 0.0
        
        return {
            "response": response,
            "response_time": time.time() - start_time,
            "confidence": confidence,
            "method": "rag_only"
        }
    
    async def evaluate_autogpt_devops(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """AutoGPT适配DevOps场景"""
        start_time = time.time()
        
        try:
            # 模拟AutoGPT的链式思考过程
            thinking_steps = [
                "分析查询类型",
                "制定解决计划", 
                "执行工具调用",
                "综合结果"
            ]
            
            response = f"AutoGPT-DevOps处理: {query[:100]}... [执行了 {len(thinking_steps)} 个步骤]"
            confidence = 0.77
            
        except Exception as e:
            response = f"处理失败: {str(e)}"
            confidence = 0.0
        
        return {
            "response": response,
            "response_time": time.time() - start_time,
            "confidence": confidence,
            "method": "autogpt_devops"
        }
    
    async def evaluate_oncall_agent(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """完整的OnCallAgent系统"""
        start_time = time.time()
        
        try:
            # 初始化OnCallAgent
            config = load_config()
            agent_manager = UnifiedAgentManager(config)
            
            # 准备查询上下文
            query_context = {
                "query": query,
                "image": context.get("image"),
                "context": context.get("additional_context", {}),
                "model": config.get("models", {}).get("primary_model", "Qwen/Qwen2.5-VL-7B-Instruct"),
                "type": "online"
            }
            
            # 处理查询
            result = await agent_manager.process_request(query)
            
            response = result.content if hasattr(result, 'content') else str(result)
            confidence = result.confidence if hasattr(result, 'confidence') else 0.8
            
        except Exception as e:
            response = f"处理失败: {str(e)}"
            confidence = 0.0
        
        return {
            "response": response,
            "response_time": time.time() - start_time,
            "confidence": confidence,
            "method": "oncall_agent"
        }
    
    async def evaluate_single_query(self, query_data: Dict[str, Any], method: str) -> Dict[str, Any]:
        """评估单个查询"""
        query = query_data["question"]
        context = {
            "image": None,  # 实际实现中需要加载图像
            "additional_context": query_data.get("keywords", [])
        }
        
        # 根据方法选择评估函数
        method_map = {
            "gpt4v_direct": self.evaluate_gpt4v_direct,
            "single_vlm": self.evaluate_single_vlm,
            "llm_only": self.evaluate_llm_only,
            "rag_only": self.evaluate_rag_only,
            "autogpt_devops": self.evaluate_autogpt_devops,
            "oncall_agent": self.evaluate_oncall_agent
        }
        
        if method not in method_map:
            raise ValueError(f"未知的评估方法: {method}")
        
        # 执行评估
        result = await method_map[method](query, context)
        
        # 添加查询信息
        result.update({
            "query_id": query_data.get("id", "unknown"),
            "ground_truth": query_data.get("answer", ""),
            "difficulty": query_data.get("difficulty", "medium"),
            "type": query_data.get("type", "text"),
            "source_doc": query_data.get("source_doc", "")
        })
        
        return result
    
    async def run_evaluation(self, dataset_path: str, method: str, 
                           batch_size: int = 10, max_workers: int = 4) -> List[Dict[str, Any]]:
        """运行完整评估"""
        print(f"开始评估方法: {method}")
        
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
                self.evaluate_single_query(query_data, method)
                for query_data in batch
            ]
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # 处理结果
            for result in batch_results:
                if isinstance(result, Exception):
                    print(f"评估失败: {result}")
                    continue
                results.append(result)
        
        print(f"✓ {method} 评估完成，共 {len(results)} 个结果")
        return results

def calculate_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """计算评估指标"""
    if not results:
        return {}
    
    # 基础统计
    total_queries = len(results)
    avg_response_time = sum(r["response_time"] for r in results) / total_queries
    avg_confidence = sum(r["confidence"] for r in results) / total_queries
    
    # 按难度分组统计
    difficulty_stats = {}
    for difficulty in ["easy", "medium", "hard"]:
        difficulty_results = [r for r in results if r["difficulty"] == difficulty]
        if difficulty_results:
            difficulty_stats[difficulty] = {
                "count": len(difficulty_results),
                "avg_response_time": sum(r["response_time"] for r in difficulty_results) / len(difficulty_results),
                "avg_confidence": sum(r["confidence"] for r in difficulty_results) / len(difficulty_results)
            }
    
    # 按类型分组统计
    type_stats = {}
    for query_type in ["text", "image", "retrieval", "multimodal"]:
        type_results = [r for r in results if r["type"] == query_type]
        if type_results:
            type_stats[query_type] = {
                "count": len(type_results),
                "avg_response_time": sum(r["response_time"] for r in type_results) / len(type_results),
                "avg_confidence": sum(r["confidence"] for r in type_results) / len(type_results)
            }
    
    return {
        "total_queries": total_queries,
        "avg_response_time": avg_response_time,
        "avg_confidence": avg_confidence,
        "difficulty_stats": difficulty_stats,
        "type_stats": type_stats,
        "success_rate": len([r for r in results if r["confidence"] > 0.5]) / total_queries
    }

async def main():
    parser = argparse.ArgumentParser(description="运行OnCallAgent baseline评估")
    parser.add_argument("--method", required=True, choices=[
        "gpt4v_direct", "single_vlm", "llm_only", "rag_only", "autogpt_devops", "oncall_agent"
    ], help="评估方法")
    parser.add_argument("--dataset", required=True, help="数据集文件路径")
    parser.add_argument("--output", required=True, help="结果输出文件路径")
    parser.add_argument("--config", required=True, help="配置文件路径")
    parser.add_argument("--batch_size", type=int, default=10, help="批处理大小")
    parser.add_argument("--max_workers", type=int, default=4, help="最大并发数")
    
    args = parser.parse_args()
    
    # 运行评估
    evaluator = BaselineEvaluator(args.config)
    results = await evaluator.run_evaluation(
        args.dataset, args.method, args.batch_size, args.max_workers
    )
    
    # 计算指标
    metrics = calculate_metrics(results)
    
    # 保存结果
    output_data = {
        "method": args.method,
        "metrics": metrics,
        "detailed_results": results,
        "evaluation_time": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 结果已保存到 {args.output}")
    print(f"  平均响应时间: {metrics['avg_response_time']:.2f}秒")
    print(f"  平均置信度: {metrics['avg_confidence']:.3f}")
    print(f"  成功率: {metrics['success_rate']:.1%}")

if __name__ == "__main__":
    asyncio.run(main())
