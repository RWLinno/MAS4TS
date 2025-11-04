"""
Unified Manager for Multi-Agent System
负责统一配置管理预训练模型，并发推理和多智能体协调
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from dataclasses import dataclass
import logging
import time

from ..utils.logger import setup_logger
from ..utils.config_loader import load_config
from .processor import BatchProcessor


@dataclass
class AgentTask:
    """Agent任务定义"""
    agent_name: str
    task_type: str  # 'visual_anchor', 'data_analysis', 'numerical_reasoning', etc.
    input_data: Any
    priority: int = 0
    metadata: Dict[str, Any] = None


@dataclass
class AgentResult:
    """Agent执行结果"""
    agent_name: str
    task_type: str
    result: Any
    confidence: float
    execution_time: float
    metadata: Dict[str, Any] = None


class UnifiedManager:
    """
    统一管理器：负责多智能体的协调、调度和并发推理
    
    核心功能：
    1. 预训练模型的统一配置和管理
    2. 多Agent的并发调度和执行
    3. 任务分发和结果聚合
    4. 性能监控和资源管理
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化统一管理器
        
        Args:
            config: 配置字典，如果为None则从config.json加载
        """
        self.config = config or load_config()
        self.logger = setup_logger('UnifiedManager')
        
        # 设备配置
        self.device = self._setup_device()
        
        # Agent注册表
        self.agents = {}
        
        # 任务队列和执行器
        self.task_queue = []
        self.max_workers = self.config.get('max_workers', 4)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # 性能监控
        self.metrics = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'total_time': 0.0
        }
        
        # Batch处理器
        self.batch_processor = BatchProcessor(config=self.config, device=self.device)
        
        self.logger.info(f"UnifiedManager初始化完成，设备: {self.device}, 最大并发数: {self.max_workers}")
    
    def _setup_device(self) -> torch.device:
        """设置计算设备"""
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            self.logger.info(f"使用GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            self.logger.info("使用CPU")
        return device
    
    def register_agent(self, agent_name: str, agent_instance):
        """
        注册Agent到管理器
        
        Args:
            agent_name: Agent名称
            agent_instance: Agent实例
        """
        self.agents[agent_name] = agent_instance
        self.logger.info(f"注册Agent: {agent_name}")
    
    def dispatch_task(self, task: AgentTask) -> AgentResult:
        """
        分发单个任务给对应的Agent
        
        Args:
            task: Agent任务
            
        Returns:
            Agent执行结果
        """
        start_time = time.time()
        
        try:
            if task.agent_name not in self.agents:
                raise ValueError(f"Agent {task.agent_name} 未注册")
            
            agent = self.agents[task.agent_name]
            
            # 执行Agent任务
            result = agent.execute(task)
            
            execution_time = time.time() - start_time
            
            self.metrics['completed_tasks'] += 1
            self.metrics['total_time'] += execution_time
            
            return AgentResult(
                agent_name=task.agent_name,
                task_type=task.task_type,
                result=result,
                confidence=getattr(result, 'confidence', 1.0) if hasattr(result, 'confidence') else 1.0,
                execution_time=execution_time,
                metadata=task.metadata
            )
            
        except Exception as e:
            self.logger.error(f"任务执行失败 [{task.agent_name}/{task.task_type}]: {str(e)}")
            self.metrics['failed_tasks'] += 1
            
            return AgentResult(
                agent_name=task.agent_name,
                task_type=task.task_type,
                result=None,
                confidence=0.0,
                execution_time=time.time() - start_time,
                metadata={'error': str(e)}
            )
    
    def dispatch_parallel(self, tasks: List[AgentTask]) -> List[AgentResult]:
        """
        并发执行多个任务
        
        Args:
            tasks: 任务列表
            
        Returns:
            执行结果列表
        """
        self.logger.info(f"并发执行 {len(tasks)} 个任务")
        self.metrics['total_tasks'] += len(tasks)
        
        # 按优先级排序
        tasks = sorted(tasks, key=lambda x: x.priority, reverse=True)
        
        results = []
        futures = {}
        
        # 提交任务到线程池
        for task in tasks:
            future = self.executor.submit(self.dispatch_task, task)
            futures[future] = task
        
        # 收集结果
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                task = futures[future]
                self.logger.error(f"任务 {task.agent_name} 执行异常: {str(e)}")
                results.append(AgentResult(
                    agent_name=task.agent_name,
                    task_type=task.task_type,
                    result=None,
                    confidence=0.0,
                    execution_time=0.0,
                    metadata={'error': str(e)}
                ))
        
        return results
    
    async def dispatch_async(self, tasks: List[AgentTask]) -> List[AgentResult]:
        """
        异步并发执行多个任务
        
        Args:
            tasks: 任务列表
            
        Returns:
            执行结果列表
        """
        self.logger.info(f"异步并发执行 {len(tasks)} 个任务")
        self.metrics['total_tasks'] += len(tasks)
        
        # 创建异步任务
        async_tasks = []
        for task in tasks:
            if hasattr(self.agents.get(task.agent_name), 'execute_async'):
                async_tasks.append(self._execute_async(task))
            else:
                # 对于不支持异步的Agent，使用同步执行
                async_tasks.append(asyncio.to_thread(self.dispatch_task, task))
        
        # 等待所有任务完成
        results = await asyncio.gather(*async_tasks, return_exceptions=True)
        
        # 处理异常结果
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                task = tasks[i]
                self.logger.error(f"异步任务 {task.agent_name} 执行异常: {str(result)}")
                processed_results.append(AgentResult(
                    agent_name=task.agent_name,
                    task_type=task.task_type,
                    result=None,
                    confidence=0.0,
                    execution_time=0.0,
                    metadata={'error': str(result)}
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _execute_async(self, task: AgentTask) -> AgentResult:
        """异步执行单个任务"""
        start_time = time.time()
        
        try:
            agent = self.agents[task.agent_name]
            result = await agent.execute_async(task)
            
            execution_time = time.time() - start_time
            self.metrics['completed_tasks'] += 1
            self.metrics['total_time'] += execution_time
            
            return AgentResult(
                agent_name=task.agent_name,
                task_type=task.task_type,
                result=result,
                confidence=getattr(result, 'confidence', 1.0),
                execution_time=execution_time,
                metadata=task.metadata
            )
            
        except Exception as e:
            self.logger.error(f"异步任务执行失败 [{task.agent_name}]: {str(e)}")
            self.metrics['failed_tasks'] += 1
            
            return AgentResult(
                agent_name=task.agent_name,
                task_type=task.task_type,
                result=None,
                confidence=0.0,
                execution_time=time.time() - start_time,
                metadata={'error': str(e)}
            )
    
    def aggregate_results(self, results: List[AgentResult], 
                         strategy: str = 'weighted_average') -> Dict[str, Any]:
        """
        聚合多个Agent的结果
        
        Args:
            results: Agent结果列表
            strategy: 聚合策略 ('weighted_average', 'max_confidence', 'voting')
            
        Returns:
            聚合后的结果
        """
        if not results:
            return {'result': None, 'confidence': 0.0}
        
        if strategy == 'weighted_average':
            # 基于置信度的加权平均
            total_confidence = sum(r.confidence for r in results)
            if total_confidence == 0:
                return {'result': results[0].result, 'confidence': 0.0}
            
            # 对于数值结果进行加权
            if all(isinstance(r.result, (int, float, torch.Tensor)) for r in results if r.result is not None):
                weighted_sum = sum(r.result * r.confidence for r in results if r.result is not None)
                final_result = weighted_sum / total_confidence
                return {
                    'result': final_result,
                    'confidence': total_confidence / len(results),
                    'individual_results': [(r.agent_name, r.result, r.confidence) for r in results]
                }
        
        elif strategy == 'max_confidence':
            # 选择置信度最高的结果
            best_result = max(results, key=lambda r: r.confidence)
            return {
                'result': best_result.result,
                'confidence': best_result.confidence,
                'agent': best_result.agent_name
            }
        
        elif strategy == 'voting':
            # 投票机制（适用于分类任务）
            from collections import Counter
            votes = [r.result for r in results if r.result is not None]
            if not votes:
                return {'result': None, 'confidence': 0.0}
            
            vote_counts = Counter(votes)
            most_common = vote_counts.most_common(1)[0]
            return {
                'result': most_common[0],
                'confidence': most_common[1] / len(votes),
                'vote_distribution': dict(vote_counts)
            }
        
        return {'result': results[0].result, 'confidence': results[0].confidence}
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        avg_time = self.metrics['total_time'] / max(self.metrics['completed_tasks'], 1)
        success_rate = self.metrics['completed_tasks'] / max(self.metrics['total_tasks'], 1)
        
        return {
            'total_tasks': self.metrics['total_tasks'],
            'completed_tasks': self.metrics['completed_tasks'],
            'failed_tasks': self.metrics['failed_tasks'],
            'success_rate': success_rate,
            'average_execution_time': avg_time,
            'total_execution_time': self.metrics['total_time']
        }
    
    def reset_metrics(self):
        """重置性能指标"""
        self.metrics = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'total_time': 0.0
        }
        self.logger.info("性能指标已重置")
    
    def shutdown(self):
        """关闭管理器，清理资源"""
        self.logger.info("正在关闭UnifiedManager...")
        self.executor.shutdown(wait=True)
        self.logger.info("UnifiedManager已关闭")
    
    def __del__(self):
        """析构函数"""
        try:
            self.shutdown()
        except:
            pass

