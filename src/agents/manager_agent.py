"""
ManagerAgent: 多智能体系统的中央调度器
负责任务分配、决策制定和结果整合
"""

import torch
import asyncio
from typing import Dict, Any, List, Optional
from .base_agent_ts import BaseAgentTS, AgentOutput, AgentMessage
import logging

logger = logging.getLogger(__name__)


class ManagerAgent(BaseAgentTS):
    """
    管理器Agent：协调其他agents完成时序分析任务
    
    职责:
    1. 接收用户任务请求，分析任务类型
    2. 制定执行计划，分配子任务给专门的agents
    3. 协调agents之间的通信和数据流
    4. 整合各agent的结果，生成最终输出
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("ManagerAgent", config)
        
        # 任务类型映射
        self.task_types = {
            'classification': self._plan_classification,
            'forecasting': self._plan_forecasting,
            'imputation': self._plan_imputation,
            'anomaly_detection': self._plan_anomaly_detection
        }
        
        # Agent优先级配置
        self.agent_priority = config.get('agent_priority', {
            'data_analyzer': 1,
            'visual_anchor': 2,
            'numerologic_adapter': 3,
            'task_executor': 4
        })
        
        self.use_parallel = config.get('use_parallel_agents', True)
        
    async def process(self, input_data: Dict[str, Any]) -> AgentOutput:
        """
        处理任务请求
        
        Args:
            input_data: {
                'task_type': str,  # 'classification', 'forecasting', etc.
                'data': torch.Tensor,  # 时序数据
                'config': Dict,  # 任务配置
                'agents': Dict[str, BaseAgentTS]  # 可用的agents
            }
        """
        try:
            # 验证输入
            if not self.validate_input(input_data, ['task_type', 'data', 'agents']):
                return AgentOutput(
                    agent_name=self.name,
                    success=False,
                    result=None,
                    confidence=0.0,
                    metadata={'error': 'Invalid input data'}
                )
            
            task_type = input_data['task_type']
            data = input_data['data']
            agents = input_data['agents']
            config = input_data.get('config', {})
            
            self.log_info(f"Processing task: {task_type}")
            
            # 制定执行计划
            plan = await self._create_execution_plan(task_type, data, config, agents)
            
            # 执行计划
            results = await self._execute_plan(plan, agents)
            
            # 整合结果
            final_result = self._integrate_results(results, task_type)
            
            return AgentOutput(
                agent_name=self.name,
                success=True,
                result=final_result,
                confidence=self._calculate_confidence(results),
                metadata={
                    'plan': plan,
                    'agent_results': results
                }
            )
            
        except Exception as e:
            self.log_error(f"Error in processing: {e}")
            return AgentOutput(
                agent_name=self.name,
                success=False,
                result=None,
                confidence=0.0,
                metadata={'error': str(e)}
            )
    
    async def _create_execution_plan(self, task_type: str, data: torch.Tensor, 
                                    config: Dict, agents: Dict) -> Dict[str, Any]:
        """
        根据任务类型创建执行计划
        """
        if task_type in self.task_types:
            plan_func = self.task_types[task_type]
            return await plan_func(data, config, agents)
        else:
            # 默认计划
            return await self._plan_default(data, config, agents)
    
    async def _plan_forecasting(self, data: torch.Tensor, config: Dict, 
                               agents: Dict) -> Dict[str, Any]:
        """制定预测任务的执行计划"""
        plan = {
            'task_type': 'forecasting',
            'stages': [
                {
                    'stage': 1,
                    'agents': ['data_analyzer'],
                    'parallel': False,
                    'input': {'data': data, 'config': config},
                    'output_keys': ['processed_data', 'data_features']
                },
                {
                    'stage': 2,
                    'agents': ['visual_anchor', 'knowledge_retriever'],
                    'parallel': True,
                    'input': {'data': 'processed_data', 'features': 'data_features'},
                    'output_keys': ['visual_anchors', 'semantic_priors']
                },
                {
                    'stage': 3,
                    'agents': ['numerologic_adapter'],
                    'parallel': False,
                    'input': {
                        'data': 'processed_data',
                        'anchors': 'visual_anchors',
                        'priors': 'semantic_priors'
                    },
                    'output_keys': ['adapted_features']
                },
                {
                    'stage': 4,
                    'agents': ['task_executor'],
                    'parallel': False,
                    'input': {
                        'data': 'processed_data',
                        'features': 'adapted_features',
                        'task': 'forecasting'
                    },
                    'output_keys': ['predictions']
                }
            ]
        }
        return plan
    
    async def _plan_classification(self, data: torch.Tensor, config: Dict,
                                   agents: Dict) -> Dict[str, Any]:
        """制定分类任务的执行计划"""
        plan = {
            'task_type': 'classification',
            'stages': [
                {
                    'stage': 1,
                    'agents': ['data_analyzer', 'visual_anchor'],
                    'parallel': True,
                    'input': {'data': data, 'config': config},
                    'output_keys': ['data_features', 'visual_features']
                },
                {
                    'stage': 2,
                    'agents': ['task_executor'],
                    'parallel': False,
                    'input': {
                        'data': data,
                        'features': ['data_features', 'visual_features'],
                        'task': 'classification'
                    },
                    'output_keys': ['class_predictions']
                }
            ]
        }
        return plan
    
    async def _plan_imputation(self, data: torch.Tensor, config: Dict,
                              agents: Dict) -> Dict[str, Any]:
        """制定填补任务的执行计划"""
        plan = {
            'task_type': 'imputation',
            'stages': [
                {
                    'stage': 1,
                    'agents': ['data_analyzer'],
                    'parallel': False,
                    'input': {'data': data, 'config': config},
                    'output_keys': ['missing_info', 'data_features']
                },
                {
                    'stage': 2,
                    'agents': ['visual_anchor', 'knowledge_retriever'],
                    'parallel': True,
                    'input': {'data': data, 'missing_info': 'missing_info'},
                    'output_keys': ['visual_context', 'semantic_hints']
                },
                {
                    'stage': 3,
                    'agents': ['task_executor'],
                    'parallel': False,
                    'input': {
                        'data': data,
                        'context': ['visual_context', 'semantic_hints'],
                        'task': 'imputation'
                    },
                    'output_keys': ['imputed_data']
                }
            ]
        }
        return plan
    
    async def _plan_anomaly_detection(self, data: torch.Tensor, config: Dict,
                                     agents: Dict) -> Dict[str, Any]:
        """制定异常检测任务的执行计划"""
        plan = {
            'task_type': 'anomaly_detection',
            'stages': [
                {
                    'stage': 1,
                    'agents': ['data_analyzer', 'visual_anchor'],
                    'parallel': True,
                    'input': {'data': data, 'config': config},
                    'output_keys': ['statistical_anomalies', 'visual_anomalies']
                },
                {
                    'stage': 2,
                    'agents': ['task_executor'],
                    'parallel': False,
                    'input': {
                        'data': data,
                        'anomalies': ['statistical_anomalies', 'visual_anomalies'],
                        'task': 'anomaly_detection'
                    },
                    'output_keys': ['anomaly_scores']
                }
            ]
        }
        return plan
    
    async def _plan_default(self, data: torch.Tensor, config: Dict,
                           agents: Dict) -> Dict[str, Any]:
        """默认执行计划"""
        return {
            'task_type': 'default',
            'stages': [
                {
                    'stage': 1,
                    'agents': ['data_analyzer'],
                    'parallel': False,
                    'input': {'data': data, 'config': config},
                    'output_keys': ['processed_data']
                },
                {
                    'stage': 2,
                    'agents': ['task_executor'],
                    'parallel': False,
                    'input': {'data': 'processed_data', 'task': 'default'},
                    'output_keys': ['result']
                }
            ]
        }
    
    async def _execute_plan(self, plan: Dict[str, Any], 
                           agents: Dict[str, BaseAgentTS]) -> Dict[str, Any]:
        """
        执行计划
        """
        results = {}
        
        for stage_info in plan['stages']:
            stage = stage_info['stage']
            agent_names = stage_info['agents']
            parallel = stage_info['parallel']
            stage_input = self._prepare_stage_input(stage_info['input'], results)
            
            self.log_info(f"Executing stage {stage} with agents: {agent_names}")
            
            if parallel and len(agent_names) > 1:
                # 并行执行
                tasks = []
                for agent_name in agent_names:
                    if agent_name in agents:
                        agent = agents[agent_name]
                        tasks.append(agent.process(stage_input))
                
                stage_results = await asyncio.gather(*tasks)
                
                # 合并结果
                for i, agent_name in enumerate(agent_names):
                    output = stage_results[i]
                    output_keys = stage_info['output_keys']
                    if len(output_keys) > i:
                        results[output_keys[i]] = output.result
            else:
                # 顺序执行
                for i, agent_name in enumerate(agent_names):
                    if agent_name in agents:
                        agent = agents[agent_name]
                        output = await agent.process(stage_input)
                        
                        output_keys = stage_info['output_keys']
                        if len(output_keys) > i:
                            results[output_keys[i]] = output.result
        
        return results
    
    def _prepare_stage_input(self, input_spec: Dict, previous_results: Dict) -> Dict[str, Any]:
        """
        准备阶段输入，解析引用previous_results中的值
        """
        stage_input = {}
        for key, value in input_spec.items():
            if isinstance(value, str) and value in previous_results:
                stage_input[key] = previous_results[value]
            elif isinstance(value, list):
                stage_input[key] = [previous_results.get(v, v) for v in value]
            else:
                stage_input[key] = value
        return stage_input
    
    def _integrate_results(self, results: Dict[str, Any], task_type: str) -> Any:
        """
        整合各agent的结果
        """
        # 根据任务类型选择最终结果
        if task_type == 'forecasting':
            return results.get('predictions', None)
        elif task_type == 'classification':
            return results.get('class_predictions', None)
        elif task_type == 'imputation':
            return results.get('imputed_data', None)
        elif task_type == 'anomaly_detection':
            return results.get('anomaly_scores', None)
        else:
            return results.get('result', results)
    
    def _calculate_confidence(self, results: Dict[str, Any]) -> float:
        """
        计算整体置信度
        """
        # 简单平均（可以根据实际情况加权）
        confidences = []
        for value in results.values():
            if isinstance(value, dict) and 'confidence' in value:
                confidences.append(value['confidence'])
        
        if confidences:
            return sum(confidences) / len(confidences)
        return 0.8  # 默认置信度

