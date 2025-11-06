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
    5. 优化批量并行推理效率
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("ManagerAgent", config)
        
        # 从config.json读取全局parallel配置
        self._load_parallel_config()
        
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
        
        # 允许传入的config覆盖
        self.use_parallel = config.get('use_parallel_agents', self.use_parallel)
        self.enable_batch_parallel = config.get('enable_batch_parallel', self.enable_batch_parallel)
        self.max_parallel_batches = config.get('max_parallel_batches', self.max_parallel_batches)
    
    def _load_parallel_config(self):
        """从config.json加载全局parallel配置"""
        try:
            import json
            from pathlib import Path
            
            config_path = Path(__file__).parent.parent / 'config.json'
            if config_path.exists():
                with open(config_path, 'r') as f:
                    json_config = json.load(f)
                    global_config = json_config.get('global_config', {})
                    parallel_config = global_config.get('parallel_execution', {})
                    
                    self.use_parallel = global_config.get('use_parallel_agents', True)
                    self.enable_batch_parallel = parallel_config.get('enable_batch_parallel', True)
                    self.max_parallel_batches = parallel_config.get('max_parallel_batches', 4)
                    self.enable_batch_splitting = parallel_config.get('enable_batch_splitting', True)
                    self.num_splits = parallel_config.get('num_splits', 2)
                    self.enable_concurrent_llm = parallel_config.get('enable_concurrent_llm', True)
                    self.max_concurrent_requests = parallel_config.get('max_concurrent_requests', 5)
                    
                    self.log_info(f"Loaded parallel config: batch_parallel={self.enable_batch_parallel}, max_batches={self.max_parallel_batches}")
            else:
                # 默认值
                self.use_parallel = True
                self.enable_batch_parallel = True
                self.max_parallel_batches = 4
                self.enable_batch_splitting = True
                self.num_splits = 2
                self.enable_concurrent_llm = True
                self.max_concurrent_requests = 5
                
        except Exception as e:
            self.log_error(f"Failed to load parallel config: {e}, using defaults")
            self.use_parallel = True
            self.enable_batch_parallel = True
            self.max_parallel_batches = 4
            self.enable_batch_splitting = True
            self.num_splits = 2
            self.enable_concurrent_llm = True
            self.max_concurrent_requests = 5
        
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
        """
        制定预测任务的执行计划 - 完整的4个Agent流程:
        1. Data Analyzer: 分析数据+生成plot图
        2. Visual Anchor: VLM读图+生成锚点
        3. Numerical Adapter: LLM数值推理+并发ensemble
        4. Task Executor: 最终预测输出
        """
        plan = {
            'task_type': 'forecasting',
            'stages': [
                {
                    'stage': 1,
                    'name': 'Data Analysis & Visualization',
                    'agents': ['data_analyzer'],
                    'parallel': False,
                    'input': {
                        'data': data,
                        'original_data': config.get('original_data', data),
                        'config': config,
                        'task': 'full_analysis_with_plot'
                    },
                    'output_keys': ['data_features', 'plot_path', 'statistics_text']
                },
                {
                    'stage': 2,
                    'name': 'Visual Anchoring with VLM',
                    'agents': ['visual_anchor'],
                    'parallel': False,
                    'input': {
                        'plot_path': 'plot_path',
                        'statistics_text': 'statistics_text',
                        'data_features': 'data_features',
                        'task': 'forecasting',
                        'pred_len': config.get('pred_len', 96),
                        'batch_idx': config.get('batch_idx', 0)
                    },
                    'output_keys': ['visual_anchors', 'semantic_priors', 'anchor_image_path']
                },
                {
                    'stage': 3,
                    'name': 'Numerical Reasoning with LLM Ensemble',
                    'agents': ['numerologic_adapter'],
                    'parallel': False,
                    'input': {
                        'data': data,
                        'data_features': 'data_features',
                        'anchors': 'visual_anchors',
                        'semantic_priors': 'semantic_priors',
                        'statistics_text': 'statistics_text',
                        'use_parallel_llm': True,
                        'num_llm_models': 3,
                        'batch_idx': config.get('batch_idx', 0)
                    },
                    'output_keys': ['adapted_features', 'numerical_constraints']
                },
                {
                    'stage': 4,
                    'name': 'Task Execution',
                    'agents': ['task_executor'],
                    'parallel': False,
                    'input': {
                        'data': data,
                        'task': 'forecasting',
                        'features': {
                            'adapted_features': 'adapted_features',
                            'numerical_constraints': 'numerical_constraints',
                            'visual_anchors': 'visual_anchors'
                        },
                        'config': {
                        'pred_len': config.get('pred_len', 96)
                        }
                    },
                    'output_keys': ['final_predictions']
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
        执行计划 - 支持批量并行优化
        """
        results = {}
        
        for stage_info in plan['stages']:
            stage = stage_info['stage']
            agent_names = stage_info['agents']
            parallel = stage_info['parallel']
            stage_input = self._prepare_stage_input(stage_info['input'], results)
            
            self.log_info(f"Executing stage {stage} with agents: {agent_names}")
            
            # 检查是否可以批量并行处理
            can_batch_parallel = self.enable_batch_parallel and 'data' in stage_input
            
            if can_batch_parallel and isinstance(stage_input.get('data'), torch.Tensor):
                # 批量并行处理
                data = stage_input['data']
                batch_size = data.size(0)
                
                # 如果batch很大，分割并并行处理
                if batch_size > 8:  # 阈值可配置
                    self.log_info(f"Batch size {batch_size} > 8, enabling batch-level parallelism")
                    stage_output = await self._execute_stage_with_batch_parallel(
                        agent_names, stage_input, stage_info, agents
                    )
                    # 合并批量结果
                    output_keys = stage_info['output_keys']
                    for key in output_keys:
                        if key in stage_output:
                            results[key] = stage_output[key]
                    continue
            
            # 原有逻辑：agent级别的并行或顺序执行
            if parallel and len(agent_names) > 1:
                # 并行执行多个agent
                tasks = []
                for agent_name in agent_names:
                    if agent_name in agents:
                        agent = agents[agent_name]
                        tasks.append(agent.process(stage_input))
                
                stage_results = await asyncio.gather(*tasks)
                
                # 合并结果
                for i, agent_name in enumerate(agent_names):
                    if i >= len(stage_results):
                        continue
                    output = stage_results[i]
                    output_keys = stage_info['output_keys']
                    if isinstance(output.result, dict):
                        # 如果result是字典，展开其内容到对应的output_keys
                        for j, key in enumerate(output_keys):
                            if key in output.result:
                                results[key] = output.result[key]
                            elif j == 0:
                                # 第一个output_key获取整个result字典
                                results[key] = output.result
                    else:
                        # 单个结果
                        if len(output_keys) > 0 and i < len(output_keys):
                            results[output_keys[i]] = output.result
            else:
                # 顺序执行
                for i, agent_name in enumerate(agent_names):
                    if agent_name in agents:
                        agent = agents[agent_name]
                        output = await agent.process(stage_input)
                        
                        output_keys = stage_info['output_keys']
                        # 处理agent的输出
                        if isinstance(output.result, dict):
                            # 展开字典结果到results中
                            for j, key in enumerate(output_keys):
                                if key in output.result:
                                    results[key] = output.result[key]
                                elif j == 0:
                                    # 第一个output_key获取整个result
                                    results[key] = output.result
                        else:
                            # 单个结果
                            if len(output_keys) > 0:
                                results[output_keys[0]] = output.result
        
        return results
    
    async def _execute_stage_with_batch_parallel(self, agent_names: List[str],
                                                  stage_input: Dict,
                                                  stage_info: Dict,
                                                  agents: Dict) -> Dict[str, Any]:
        """
        批量并行执行 - 将大batch分割成小batch并发处理
        """
        data = stage_input['data']
        batch_size = data.size(0)
        
        # 计算每个sub-batch的大小
        num_sub_batches = min(self.max_parallel_batches, max(2, batch_size // 4))
        sub_batch_size = batch_size // num_sub_batches
        
        self.log_info(f"Splitting batch {batch_size} into {num_sub_batches} sub-batches")
        
        # 创建sub-batch任务
        tasks = []
        for i in range(num_sub_batches):
            start_idx = i * sub_batch_size
            end_idx = (i + 1) * sub_batch_size if i < num_sub_batches - 1 else batch_size
            
            # 创建sub-batch输入
            sub_input = stage_input.copy()
            sub_input['data'] = data[start_idx:end_idx]
            if 'batch_idx' in sub_input:
                sub_input['batch_idx'] = i
            
            # 为每个sub-batch创建agent处理任务
            for agent_name in agent_names:
                if agent_name in agents:
                    agent = agents[agent_name]
                    tasks.append((i, agent.process(sub_input)))
        
        # 并发执行所有sub-batch
        all_results = await asyncio.gather(*[task[1] for task in tasks])
        
        # 合并结果
        merged_results = {}
        output_keys = stage_info['output_keys']
        
        for key in output_keys:
            key_results = []
            for result_output in all_results:
                if result_output.success and isinstance(result_output.result, dict):
                    if key in result_output.result:
                        key_results.append(result_output.result[key])
            
            # 合并tensor结果
            if key_results and isinstance(key_results[0], torch.Tensor):
                merged_results[key] = torch.cat(key_results, dim=0)
            elif key_results and isinstance(key_results[0], dict):
                # 对于字典结果，取第一个（假设它们是相同的配置）
                merged_results[key] = key_results[0]
            elif key_results:
                merged_results[key] = key_results[0]
        
        return merged_results
    
    def _prepare_stage_input(self, input_spec: Dict, previous_results: Dict) -> Dict[str, Any]:
        """
        准备阶段输入，解析引用previous_results中的值
        """
        stage_input = {}
        for key, value in input_spec.items():
            if isinstance(value, str) and value in previous_results:
                stage_input[key] = previous_results[value]
            elif isinstance(value, dict):
                # 递归处理嵌套字典
                stage_input[key] = self._prepare_stage_input(value, previous_results)
            elif isinstance(value, list):
                stage_input[key] = [previous_results.get(v, v) if isinstance(v, str) else v for v in value]
            else:
                stage_input[key] = value
        return stage_input
    
    def _integrate_results(self, results: Dict[str, Any], task_type: str) -> Any:
        """
        整合各agent的结果
        """
        # 根据任务类型选择最终结果
        if task_type == 'forecasting':
            return results.get('final_predictions', results.get('predictions', None))
        elif task_type == 'classification':
            return results.get('final_predictions', results.get('class_predictions', None))
        elif task_type == 'imputation':
            return results.get('final_predictions', results.get('imputed_data', None))
        elif task_type == 'anomaly_detection':
            return results.get('final_predictions', results.get('anomaly_scores', None))
        else:
            return results.get('final_predictions', results.get('result', results))
    
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

