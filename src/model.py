"""
MAS4TS: Multi-Agent System for Time Series Analysis
模型入口文件，整合所有组件
"""

import torch
import asyncio
from typing import Dict, Any, Optional, List
from pathlib import Path
import sys

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.base import UnifiedManager, BatchProcessor
from src.agents import (
    ManagerAgent, DataAnalyzerAgent, VisualAnchorAgent,
    NumerologicAdapterAgent, KnowledgeRetrieverAgent, TaskExecutorAgent
)
from src.base.processor import BatchData
from src.utils.logger import setup_logger
import logging

logger = logging.getLogger(__name__)


class MAS4TS:
    """
    MAS4TS: Multi-Agent System for Time Series Analysis
    
    核心创新:
    1. 视觉锚定 (Visual Anchoring): 将时序转为图像，生成预测锚点
    2. 数值推理 (Numerical Reasoning): 融合多模态信息进行精确推理
    3. 多智能体协作 (Multi-Agent Collaboration): 并发处理，统一决策
    
    支持的任务:
    - Long-term Forecasting
    - Short-term Forecasting  
    - Classification
    - Imputation
    - Anomaly Detection
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化MAS4TS系统
        
        Args:
            config: 配置字典，包含模型、agents、数据等配置
        """
        self.config = config
        self.device = config.get('device', 'cpu')
        
        # 设置日志
        log_level = config.get('log_level', 'INFO')
        setup_logger(log_level)
        
        logger.info("=" * 50)
        logger.info("Initializing MAS4TS System")
        logger.info("=" * 50)
        
        # 初始化核心组件
        self.unified_manager = UnifiedManager(config)
        self.batch_processor = BatchProcessor(config)
        
        # 初始化各个agents
        self.agents = self._initialize_agents()
        
        # 管理器agent
        self.manager_agent = self.agents.get('manager', None)
        
        logger.info(f"MAS4TS initialized with {len(self.agents)} agents")
        logger.info(f"Device: {self.device}")
        logger.info("=" * 50)
    
    def _initialize_agents(self) -> Dict[str, Any]:
        """初始化所有agents"""
        agents = {}
        
        agent_config = self.config.copy()
        
        # 创建各个专用agents
        agents['manager'] = ManagerAgent(agent_config)
        agents['data_analyzer'] = DataAnalyzerAgent(agent_config)
        agents['visual_anchor'] = VisualAnchorAgent(agent_config)
        agents['numerologic_adapter'] = NumerologicAdapterAgent(agent_config)
        agents['knowledge_retriever'] = KnowledgeRetrieverAgent(agent_config)
        agents['task_executor'] = TaskExecutorAgent(agent_config)
        
        logger.info(f"Initialized agents: {list(agents.keys())}")
        
        return agents
    
    async def process_async(self, data: torch.Tensor, 
                           task_type: str,
                           config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        异步处理时序数据
        
        Args:
            data: 时序数据 [batch, seq_len, features]
            task_type: 任务类型 ('forecasting', 'classification', 'imputation', 'anomaly_detection')
            config: 额外配置
            
        Returns:
            result: {
                'predictions': torch.Tensor,  # 主要输出
                'confidence': float,  # 置信度
                'metadata': Dict,  # 元数据
                'agent_outputs': Dict  # 各agent的输出
            }
        """
        try:
            logger.info(f"Processing {task_type} task with data shape: {data.shape}")
            
            # 合并配置
            task_config = {**self.config, **(config or {})}
            
            # 1. 数据预处理
            batch_data = BatchData(seq_x=data)
            batch_data = self.batch_processor.preprocess(batch_data)
            
            # 2. 准备管理器输入
            manager_input = {
                'task_type': task_type,
                'data': batch_data.seq_x,
                'config': task_config,
                'agents': self.agents
            }
            
            # 3. 通过管理器执行任务
            if self.manager_agent:
                manager_output = await self.manager_agent.process(manager_input)
                
                if manager_output.success:
                    # 4. 后处理结果
                    result = manager_output.result
                    
                    # 如果需要逆归一化
                    if task_type in ['forecasting', 'imputation']:
                        if isinstance(result, torch.Tensor):
                            result = self.batch_processor.postprocess(result, batch_data.metadata or {})
                    
                    return {
                        'predictions': result,
                        'confidence': manager_output.confidence,
                        'metadata': manager_output.metadata,
                        'success': True
                    }
                else:
                    logger.error("Manager agent processing failed")
                    return {
                        'predictions': None,
                        'confidence': 0.0,
                        'metadata': manager_output.metadata,
                        'success': False
                    }
            else:
                logger.error("Manager agent not initialized")
                return {
                    'predictions': None,
                    'confidence': 0.0,
                    'metadata': {'error': 'Manager agent not initialized'},
                    'success': False
                }
            
        except Exception as e:
            logger.error(f"Error in process_async: {e}")
            import traceback
            traceback.print_exc()
            return {
                'predictions': None,
                'confidence': 0.0,
                'metadata': {'error': str(e)},
                'success': False
            }
    
    def process(self, data: torch.Tensor,
                task_type: str,
                config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        同步处理时序数据（阻塞调用）
        
        Args:
            data: 时序数据 [batch, seq_len, features]
            task_type: 任务类型
            config: 额外配置
            
        Returns:
            result: 处理结果
        """
        return asyncio.run(self.process_async(data, task_type, config))
    
    def forecast(self, data: torch.Tensor, 
                 pred_len: int,
                 config: Optional[Dict] = None) -> torch.Tensor:
        """
        时序预测
        
        Args:
            data: 历史数据 [batch, seq_len, features]
            pred_len: 预测长度
            config: 额外配置
            
        Returns:
            predictions: [batch, pred_len, features]
        """
        task_config = {'pred_len': pred_len}
        if config:
            task_config.update(config)
        
        result = self.process(data, 'forecasting', task_config)
        return result['predictions'] if result['success'] else None
    
    def classify(self, data: torch.Tensor,
                 num_classes: int,
                 config: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        """
        时序分类
        
        Args:
            data: 时序数据 [batch, seq_len, features]
            num_classes: 类别数
            config: 额外配置
            
        Returns:
            result: {
                'predictions': [batch],
                'probabilities': [batch, num_classes]
            }
        """
        task_config = {'num_classes': num_classes}
        if config:
            task_config.update(config)
        
        result = self.process(data, 'classification', task_config)
        return result['predictions'] if result['success'] else None
    
    def impute(self, data: torch.Tensor,
               mask: Optional[torch.Tensor] = None,
               config: Optional[Dict] = None) -> torch.Tensor:
        """
        时序填补
        
        Args:
            data: 带缺失值的数据 [batch, seq_len, features]
            mask: 缺失值掩码 [batch, seq_len, features] (可选)
            config: 额外配置
            
        Returns:
            imputed_data: [batch, seq_len, features]
        """
        task_config = {}
        if mask is not None:
            task_config['mask'] = mask
        if config:
            task_config.update(config)
        
        result = self.process(data, 'imputation', task_config)
        return result['predictions'] if result['success'] else None
    
    def detect_anomaly(self, data: torch.Tensor,
                      threshold: Optional[float] = None,
                      config: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        """
        异常检测
        
        Args:
            data: 时序数据 [batch, seq_len, features]
            threshold: 异常阈值 (可选)
            config: 额外配置
            
        Returns:
            result: {
                'scores': [batch, seq_len],
                'labels': [batch, seq_len]
            }
        """
        task_config = {}
        if threshold is not None:
            task_config['anomaly_threshold'] = threshold
        if config:
            task_config.update(config)
        
        result = self.process(data, 'anomaly_detection', task_config)
        return result['predictions'] if result['success'] else None
    
    def save_checkpoint(self, save_path: str):
        """保存模型检查点"""
        checkpoint = {
            'config': self.config,
            'agents': {}
        }
        
        # 保存各agent的状态（如果有可训练参数）
        for name, agent in self.agents.items():
            if hasattr(agent, 'state_dict'):
                checkpoint['agents'][name] = agent.state_dict()
        
        torch.save(checkpoint, save_path)
        logger.info(f"Checkpoint saved to {save_path}")
    
    def load_checkpoint(self, load_path: str):
        """加载模型检查点"""
        checkpoint = torch.load(load_path, map_location=self.device)
        
        # 加载各agent的状态
        for name, state in checkpoint.get('agents', {}).items():
            if name in self.agents and hasattr(self.agents[name], 'load_state_dict'):
                self.agents[name].load_state_dict(state)
        
        logger.info(f"Checkpoint loaded from {load_path}")
    
    def get_agent_info(self) -> Dict[str, str]:
        """获取各agent的信息"""
        return {
            name: agent.__class__.__name__
            for name, agent in self.agents.items()
        }
    
    def __repr__(self) -> str:
        return f"MAS4TS(agents={len(self.agents)}, device={self.device})"


def create_model(config: Dict[str, Any]) -> MAS4TS:
    """
    创建MAS4TS模型实例的工厂函数
    
    Args:
        config: 配置字典
        
    Returns:
        model: MAS4TS实例
    """
    return MAS4TS(config)


# 预定义配置
DEFAULT_CONFIG = {
    'device': 'cpu',
    'seq_len': 96,
    'pred_len': 96,
    'label_len': 48,
    'use_norm': True,
    'log_level': 'INFO',
    'agent_priority': {
        'data_analyzer': 1,
        'visual_anchor': 2,
        'numerologic_adapter': 3,
        'task_executor': 4
    },
    'use_parallel_agents': True,
    'model_config': {
        'd_model': 512,
        'n_heads': 8,
        'e_layers': 2,
        'd_layers': 1,
        'd_ff': 2048,
        'dropout': 0.1
    },
    'default_model': 'DLinear'
}


if __name__ == '__main__':
    # 测试代码
    print("Testing MAS4TS...")
    
    # 创建配置
    config = DEFAULT_CONFIG.copy()
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建模型
    model = create_model(config)
    
    print(model)
    print(f"Agents: {model.get_agent_info()}")
    
    # 创建测试数据
    batch_size = 4
    seq_len = 96
    features = 7
    pred_len = 96
    
    test_data = torch.randn(batch_size, seq_len, features)
    
    # 测试预测
    print(f"\nTesting forecasting with data shape: {test_data.shape}")
    predictions = model.forecast(test_data, pred_len)
    if predictions is not None:
        print(f"Predictions shape: {predictions.shape}")
    
    print("\nMAS4TS test completed!")

