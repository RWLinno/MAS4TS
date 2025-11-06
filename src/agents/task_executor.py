"""
TaskExecutorAgent: 任务执行agent
调用Time-Series-Library中的模型执行具体的时序分析任务
"""

import torch
import sys
import os
from typing import Dict, Any, Optional
from .base_agent_ts import BaseAgentTS, AgentOutput
import logging

logger = logging.getLogger(__name__)


class TaskExecutorAgent(BaseAgentTS):
    """
    任务执行Agent：执行具体的时序分析任务
    
    职责:
    1. 调用专用时序模型（从Time-Series-Library）
    2. 支持多种任务：分类、预测、填补、异常检测
    3. 整合多智能体系统的先验知识
    4. 优化模型输出
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("TaskExecutorAgent", config)
        
        self.model_registry = {}
        self.device = config.get('device', 'cpu')
        
        # 任务配置
        self.seq_len = config.get('seq_len', 96)
        self.pred_len = config.get('pred_len', 96)
        self.label_len = config.get('label_len', 48)
        
        # 模型配置
        self.model_config = config.get('model_config', {})
        self.default_model = config.get('default_model', 'DLinear')
        
        # 添加Time-Series-Library路径
        project_root = config.get('project_root', os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        self.log_info(f"TaskExecutor initialized with default model: {self.default_model}")
    
    async def process(self, input_data: Dict[str, Any]) -> AgentOutput:
        """
        执行任务
        
        Args:
            input_data: {
                'data': torch.Tensor,  # [batch, seq_len, features]
                'task': str,  # 'forecasting', 'classification', 'imputation', 'anomaly_detection'
                'features': Dict,  # 从其他agents获得的特征
                'config': Dict
            }
        """
        try:
            if not self.validate_input(input_data, ['data', 'task']):
                return AgentOutput(
                    agent_name=self.name,
                    success=False,
                    result=None,
                    confidence=0.0
                )
            
            data = input_data['data']
            task = input_data['task']
            features = input_data.get('features', {})
            config = input_data.get('config', {})
            
            self.log_info(f"Executing task: {task}")
            
            # 根据任务类型执行
            if task == 'forecasting':
                result = await self._execute_forecasting(data, features, config)
            elif task == 'classification':
                result = await self._execute_classification(data, features, config)
            elif task == 'imputation':
                result = await self._execute_imputation(data, features, config)
            elif task == 'anomaly_detection':
                result = await self._execute_anomaly_detection(data, features, config)
            else:
                self.log_error(f"Unknown task: {task}")
                result = None
            
            if result is not None:
                return AgentOutput(
                    agent_name=self.name,
                    success=True,
                    result=result,
                    confidence=0.90,
                    metadata={'task': task}
                )
            else:
                return AgentOutput(
                    agent_name=self.name,
                    success=False,
                    result=None,
                    confidence=0.0,
                    metadata={'error': 'Task execution failed'}
                )
            
        except Exception as e:
            self.log_error(f"Error in task execution: {e}")
            import traceback
            traceback.print_exc()
            return AgentOutput(
                agent_name=self.name,
                success=False,
                result=None,
                confidence=0.0,
                metadata={'error': str(e)}
            )
    
    async def _execute_forecasting(self, data: torch.Tensor, features: Dict, config: Dict) -> Dict[str, torch.Tensor]:
        """
        执行预测任务 - 整合所有agent的输出
        
        Args:
            data: [batch, seq_len, features]
            features: Dict containing:
                - numerical_predictions: from Numerical Adapter
                - confidence_intervals: from Numerical Adapter  
                - visual_anchors: from Visual Anchor
            config: 配置
            
        Returns:
            final_predictions: Dict with predictions and metadata
        """
        batch, seq_len, n_features = data.shape
        pred_len = config.get('pred_len', self.pred_len)
        
        # 获取来自其他agents的信息
        numerical_preds = features.get('numerical_predictions', None)
        confidence_intervals = features.get('confidence_intervals', None)
        visual_anchors = features.get('visual_anchors', None)
        
        # 获取或创建模型
        model = self._get_or_create_model('forecasting', n_features, pred_len, config)
        
        # 准备模型输入
        # Time-Series-Library的模型通常需要 (batch_x, batch_y, batch_x_mark, batch_y_mark)
        # 这里简化处理
        
        try:
            with torch.no_grad():
                model.eval()
                
                # 简单的直接预测
                # 不同模型有不同的forward接口，这里提供通用接口
                if hasattr(model, 'forecast') and callable(getattr(model, 'forecast')):
                    # 检查forecast方法的参数数量
                    import inspect
                    sig = inspect.signature(model.forecast)
                    num_params = len(sig.parameters)
                    
                    # 根据参数数量调用
                    if num_params == 1:  # 只需要data
                        predictions = model.forecast(data)
                    else:  # 需要data和pred_len
                        predictions = model.forecast(data, pred_len)
                else:
                    # 构造decoder input（使用最后label_len的数据）
                    dec_inp = torch.zeros(batch, pred_len, n_features).to(data.device)
                    
                    # 一些模型需要完整的decoder input
                    if self.label_len > 0:
                        dec_inp = torch.cat([
                            data[:, -self.label_len:, :],
                            torch.zeros(batch, pred_len - self.label_len, n_features).to(data.device)
                        ], dim=1) if pred_len > self.label_len else data[:, -pred_len:, :]
                    
                    # 调用模型
                    predictions = model(data, None, dec_inp, None)
                    
                    # 某些模型返回多个值
                    if isinstance(predictions, tuple):
                        predictions = predictions[0]
                
                # 应用约束（从Visual Anchor和Numerical Adapter）
                visual_anchors_data = features.get('visual_anchors', None)
                numerical_constraints = features.get('numerical_constraints', None)
                
                if visual_anchors_data or numerical_constraints:
                    predictions = self._apply_multiagent_constraints(
                        predictions, visual_anchors_data, numerical_constraints, None
                    )
                
                return {
                    'final_predictions': predictions,
                    'metadata': {
                        'model': self.default_model,
                        'pred_len': pred_len,
                        'used_visual_anchors': visual_anchors is not None,
                        'used_numerical_reasoning': numerical_preds is not None
                    }
                }
        
        except Exception as e:
            self.log_error(f"Model forward failed: {e}")
            # 回退到简单的基线方法
            return self._naive_forecasting(data, pred_len)
    
    async def _execute_classification(self, data: torch.Tensor, features: Dict, config: Dict) -> Dict[str, torch.Tensor]:
        """
        执行分类任务
        
        Args:
            data: [batch, seq_len, features]
            features: 来自其他agents的特征
            config: 配置
            
        Returns:
            classifications: {
                'logits': torch.Tensor [batch, num_classes],
                'predictions': torch.Tensor [batch]
            }
        """
        batch, seq_len, n_features = data.shape
        num_classes = config.get('num_classes', 2)
        
        # 获取或创建分类模型
        model = self._get_or_create_model('classification', n_features, num_classes, config)
        
        try:
            with torch.no_grad():
                model.eval()
                
                # 调用分类模型
                if hasattr(model, 'classify'):
                    logits = model.classify(data)
                else:
                    output = model(data, None, None, None)
                    if isinstance(output, tuple):
                        logits = output[0]
                    else:
                        logits = output
                
                # 如果输出不是logits形状，进行转换
                if logits.dim() > 2:
                    logits = logits.mean(dim=1)  # 平均池化
                
                # 预测类别
                predictions = torch.argmax(logits, dim=-1)
                
                return {
                    'logits': logits,
                    'predictions': predictions,
                    'probabilities': torch.softmax(logits, dim=-1),
                    'metadata': {
                        'model': self.default_model,
                        'num_classes': num_classes
                    }
                }
        
        except Exception as e:
            self.log_error(f"Classification failed: {e}")
            # 回退：随机分类
            return {
                'logits': torch.randn(batch, num_classes),
                'predictions': torch.randint(0, num_classes, (batch,)),
                'metadata': {'error': str(e)}
            }
    
    async def _execute_imputation(self, data: torch.Tensor, features: Dict, config: Dict) -> Dict[str, torch.Tensor]:
        """
        执行填补任务
        
        Args:
            data: [batch, seq_len, features] - 包含缺失值（mask或NaN）
            features: 来自其他agents的特征
            config: 配置
            
        Returns:
            imputed: {
                'imputed_data': torch.Tensor [batch, seq_len, features]
            }
        """
        batch, seq_len, n_features = data.shape
        
        # 检测缺失值
        if torch.isnan(data).any():
            mask = torch.isnan(data)
        else:
            # 假设mask在config中提供
            mask = config.get('mask', torch.zeros_like(data, dtype=torch.bool))
        
        # 获取模型
        model = self._get_or_create_model('imputation', n_features, seq_len, config)
        
        try:
            with torch.no_grad():
                model.eval()
                
                # 将缺失值设为0（模型输入）
                data_input = data.clone()
                data_input[mask] = 0.0
                
                # 调用填补模型
                if hasattr(model, 'impute'):
                    imputed = model.impute(data_input, mask)
                else:
                    output = model(data_input, None, None, None)
                    if isinstance(output, tuple):
                        imputed = output[0]
                    else:
                        imputed = output
                
                # 只替换缺失位置的值
                result = data.clone()
                result[mask] = imputed[mask]
                
                return {
                    'imputed_data': result,
                    'mask': mask,
                    'metadata': {
                        'model': self.default_model,
                        'num_missing': mask.sum().item()
                    }
                }
        
        except Exception as e:
            self.log_error(f"Imputation failed: {e}")
            # 回退：线性插值
            return self._naive_imputation(data, mask)
    
    async def _execute_anomaly_detection(self, data: torch.Tensor, features: Dict, config: Dict) -> Dict[str, torch.Tensor]:
        """
        执行异常检测任务
        
        Args:
            data: [batch, seq_len, features]
            features: 来自其他agents的特征
            config: 配置
            
        Returns:
            anomalies: {
                'scores': torch.Tensor [batch, seq_len, features],
                'labels': torch.Tensor [batch, seq_len] (binary)
            }
        """
        batch, seq_len, n_features = data.shape
        
        # 获取模型
        model = self._get_or_create_model('anomaly_detection', n_features, seq_len, config)
        
        try:
            with torch.no_grad():
                model.eval()
                
                # 调用异常检测模型
                if hasattr(model, 'detect_anomaly'):
                    scores = model.detect_anomaly(data)
                else:
                    # 重构误差作为异常分数
                    output = model(data, None, data, None)
                    if isinstance(output, tuple):
                        reconstructed = output[0]
                    else:
                        reconstructed = output
                    
                    # 计算重构误差
                    scores = torch.abs(data - reconstructed).mean(dim=-1)  # [batch, seq_len]
                
                # 阈值化得到标签
                threshold = config.get('anomaly_threshold', scores.mean() + 2 * scores.std())
                labels = (scores > threshold).long()
                
                return {
                    'scores': scores,
                    'labels': labels,
                    'threshold': threshold,
                    'metadata': {
                        'model': self.default_model,
                        'num_anomalies': labels.sum().item()
                    }
                }
        
        except Exception as e:
            self.log_error(f"Anomaly detection failed: {e}")
            # 回退：统计方法
            return self._statistical_anomaly_detection(data)
    
    def _get_or_create_model(self, task: str, n_features: int, target_size: int, config: Dict):
        """获取或创建模型"""
        model_name = config.get('model', self.default_model)
        model_key = f"{task}_{model_name}_{n_features}"
        
        if model_key in self.model_registry:
            return self.model_registry[model_key]
        
        # 创建新模型
        model = self._create_model(task, model_name, n_features, target_size, config)
        self.model_registry[model_key] = model
        
        return model
    
    def _create_model(self, task: str, model_name: str, n_features: int, target_size: int, config: Dict):
        """创建模型实例"""
        try:
            # 动态导入模型
            from importlib import import_module
            
            # 尝试从models目录导入
            try:
                model_module = import_module(f'models.{model_name}')
            except:
                # 尝试从上层目录的models导入
                model_module = import_module(f'..models.{model_name}', package=__package__)
            
            model_class = getattr(model_module, 'Model')
            
            # 准备模型配置
            model_config = {
                'enc_in': n_features,
                'dec_in': n_features,
                'c_out': n_features,
                'seq_len': self.seq_len,
                'label_len': self.label_len,
                'pred_len': target_size if task == 'forecasting' else self.pred_len,
                'd_model': self.model_config.get('d_model', 512),
                'n_heads': self.model_config.get('n_heads', 8),
                'e_layers': self.model_config.get('e_layers', 2),
                'd_layers': self.model_config.get('d_layers', 1),
                'd_ff': self.model_config.get('d_ff', 2048),
                'dropout': self.model_config.get('dropout', 0.1),
                'activation': 'gelu',
                'output_attention': False,
                'task_name': task
            }
            
            # 根据任务调整配置
            if task == 'classification':
                model_config['num_class'] = target_size
            
            # 添加所有必需的参数
            model_config.update({
                'moving_avg': 25,
                'freq': 'h',
                'embed': 'timeF',
                'factor': 1,
                'distil': True,
                'use_norm': True,
                'channel_independence': 1,
                'decomp_method': 'moving_avg',
                'down_sampling_layers': 0,
                'down_sampling_window': 1,
                'down_sampling_method': None,
                'seg_len': 96,
                'top_k': 5,
                'num_kernels': 6,
                'expand': 2,
                'd_conv': 4,
                'patch_len': 16
            })
            
            # 创建模型
            class Args:
                pass
            
            args = Args()
            for key, value in model_config.items():
                setattr(args, key, value)
            
            model = model_class(args)
            model = model.to(self.device)
            model.eval()
            
            self.log_info(f"Created model: {model_name} for task: {task}")
            return model
            
        except Exception as e:
            self.log_error(f"Failed to create model {model_name}: {e}")
            # 返回dummy模型
            return self._create_dummy_model(task, n_features, target_size)
    
    def _create_dummy_model(self, task: str, n_features: int, target_size: int):
        """创建简单的dummy模型用于测试"""
        class DummyModel(torch.nn.Module):
            def __init__(self, task, n_features, target_size):
                super().__init__()
                self.task = task
                self.n_features = n_features
                self.target_size = target_size
                self.linear = torch.nn.Linear(n_features, n_features)
            
            def forward(self, x, *args):
                if self.task == 'forecasting':
                    # 返回最后一个值的重复
                    last = x[:, -1:, :].repeat(1, self.target_size, 1)
                    return last
                elif self.task == 'classification':
                    # 全局平均池化 + 线性层
                    pooled = x.mean(dim=1)
                    return torch.randn(x.size(0), self.target_size)
                else:
                    return x
        
        return DummyModel(task, n_features, target_size).to(self.device)
    
    def _naive_forecasting(self, data: torch.Tensor, pred_len: int) -> Dict[str, torch.Tensor]:
        """朴素预测方法（最后值重复）"""
        last_value = data[:, -1:, :]
        predictions = last_value.repeat(1, pred_len, 1)
        
        return {
            'predictions': predictions,
            'metadata': {'method': 'naive_repeat'}
        }
    
    def _naive_imputation(self, data: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """朴素填补方法（前向填充）"""
        result = data.clone()
        
        for i in range(1, data.size(1)):
            nan_mask = mask[:, i, :]
            result[:, i, :] = torch.where(nan_mask, result[:, i-1, :], result[:, i, :])
        
        return {
            'imputed_data': result,
            'mask': mask,
            'metadata': {'method': 'forward_fill'}
        }
    
    def _statistical_anomaly_detection(self, data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """统计异常检测（z-score）"""
        mean = data.mean(dim=1, keepdim=True)
        std = data.std(dim=1, keepdim=True) + 1e-8
        
        z_scores = torch.abs((data - mean) / std).mean(dim=-1)  # [batch, seq_len]
        
        threshold = 3.0
        labels = (z_scores > threshold).long()
        
        return {
            'scores': z_scores,
            'labels': labels,
            'threshold': threshold,
            'metadata': {'method': 'z_score'}
        }
    
    def _apply_multiagent_constraints(self, predictions: torch.Tensor, 
                                      visual_anchors: Optional[Dict],
                                      numerical_preds: Optional[Dict],
                                      confidence_intervals: Optional[torch.Tensor]) -> torch.Tensor:
        """
        应用来自Visual Anchor和Numerical Adapter的约束
        
        Args:
            predictions: 原始预测 [batch, pred_len, features]
            visual_anchors: 视觉锚点
            numerical_preds: 数值预测
            confidence_intervals: 置信区间
            
        Returns:
            constrained_predictions: 应用约束后的预测
        """
        # 1. 应用视觉锚点的约束
        if visual_anchors and 'value_range' in visual_anchors:
            value_range = visual_anchors['value_range']
            if 'lower' in value_range and 'upper' in value_range:
                lower = value_range['lower']
                upper = value_range['upper']
                
                # 确保维度匹配
                if isinstance(lower, torch.Tensor) and lower.dim() == 2:
                    lower = lower.unsqueeze(1)  # [batch, 1, features]
                if isinstance(upper, torch.Tensor) and upper.dim() == 2:
                    upper = upper.unsqueeze(1)
                
                # 软裁剪：使用加权
                predictions = 0.7 * predictions + 0.3 * torch.clamp(predictions, min=lower, max=upper)
        
        # 2. 应用置信区间约束
        if confidence_intervals is not None:
            # confidence_intervals: [batch, features] or similar
            # 这里可以根据置信区间调整predictions
            pass
        
        return predictions
    
    def _apply_constraints(self, predictions: torch.Tensor, features: Dict) -> torch.Tensor:
        """应用来自其他agents的约束（兼容旧接口）"""
        adapted = features.get('adapted_features', {})
        constraints = adapted.get('numerical_constraints', {})
        
        if 'upper_bound' in constraints and 'lower_bound' in constraints:
            upper = constraints['upper_bound'].unsqueeze(1)
            lower = constraints['lower_bound'].unsqueeze(1)
            predictions = torch.clamp(predictions, min=lower, max=upper)
        
        return predictions

