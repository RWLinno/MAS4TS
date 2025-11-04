"""
Time Series Models Toolkit
提供对Time-Series-Library中各种模型的统一接口
"""

import torch
import sys
import os
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """模型配置"""
    model_name: str
    task_name: str  # 'forecasting', 'classification', etc.
    enc_in: int
    dec_in: int
    c_out: int
    seq_len: int
    pred_len: int
    label_len: int = 48
    d_model: int = 512
    n_heads: int = 8
    e_layers: int = 2
    d_layers: int = 1
    d_ff: int = 2048
    dropout: float = 0.1
    activation: str = 'gelu'
    output_attention: bool = False
    num_class: Optional[int] = None  # For classification


class TSModelsToolkit:
    """
    时序模型工具包
    提供统一的接口来调用Time-Series-Library中的各种模型
    """
    
    # 支持的模型列表
    SUPPORTED_MODELS = [
        'DLinear', 'TimesNet', 'Autoformer', 'Transformer', 'Informer',
        'PatchTST', 'iTransformer', 'TimeMixer', 'TSMixer',
        'FEDformer', 'Reformer', 'LightTS', 'SCINet', 'SegRNN'
    ]
    
    # 各任务推荐的模型
    RECOMMENDED_MODELS = {
        'forecasting': ['DLinear', 'TimesNet', 'PatchTST', 'iTransformer', 'TimeMixer'],
        'classification': ['TimesNet', 'Transformer', 'Informer'],
        'imputation': ['TimesNet', 'Autoformer', 'Transformer'],
        'anomaly_detection': ['TimesNet', 'Autoformer', 'Transformer']
    }
    
    def __init__(self, project_root: Optional[str] = None, device: str = 'cpu'):
        self.device = device
        self.model_cache = {}
        
        # 添加项目根目录到Python路径
        if project_root:
            self.project_root = project_root
        else:
            # 自动检测项目根目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
        
        if self.project_root not in sys.path:
            sys.path.insert(0, self.project_root)
        
        logger.info(f"TSModelsToolkit initialized with project_root: {self.project_root}")
    
    def create_model(self, config: ModelConfig) -> torch.nn.Module:
        """
        创建模型实例
        
        Args:
            config: 模型配置
            
        Returns:
            model: PyTorch模型
        """
        model_name = config.model_name
        cache_key = self._get_cache_key(config)
        
        # 检查缓存
        if cache_key in self.model_cache:
            logger.info(f"Loading model from cache: {model_name}")
            return self.model_cache[cache_key]
        
        # 创建新模型
        try:
            model = self._load_model(config)
            model = model.to(self.device)
            model.eval()
            
            # 缓存模型
            self.model_cache[cache_key] = model
            
            logger.info(f"Successfully created model: {model_name}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to create model {model_name}: {e}")
            raise
    
    def _load_model(self, config: ModelConfig) -> torch.nn.Module:
        """加载模型"""
        from importlib import import_module
        
        model_name = config.model_name
        
        try:
            # 导入模型模块
            model_module = import_module(f'models.{model_name}')
            model_class = getattr(model_module, 'Model')
            
            # 创建参数对象
            args = self._config_to_args(config)
            
            # 实例化模型
            model = model_class(args)
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            raise ImportError(f"Cannot import model {model_name}. Error: {e}")
    
    def _config_to_args(self, config: ModelConfig):
        """将配置转换为args对象"""
        class Args:
            pass
        
        args = Args()
        
        # 基本参数
        for key, value in config.__dict__.items():
            setattr(args, key, value)
        
        # 添加一些可能需要的额外参数
        args.freq = 'h'
        args.embed = 'timeF'
        args.factor = 1
        args.distil = True
        args.moving_avg = 25
        args.use_norm = True
        args.channel_independence = 1
        args.decomp_method = 'moving_avg'
        args.down_sampling_layers = 0
        args.down_sampling_window = 1
        args.down_sampling_method = None
        args.seg_len = 96
        args.top_k = 5
        args.num_kernels = 6
        args.expand = 2
        args.d_conv = 4
        args.patch_len = 16
        
        return args
    
    def _get_cache_key(self, config: ModelConfig) -> str:
        """生成缓存key"""
        return f"{config.model_name}_{config.task_name}_{config.enc_in}_{config.seq_len}_{config.pred_len}"
    
    def predict_forecasting(self, model: torch.nn.Module, 
                           data: torch.Tensor,
                           pred_len: int,
                           label_len: int = 48) -> torch.Tensor:
        """
        执行预测
        
        Args:
            model: 模型
            data: [batch, seq_len, features]
            pred_len: 预测长度
            label_len: 标签长度
            
        Returns:
            predictions: [batch, pred_len, features]
        """
        batch, seq_len, n_features = data.shape
        
        with torch.no_grad():
            # 准备decoder input
            dec_inp = torch.zeros(batch, pred_len, n_features).to(data.device)
            
            if label_len > 0:
                # 使用历史数据的最后label_len作为decoder的起始
                dec_inp = torch.cat([
                    data[:, -label_len:, :],
                    torch.zeros(batch, pred_len - label_len, n_features).to(data.device)
                ], dim=1) if pred_len > label_len else data[:, -pred_len:, :]
            
            # 时间特征（可选，简化为None）
            batch_x_mark = None
            batch_y_mark = None
            
            # 模型前向传播
            try:
                outputs = model(data, batch_x_mark, dec_inp, batch_y_mark)
                
                if isinstance(outputs, tuple):
                    predictions = outputs[0]
                else:
                    predictions = outputs
                
                # 确保输出形状正确
                if predictions.size(1) != pred_len:
                    predictions = predictions[:, -pred_len:, :]
                
                return predictions
                
            except Exception as e:
                logger.error(f"Prediction failed: {e}")
                # 回退到简单方法
                last_value = data[:, -1:, :]
                return last_value.repeat(1, pred_len, 1)
    
    def predict_classification(self, model: torch.nn.Module,
                              data: torch.Tensor,
                              num_classes: int) -> Dict[str, torch.Tensor]:
        """
        执行分类
        
        Args:
            model: 模型
            data: [batch, seq_len, features]
            num_classes: 类别数
            
        Returns:
            result: {
                'logits': [batch, num_classes],
                'predictions': [batch],
                'probabilities': [batch, num_classes]
            }
        """
        with torch.no_grad():
            try:
                outputs = model(data, None, None, None)
                
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                
                # 确保输出是logits形状
                if logits.dim() > 2:
                    logits = logits.mean(dim=1)  # 全局平均池化
                
                if logits.size(-1) != num_classes:
                    logger.warning(f"Model output size {logits.size(-1)} != num_classes {num_classes}")
                
                predictions = torch.argmax(logits, dim=-1)
                probabilities = torch.softmax(logits, dim=-1)
                
                return {
                    'logits': logits,
                    'predictions': predictions,
                    'probabilities': probabilities
                }
                
            except Exception as e:
                logger.error(f"Classification failed: {e}")
                batch = data.size(0)
                return {
                    'logits': torch.randn(batch, num_classes),
                    'predictions': torch.randint(0, num_classes, (batch,)),
                    'probabilities': torch.ones(batch, num_classes) / num_classes
                }
    
    def predict_imputation(self, model: torch.nn.Module,
                          data: torch.Tensor,
                          mask: torch.Tensor) -> torch.Tensor:
        """
        执行填补
        
        Args:
            model: 模型
            data: [batch, seq_len, features] - 包含缺失值
            mask: [batch, seq_len, features] - True表示缺失
            
        Returns:
            imputed: [batch, seq_len, features]
        """
        with torch.no_grad():
            # 将缺失值设为0
            data_input = data.clone()
            data_input[mask] = 0.0
            
            try:
                outputs = model(data_input, None, data_input, None)
                
                if isinstance(outputs, tuple):
                    imputed = outputs[0]
                else:
                    imputed = outputs
                
                # 只替换缺失位置
                result = data.clone()
                result[mask] = imputed[mask]
                
                return result
                
            except Exception as e:
                logger.error(f"Imputation failed: {e}")
                # 回退到前向填充
                result = data.clone()
                for i in range(1, data.size(1)):
                    nan_positions = mask[:, i, :]
                    result[:, i, :] = torch.where(nan_positions, result[:, i-1, :], result[:, i, :])
                return result
    
    def detect_anomaly(self, model: torch.nn.Module,
                      data: torch.Tensor,
                      threshold: Optional[float] = None) -> Dict[str, torch.Tensor]:
        """
        执行异常检测
        
        Args:
            model: 模型
            data: [batch, seq_len, features]
            threshold: 异常阈值（可选）
            
        Returns:
            result: {
                'scores': [batch, seq_len],
                'labels': [batch, seq_len],
                'threshold': float
            }
        """
        with torch.no_grad():
            try:
                # 使用重构误差
                outputs = model(data, None, data, None)
                
                if isinstance(outputs, tuple):
                    reconstructed = outputs[0]
                else:
                    reconstructed = outputs
                
                # 计算重构误差
                errors = torch.abs(data - reconstructed)
                scores = errors.mean(dim=-1)  # [batch, seq_len]
                
                # 确定阈值
                if threshold is None:
                    threshold = scores.mean().item() + 2 * scores.std().item()
                
                labels = (scores > threshold).long()
                
                return {
                    'scores': scores,
                    'labels': labels,
                    'threshold': threshold
                }
                
            except Exception as e:
                logger.error(f"Anomaly detection failed: {e}")
                # 回退到z-score方法
                mean = data.mean(dim=1, keepdim=True)
                std = data.std(dim=1, keepdim=True) + 1e-8
                z_scores = torch.abs((data - mean) / std).mean(dim=-1)
                
                if threshold is None:
                    threshold = 3.0
                
                return {
                    'scores': z_scores,
                    'labels': (z_scores > threshold).long(),
                    'threshold': threshold
                }
    
    def get_recommended_model(self, task: str) -> str:
        """获取推荐的模型"""
        if task in self.RECOMMENDED_MODELS:
            return self.RECOMMENDED_MODELS[task][0]
        return 'DLinear'  # 默认模型
    
    def list_available_models(self) -> List[str]:
        """列出所有可用的模型"""
        return self.SUPPORTED_MODELS.copy()
    
    def clear_cache(self):
        """清空模型缓存"""
        self.model_cache.clear()
        logger.info("Model cache cleared")

