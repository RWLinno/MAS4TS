"""
BatchProcessor: 处理单个batch的时序数据
负责数据预处理、特征提取和批处理优化
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class BatchData:
    """批数据结构"""
    seq_x: torch.Tensor  # 输入序列 [batch, seq_len, features]
    seq_y: Optional[torch.Tensor] = None  # 目标序列 [batch, pred_len, features]
    seq_x_mark: Optional[torch.Tensor] = None  # 时间戳特征
    seq_y_mark: Optional[torch.Tensor] = None
    metadata: Optional[Dict[str, Any]] = None


class BatchProcessor:
    """批处理器：处理单个batch的数据"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = config.get('device', 'cpu')
        self.task_name = config.get('task_name', 'forecasting')
        
        # 数据配置
        self.seq_len = config.get('seq_len', 96)
        self.pred_len = config.get('pred_len', 96)
        self.label_len = config.get('label_len', 48)
        
        # 归一化参数
        self.use_norm = config.get('use_norm', True)
        self.scaler = None
        
        logger.info(f"BatchProcessor initialized for task: {self.task_name}")
    
    def preprocess(self, batch_data: BatchData) -> BatchData:
        """
        预处理批数据
        - 归一化
        - 缺失值处理
        - 特征工程
        """
        try:
            seq_x = batch_data.seq_x.to(self.device)
            
            # 归一化
            if self.use_norm:
                seq_x, scaler_params = self._normalize(seq_x)
                batch_data.metadata = batch_data.metadata or {}
                batch_data.metadata['scaler_params'] = scaler_params
            
            # 处理缺失值
            seq_x = self._handle_missing_values(seq_x)
            
            batch_data.seq_x = seq_x
            
            if batch_data.seq_y is not None:
                seq_y = batch_data.seq_y.to(self.device)
                if self.use_norm and scaler_params:
                    seq_y = self._apply_normalization(seq_y, scaler_params)
                batch_data.seq_y = seq_y
            
            return batch_data
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {e}")
            raise
    
    def postprocess(self, predictions: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
        """
        后处理预测结果
        - 逆归一化
        - 裁剪异常值
        """
        try:
            # 逆归一化
            if self.use_norm and 'scaler_params' in metadata:
                predictions = self._inverse_normalize(predictions, metadata['scaler_params'])
            
            # 裁剪异常值（可选）
            if self.config.get('clip_predictions', False):
                predictions = torch.clamp(predictions, 
                                        min=self.config.get('clip_min', -1e6),
                                        max=self.config.get('clip_max', 1e6))
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in postprocessing: {e}")
            raise
    
    def _normalize(self, data: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Z-score归一化"""
        # 计算均值和标准差 (在时间维度上)
        mean = data.mean(dim=1, keepdim=True)
        std = data.std(dim=1, keepdim=True)
        std = torch.where(std == 0, torch.ones_like(std), std)  # 避免除零
        
        normalized = (data - mean) / std
        
        scaler_params = {
            'mean': mean,
            'std': std
        }
        
        return normalized, scaler_params
    
    def _apply_normalization(self, data: torch.Tensor, scaler_params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """应用已有的归一化参数"""
        mean = scaler_params['mean']
        std = scaler_params['std']
        return (data - mean) / std
    
    def _inverse_normalize(self, data: torch.Tensor, scaler_params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """逆归一化"""
        mean = scaler_params['mean']
        std = scaler_params['std']
        return data * std + mean
    
    def _handle_missing_values(self, data: torch.Tensor) -> torch.Tensor:
        """处理缺失值（NaN）"""
        if torch.isnan(data).any():
            logger.warning("Detected NaN values in data, filling with zeros")
            data = torch.nan_to_num(data, nan=0.0)
        return data
    
    def extract_features(self, batch_data: BatchData) -> Dict[str, torch.Tensor]:
        """
        提取时序特征用于多智能体分析
        - 统计特征
        - 频域特征
        - 趋势特征
        """
        seq_x = batch_data.seq_x  # [batch, seq_len, features]
        
        features = {}
        
        # 统计特征
        features['mean'] = seq_x.mean(dim=1)  # [batch, features]
        features['std'] = seq_x.std(dim=1)
        features['min'] = seq_x.min(dim=1)[0]
        features['max'] = seq_x.max(dim=1)[0]
        
        # 趋势特征（简单线性趋势）
        time_steps = torch.arange(seq_x.size(1), device=self.device).float()
        time_steps = time_steps.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1]
        
        # 计算斜率（简单线性回归）
        x_mean = time_steps.mean(dim=1, keepdim=True)
        y_mean = seq_x.mean(dim=1, keepdim=True)
        
        numerator = ((time_steps - x_mean) * (seq_x - y_mean)).sum(dim=1)
        denominator = ((time_steps - x_mean) ** 2).sum(dim=1)
        slope = numerator / (denominator + 1e-8)
        features['trend_slope'] = slope
        
        # 变化率
        diff = torch.diff(seq_x, dim=1)
        features['change_rate'] = diff.abs().mean(dim=1)
        
        return features
    
    def create_visual_representation(self, batch_data: BatchData, num_samples: int = 4) -> List[np.ndarray]:
        """
        为视觉锚定agent创建可视化表示
        返回: List of numpy arrays representing time series images
        """
        seq_x = batch_data.seq_x.cpu().numpy()  # [batch, seq_len, features]
        
        images = []
        batch_size = min(num_samples, seq_x.shape[0])
        
        for i in range(batch_size):
            # 每个样本的时序数据
            sample = seq_x[i]  # [seq_len, features]
            
            # 简单地将时序数据作为图像矩阵
            # 可以在visual_anchor.py中进一步处理和可视化
            images.append(sample)
        
        return images
    
    def split_for_concurrent_inference(self, batch_data: BatchData, 
                                       num_splits: int = 2) -> List[BatchData]:
        """
        将batch分割为多个子batch以支持并发推理
        """
        batch_size = batch_data.seq_x.size(0)
        split_size = batch_size // num_splits
        
        sub_batches = []
        for i in range(num_splits):
            start_idx = i * split_size
            end_idx = (i + 1) * split_size if i < num_splits - 1 else batch_size
            
            sub_batch = BatchData(
                seq_x=batch_data.seq_x[start_idx:end_idx],
                seq_y=batch_data.seq_y[start_idx:end_idx] if batch_data.seq_y is not None else None,
                seq_x_mark=batch_data.seq_x_mark[start_idx:end_idx] if batch_data.seq_x_mark is not None else None,
                seq_y_mark=batch_data.seq_y_mark[start_idx:end_idx] if batch_data.seq_y_mark is not None else None,
                metadata=batch_data.metadata
            )
            sub_batches.append(sub_batch)
        
        return sub_batches
    
    def merge_concurrent_results(self, results: List[torch.Tensor]) -> torch.Tensor:
        """
        合并并发推理的结果
        """
        return torch.cat(results, dim=0)

