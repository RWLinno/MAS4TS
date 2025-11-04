"""
DataAnalyzerAgent: 数据分析和处理agent
负责时序数据的预处理、特征提取和异常值检测
"""

import torch
import numpy as np
from typing import Dict, Any, Tuple, Optional
from .base_agent_ts import BaseAgentTS, AgentOutput
import logging

logger = logging.getLogger(__name__)


class DataAnalyzerAgent(BaseAgentTS):
    """
    数据分析Agent：处理时序数据的各种分析任务
    
    职责:
    1. 缺失值检测和处理
    2. 异常值检测
    3. 趋势和季节性分解
    4. 统计特征提取
    5. 数据平滑和去噪
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("DataAnalyzerAgent", config)
        
        self.use_diff = config.get('use_differencing', False)
        self.smooth_window = config.get('smooth_window', 5)
        self.anomaly_threshold = config.get('anomaly_threshold', 3.0)  # z-score阈值
        
    async def process(self, input_data: Dict[str, Any]) -> AgentOutput:
        """
        处理数据分析请求
        
        Args:
            input_data: {
                'data': torch.Tensor,  # [batch, seq_len, features]
                'config': Dict,  # 配置
                'task': str  # 'preprocess', 'feature_extract', 'anomaly_detect'
            }
        """
        try:
            if not self.validate_input(input_data, ['data']):
                return AgentOutput(
                    agent_name=self.name,
                    success=False,
                    result=None,
                    confidence=0.0
                )
            
            data = input_data['data']
            task = input_data.get('task', 'full_analysis')
            
            self.log_info(f"Analyzing data with shape: {data.shape}")
            
            result = {}
            
            # 执行不同的分析任务
            if task in ['preprocess', 'full_analysis']:
                # 数据预处理
                processed_data, preprocess_info = self._preprocess_data(data)
                result['processed_data'] = processed_data
                result['preprocess_info'] = preprocess_info
            else:
                processed_data = data
                result['processed_data'] = data
            
            if task in ['feature_extract', 'full_analysis']:
                # 特征提取
                features = self._extract_features(processed_data)
                result['data_features'] = features
            
            if task in ['anomaly_detect', 'full_analysis']:
                # 异常检测
                anomalies = self._detect_anomalies(processed_data)
                result['anomalies'] = anomalies
            
            if task in ['decompose', 'full_analysis']:
                # 趋势季节性分解
                trend, seasonal, residual = self._decompose_series(processed_data)
                result['trend'] = trend
                result['seasonal'] = seasonal
                result['residual'] = residual
            
            return AgentOutput(
                agent_name=self.name,
                success=True,
                result=result,
                confidence=0.95,
                metadata={'task': task, 'data_shape': list(data.shape)}
            )
            
        except Exception as e:
            self.log_error(f"Error in data analysis: {e}")
            import traceback
            traceback.print_exc()
            return AgentOutput(
                agent_name=self.name,
                success=False,
                result=None,
                confidence=0.0,
                metadata={'error': str(e)}
            )
    
    def _preprocess_data(self, data: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        预处理数据
        - 缺失值处理
        - 异常值平滑
        - 数据平滑
        """
        preprocess_info = {}
        
        # 检测缺失值
        if torch.isnan(data).any():
            self.log_info("Detected NaN values, filling with interpolation")
            data = self._fill_missing_values(data)
            preprocess_info['had_missing'] = True
        else:
            preprocess_info['had_missing'] = False
        
        # 平滑处理
        if self.smooth_window > 1:
            data = self._smooth_data(data, window=self.smooth_window)
            preprocess_info['smoothed'] = True
        
        # 差分（可选）
        if self.use_diff:
            data, diff_info = self._apply_differencing(data)
            preprocess_info['differencing'] = diff_info
        
        return data, preprocess_info
    
    def _fill_missing_values(self, data: torch.Tensor) -> torch.Tensor:
        """填充缺失值 - 使用线性插值"""
        # 简单实现：用前后值的均值填充
        data = data.clone()
        mask = torch.isnan(data)
        
        # 用0填充（后续可以改进为更复杂的插值方法）
        data[mask] = 0.0
        
        # 如果需要更精确的插值，可以实现线性插值
        # 这里先用简单的前向填充
        for i in range(1, data.size(1)):
            nan_mask = torch.isnan(data[:, i, :])
            data[:, i, :] = torch.where(nan_mask, data[:, i-1, :], data[:, i, :])
        
        return data
    
    def _smooth_data(self, data: torch.Tensor, window: int = 5) -> torch.Tensor:
        """数据平滑 - 移动平均"""
        if window <= 1:
            return data
        
        # 使用1D卷积进行移动平均
        batch, seq_len, features = data.shape
        
        # 为每个特征分别平滑
        smoothed = torch.zeros_like(data)
        kernel = torch.ones(window, device=data.device) / window
        
        for f in range(features):
            for b in range(batch):
                signal = data[b, :, f]
                # 简单的移动平均
                padded = torch.nn.functional.pad(signal.unsqueeze(0).unsqueeze(0), 
                                                (window//2, window//2), mode='replicate')
                smoothed[b, :, f] = torch.nn.functional.conv1d(
                    padded, kernel.unsqueeze(0).unsqueeze(0)
                ).squeeze()
        
        return smoothed
    
    def _apply_differencing(self, data: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """应用差分使数据平稳"""
        diff_data = torch.diff(data, dim=1)
        
        # 保持原始维度，第一个时间步用原始值
        first_step = data[:, :1, :]
        diff_data = torch.cat([first_step, diff_data], dim=1)
        
        diff_info = {
            'applied': True,
            'order': 1,
            'first_value': first_step
        }
        
        return diff_data, diff_info
    
    def _extract_features(self, data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        提取统计特征
        """
        features = {}
        
        # 基本统计量
        features['mean'] = data.mean(dim=1)  # [batch, features]
        features['std'] = data.std(dim=1)
        features['min'] = data.min(dim=1)[0]
        features['max'] = data.max(dim=1)[0]
        features['median'] = data.median(dim=1)[0]
        
        # 四分位数
        features['q25'] = torch.quantile(data, 0.25, dim=1)
        features['q75'] = torch.quantile(data, 0.75, dim=1)
        
        # 偏度和峰度（简化版）
        centered = data - features['mean'].unsqueeze(1)
        features['skewness'] = (centered ** 3).mean(dim=1) / (features['std'] ** 3 + 1e-8)
        features['kurtosis'] = (centered ** 4).mean(dim=1) / (features['std'] ** 4 + 1e-8)
        
        # 变化率
        diff = torch.diff(data, dim=1)
        features['mean_change'] = diff.mean(dim=1)
        features['std_change'] = diff.std(dim=1)
        features['abs_change'] = diff.abs().mean(dim=1)
        
        # 趋势（线性回归斜率）
        features['trend'] = self._compute_trend(data)
        
        # 自相关（lag-1）
        features['autocorr_1'] = self._compute_autocorrelation(data, lag=1)
        
        return features
    
    def _compute_trend(self, data: torch.Tensor) -> torch.Tensor:
        """计算线性趋势斜率"""
        batch, seq_len, n_features = data.shape
        
        # 时间索引
        t = torch.arange(seq_len, device=data.device, dtype=data.dtype).unsqueeze(0).unsqueeze(2)
        t = t.expand(batch, seq_len, n_features)
        
        # 计算斜率 (简单线性回归)
        t_mean = t.mean(dim=1, keepdim=True)
        y_mean = data.mean(dim=1, keepdim=True)
        
        numerator = ((t - t_mean) * (data - y_mean)).sum(dim=1)
        denominator = ((t - t_mean) ** 2).sum(dim=1) + 1e-8
        
        slope = numerator / denominator
        
        return slope
    
    def _compute_autocorrelation(self, data: torch.Tensor, lag: int = 1) -> torch.Tensor:
        """计算自相关系数"""
        if lag >= data.size(1):
            return torch.zeros(data.size(0), data.size(2), device=data.device)
        
        y1 = data[:, :-lag, :]
        y2 = data[:, lag:, :]
        
        y1_mean = y1.mean(dim=1, keepdim=True)
        y2_mean = y2.mean(dim=1, keepdim=True)
        
        numerator = ((y1 - y1_mean) * (y2 - y2_mean)).sum(dim=1)
        denominator = torch.sqrt(((y1 - y1_mean) ** 2).sum(dim=1) * 
                                ((y2 - y2_mean) ** 2).sum(dim=1)) + 1e-8
        
        corr = numerator / denominator
        
        return corr
    
    def _detect_anomalies(self, data: torch.Tensor) -> Dict[str, Any]:
        """
        检测异常值（基于z-score）
        """
        # 计算z-score
        mean = data.mean(dim=1, keepdim=True)
        std = data.std(dim=1, keepdim=True) + 1e-8
        z_scores = (data - mean) / std
        
        # 检测异常
        anomaly_mask = torch.abs(z_scores) > self.anomaly_threshold
        
        anomalies = {
            'mask': anomaly_mask,  # [batch, seq_len, features]
            'scores': z_scores.abs(),
            'count': anomaly_mask.sum(dim=1),  # [batch, features]
            'indices': torch.where(anomaly_mask)
        }
        
        self.log_info(f"Detected {anomaly_mask.sum().item()} anomalies")
        
        return anomalies
    
    def _decompose_series(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        简单的时序分解：趋势 + 季节性 + 残差
        使用移动平均法提取趋势
        """
        # 提取趋势（使用较大窗口的移动平均）
        trend = self._smooth_data(data, window=max(self.smooth_window * 2, 10))
        
        # 去趋势
        detrended = data - trend
        
        # 简化的季节性提取（使用周期性平均）
        # 假设周期为12（可配置）
        period = self.config.get('seasonal_period', 12)
        if data.size(1) >= period * 2:
            seasonal = self._extract_seasonal(detrended, period)
        else:
            seasonal = torch.zeros_like(data)
        
        # 残差
        residual = detrended - seasonal
        
        return trend, seasonal, residual
    
    def _extract_seasonal(self, data: torch.Tensor, period: int) -> torch.Tensor:
        """提取季节性成分"""
        batch, seq_len, features = data.shape
        seasonal = torch.zeros_like(data)
        
        # 对每个周期位置计算平均值
        for i in range(period):
            indices = list(range(i, seq_len, period))
            if len(indices) > 0:
                seasonal[:, i::period, :] = data[:, indices, :].mean(dim=1, keepdim=True)
        
        return seasonal

