"""
DataAnalyzerAgent: 数据分析和处理agent
负责时序数据的预处理、特征提取和异常值检测
"""

import torch
import numpy as np
from typing import Dict, Any, Tuple, Optional
from .base_agent_ts import BaseAgentTS, AgentOutput
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from pathlib import Path

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
        
        # 从config.json读取配置
        self._load_config_from_file()
        
        # 基础配置（可被传入的config覆盖）
        self.use_diff = config.get('use_differencing', self.use_diff)
        self.smooth_window = config.get('smooth_window', self.smooth_window)
        self.anomaly_threshold = config.get('anomaly_threshold', self.anomaly_threshold)
        self.save_visualizations = config.get('save_visualizations', True)
        self.vis_save_dir = config.get('vis_save_dir', './visualizations/data_analysis/')
        
        # 创建可视化目录
        if self.save_visualizations:
            os.makedirs(self.vis_save_dir, exist_ok=True)
    
    def _load_config_from_file(self):
        """从config.json加载data_analyzer专属配置"""
        try:
            import json
            from pathlib import Path
            
            config_path = Path(__file__).parent.parent / 'config.json'
            if config_path.exists():
                with open(config_path, 'r') as f:
                    json_config = json.load(f)
                    agent_config = json_config.get('agents_config', {}).get('data_analyzer', {})
                    
                    # 基础配置
                    self.use_diff = agent_config.get('use_differencing', False)
                    self.smooth_window = agent_config.get('smooth_window', 5)
                    self.anomaly_threshold = agent_config.get('anomaly_threshold', 3.0)
                    self.seasonal_period = agent_config.get('seasonal_period', 12)
                    self.top_k_features = agent_config.get('top_k_features', 10)
                    self.feature_selection_method = agent_config.get('feature_selection_method', 'covariance')
                    
                    # data_processing配置
                    data_proc = agent_config.get('data_processing', {})
                    self.use_norm = data_proc.get('use_norm', True)
                    self.clip_predictions = data_proc.get('clip_predictions', False)
                    self.clip_min = data_proc.get('clip_min', -1e6)
                    self.clip_max = data_proc.get('clip_max', 1e6)
                    self.handle_missing = data_proc.get('handle_missing', True)
                    
                    self.log_info(f"Loaded config from file: top_k={self.top_k_features}, method={self.feature_selection_method}")
            else:
                # 默认值
                self.use_diff = False
                self.smooth_window = 5
                self.anomaly_threshold = 3.0
                self.seasonal_period = 12
                self.top_k_features = 10
                self.feature_selection_method = 'covariance'
                self.use_norm = True
                self.clip_predictions = False
                self.clip_min = -1e6
                self.clip_max = 1e6
                self.handle_missing = True
                
        except Exception as e:
            self.log_error(f"Failed to load config: {e}, using defaults")
            # 使用默认值
            self.use_diff = False
            self.smooth_window = 5
            self.anomaly_threshold = 3.0
            self.seasonal_period = 12
            self.top_k_features = 10
            self.feature_selection_method = 'covariance'
            self.use_norm = True
            self.clip_predictions = False
            self.clip_min = -1e6
            self.clip_max = 1e6
            self.handle_missing = True
        
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
            if task in ['preprocess', 'full_analysis', 'full_analysis_with_plot']:
                processed_data, preprocess_info = self._preprocess_data(data)
                result['processed_data'] = processed_data
                result['preprocess_info'] = preprocess_info
            else:
                processed_data = data
                result['processed_data'] = data
            
            if task in ['feature_extract', 'full_analysis', 'full_analysis_with_plot']:
                features = self._extract_features(processed_data)
                result['data_features'] = features
            
            if task in ['anomaly_detect', 'full_analysis', 'full_analysis_with_plot']:
                anomalies = self._detect_anomalies(processed_data)
                result['anomalies'] = anomalies
            
            if task in ['decompose', 'full_analysis', 'full_analysis_with_plot']:
                trend, seasonal, residual = self._decompose_series(processed_data)
                result['trend'] = trend
                result['seasonal'] = seasonal
                result['residual'] = residual
            
            # 协变量分析和特征选择
            if task in ['full_analysis', 'full_analysis_with_plot']:
                covariance_matrix, feature_importances, selected_features = self._covariate_analysis(processed_data)
                result['covariance_matrix'] = covariance_matrix
                result['feature_importances'] = feature_importances
                result['selected_features'] = selected_features
                result['top_k_indices'] = selected_features
                self.log_info(f"Feature selection: {len(selected_features)} features selected from {data.shape[2]}")
            
            # 生成可视化和统计文本
            if task == 'full_analysis_with_plot':
                batch_idx = input_data.get('batch_idx', 0)
                # 使用选择的特征生成plot
                selected_data = self._select_top_features(processed_data, result.get('selected_features', None))
                plot_path = self._generate_plot(selected_data, batch_idx, remove_text=True)
                statistics_text = self._generate_statistics_text(result.get('data_features', {}))
                result['plot_path'] = plot_path
                result['statistics_text'] = statistics_text
                result['selected_data'] = selected_data  # 用于后续agent
                self.log_info(f"Generated plot: {plot_path}")
            
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
        
        batch, seq_len, features = data.shape
        smoothed = torch.zeros_like(data)
        kernel = torch.ones(window, device=data.device) / window
        
        for f in range(features):
            for b in range(batch):
                signal = data[b, :, f]
                padded = torch.nn.functional.pad(signal.unsqueeze(0).unsqueeze(0), 
                                                (window//2, window//2), mode='replicate')
                conv_out = torch.nn.functional.conv1d(
                    padded, kernel.unsqueeze(0).unsqueeze(0)
                ).squeeze()
                
                # 确保输出长度正确
                if conv_out.size(0) != seq_len:
                    conv_out = conv_out[:seq_len]
                
                smoothed[b, :, f] = conv_out
        
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
    
    def _covariate_analysis(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, list[int]]:
        """
        协变量分析：分析特征间的协方差，选择top_k重要特征
        
        Args:
            data: [batch, seq_len, features]
            
        Returns:
            covariance_matrix: [features, features]
            feature_importances: [features]
            selected_features: List of selected feature indices
        """
        batch, seq_len, n_features = data.shape
        
        # 计算协方差矩阵
        # 先将数据reshape为 [batch*seq_len, features]
        data_reshaped = data.reshape(-1, n_features)  # [batch*seq_len, features]
        
        # 中心化
        mean = data_reshaped.mean(dim=0, keepdim=True)
        centered = data_reshaped - mean
        
        # 协方差矩阵 = (X^T X) / (n-1)
        cov_matrix = torch.matmul(centered.t(), centered) / (data_reshaped.size(0) - 1)
        
        # 计算特征重要性
        if self.feature_selection_method == 'covariance':
            # 方法1：基于协方差和的绝对值（特征与其他特征的总相关性）
            feature_importances = torch.abs(cov_matrix).sum(dim=1)
        elif self.feature_selection_method == 'variance':
            # 方法2：基于方差（对角线元素）
            feature_importances = torch.diag(cov_matrix)
        else:
            # 默认：综合方法（方差 + 协方差和）
            variance = torch.diag(cov_matrix)
            covariance_sum = torch.abs(cov_matrix).sum(dim=1) - variance  # 减去自己
            feature_importances = variance + 0.5 * covariance_sum
        
        # 选择top k个特征
        k = min(self.top_k_features, n_features)
        _, top_indices = torch.topk(feature_importances, k)
        selected_features = top_indices.tolist()
        
        self.log_info(f"Covariance analysis: selected {k} features with importance scores: {feature_importances[top_indices].tolist()}")
        
        return cov_matrix, feature_importances, selected_features
    
    def _select_top_features(self, data: torch.Tensor, selected_features: Optional[list[int]]) -> torch.Tensor:
        """
        选择top k个特征
        
        Args:
            data: [batch, seq_len, features]
            selected_features: 选择的特征索引列表
            
        Returns:
            selected_data: [batch, seq_len, k]
        """
        if selected_features is None or len(selected_features) == 0:
            return data
        
        # 选择指定的特征
        selected_data = data[:, :, selected_features]
        
        return selected_data
    
    def _generate_plot(self, data: torch.Tensor, batch_idx: int = 0, remove_text: bool = True) -> str:
        """
        生成时序plot图并保存 - 使用VisualAnchor的visualization配置
        
        Args:
            data: [batch, seq_len, features]
            batch_idx: 批次索引
            remove_text: 是否移除文本（从config.json读取，但可以被参数覆盖）
            
        Returns:
            plot_path: 保存的图片路径
        """
        # 读取visualization配置（从VisualAnchor）
        vis_config = self._load_visualization_config()
        
        save_dir = Path(self.vis_save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f'batch_{batch_idx}_analysis.png'
        
        data_np = data[0].cpu().numpy()  # 取第一个样本
        seq_len, n_features = data_np.shape
        
        # 绘制前3个特征（如果超过3个）
        num_plots = min(3, n_features)
        fig, axes = plt.subplots(num_plots, 1, figsize=vis_config['fig_size'])
        if num_plots == 1:
            axes = [axes]
        
        time_steps = np.arange(seq_len)
        
        for i in range(num_plots):
            ax = axes[i]
            ax.plot(time_steps, data_np[:, i], 'b-', linewidth=vis_config['line_width'])
            ax.fill_between(time_steps, data_np[:, i], alpha=vis_config['fill_alpha'])
            
            # 网格
            if vis_config['show_grid']:
                ax.grid(True, alpha=0.3, linestyle='--')
            
            # 根据配置决定是否添加文本标签
            if not vis_config['remove_plot_text']:
                ax.set_xlabel('Time Steps', fontsize=10)
                ax.set_ylabel('Value', fontsize=10)
                ax.set_title(f'Feature {i} Time Series', fontsize=12, fontweight='bold')
                if vis_config['show_legend']:
                    ax.legend([f'Feature {i}'], loc='upper right')
            else:
                # 移除所有文本，只保留数据线和网格
                ax.set_xticks([])
                ax.set_yticks([])
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=vis_config['dpi'], bbox_inches='tight')
        plt.close(fig)
        
        self.log_info(f"Plot saved to: {save_path}")
        return str(save_path)
    
    def _load_visualization_config(self) -> Dict[str, Any]:
        """从config.json的visual_anchor配置中加载visualization参数"""
        try:
            import json
            from pathlib import Path
            
            config_path = Path(__file__).parent.parent / 'config.json'
            if config_path.exists():
                with open(config_path, 'r') as f:
                    json_config = json.load(f)
                    agent_config = json_config.get('agents_config', {}).get('visual_anchor', {})
                    vis_config = agent_config.get('visualization', {})
                    
                    return {
                        'fig_size': tuple(vis_config.get('fig_size', [12, 6])),
                        'dpi': vis_config.get('dpi', 150),
                        'plot_style': vis_config.get('plot_style', 'default'),
                        'remove_plot_text': vis_config.get('remove_plot_text', True),
                        'show_grid': vis_config.get('show_grid', True),
                        'show_legend': vis_config.get('show_legend', False),
                        'line_width': vis_config.get('line_width', 2),
                        'fill_alpha': vis_config.get('fill_alpha', 0.3),
                        'color_scheme': vis_config.get('color_scheme', 'default')
                    }
        except Exception as e:
            self.log_error(f"Failed to load visualization config: {e}, using defaults")
        
        # 默认值
        return {
            'fig_size': (12, 6),
            'dpi': 150,
            'plot_style': 'default',
            'remove_plot_text': True,
            'show_grid': True,
            'show_legend': False,
            'line_width': 2,
            'fill_alpha': 0.3,
            'color_scheme': 'default'
        }
    
    def _generate_statistics_text(self, features: Dict[str, torch.Tensor]) -> str:
        """
        生成统计描述文本
        
        Args:
            features: 特征字典
            
        Returns:
            statistics_text: 统计描述
        """
        text = "=== Time Series Statistics ===\n\n"
        
        if 'mean' in features:
            text += f"Mean: {features['mean'].mean().item():.4f}\n"
        if 'std' in features:
            text += f"Standard Deviation: {features['std'].mean().item():.4f}\n"
        if 'min' in features:
            text += f"Min: {features['min'].mean().item():.4f}\n"
        if 'max' in features:
            text += f"Max: {features['max'].mean().item():.4f}\n"
        if 'median' in features:
            text += f"Median: {features['median'].mean().item():.4f}\n"
        
        text += "\n=== Trend Analysis ===\n\n"
        if 'trend' in features:
            trend_val = features['trend'].mean().item()
            text += f"Trend Slope: {trend_val:.6f}\n"
            if trend_val > 0.001:
                text += "Trend Direction: Increasing\n"
            elif trend_val < -0.001:
                text += "Trend Direction: Decreasing\n"
            else:
                text += "Trend Direction: Stable\n"
        
        text += "\n=== Variability ===\n\n"
        if 'abs_change' in features:
            text += f"Mean Absolute Change: {features['abs_change'].mean().item():.4f}\n"
        if 'autocorr_1' in features:
            text += f"Autocorrelation (lag-1): {features['autocorr_1'].mean().item():.4f}\n"
        
        text += "\n=== Shape Characteristics ===\n\n"
        if 'skewness' in features:
            text += f"Skewness: {features['skewness'].mean().item():.4f}\n"
        if 'kurtosis' in features:
            text += f"Kurtosis: {features['kurtosis'].mean().item():.4f}\n"
        
        return text

