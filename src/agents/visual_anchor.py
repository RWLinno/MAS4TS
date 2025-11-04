"""
VisualAnchorAgent: 视觉锚定agent
将时序数据可视化为图像，使用VLM生成锚点和语义先验
这是MAS4TS的核心创新之一
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无GUI后端
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple, Optional
from PIL import Image
import io
from .base_agent_ts import BaseAgentTS, AgentOutput
import logging

logger = logging.getLogger(__name__)


class VisualAnchorAgent(BaseAgentTS):
    """
    视觉锚定Agent：将时序转为图像，生成预测锚点
    
    核心创新:
    1. 将历史时序数据转换为图像表示
    2. 使用VLM识别模式、趋势和关键点
    3. 生成未来预测的"锚点"（可能的值范围和关键时间点）
    4. 提供语义先验（如"上升趋势"、"周期性模式"等）
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("VisualAnchorAgent", config)
        
        self.vlm_model = config.get('vlm_model', None)  # VLM模型（如GPT-4V, Qwen-VL等）
        self.use_vlm = config.get('use_vlm', False)
        self.anchor_strategy = config.get('anchor_strategy', 'confidence_interval')  # 'confidence_interval', 'pattern_based'
        self.confidence_level = config.get('confidence_level', 0.95)
        
        # 可视化参数
        self.fig_size = config.get('fig_size', (10, 6))
        self.dpi = config.get('dpi', 100)
        self.style = config.get('plot_style', 'default')
        
    async def process(self, input_data: Dict[str, Any]) -> AgentOutput:
        """
        处理时序数据，生成视觉锚点
        
        Args:
            input_data: {
                'data': torch.Tensor,  # [batch, seq_len, features]
                'task': str,  # 'forecasting', 'classification', etc.
                'pred_len': int,  # 预测长度（用于预测任务）
                'config': Dict
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
            
            data = input_data['data']  # [batch, seq_len, features]
            task = input_data.get('task', 'forecasting')
            pred_len = input_data.get('pred_len', 96)
            
            self.log_info(f"Creating visual anchors for {task} task")
            
            # 1. 生成时序图像
            images = self._create_time_series_images(data)
            
            # 2. 生成锚点
            if task == 'forecasting':
                anchors = self._generate_forecast_anchors(data, pred_len, images)
            elif task == 'classification':
                anchors = self._generate_classification_anchors(data, images)
            elif task == 'anomaly_detection':
                anchors = self._generate_anomaly_anchors(data, images)
            else:
                anchors = self._generate_default_anchors(data, images)
            
            # 3. 提取语义先验（如果使用VLM）
            if self.use_vlm and self.vlm_model:
                semantic_priors = await self._extract_semantic_priors(images, task)
            else:
                semantic_priors = self._extract_rule_based_priors(data, task)
            
            result = {
                'anchors': anchors,
                'semantic_priors': semantic_priors,
                'images': images,
                'metadata': {
                    'task': task,
                    'num_anchors': len(anchors) if isinstance(anchors, list) else anchors.get('num_points', 0)
                }
            }
            
            return AgentOutput(
                agent_name=self.name,
                success=True,
                result=result,
                confidence=0.90,
                metadata={'task': task}
            )
            
        except Exception as e:
            self.log_error(f"Error in visual anchoring: {e}")
            import traceback
            traceback.print_exc()
            return AgentOutput(
                agent_name=self.name,
                success=False,
                result=None,
                confidence=0.0,
                metadata={'error': str(e)}
            )
    
    def _create_time_series_images(self, data: torch.Tensor) -> List[Image.Image]:
        """
        将时序数据转换为图像
        
        Args:
            data: [batch, seq_len, features]
            
        Returns:
            List of PIL Images
        """
        images = []
        data_np = data.cpu().numpy()
        
        batch_size = min(data_np.shape[0], 4)  # 最多处理4个样本
        
        for i in range(batch_size):
            fig, axes = plt.subplots(min(data_np.shape[2], 3), 1, 
                                    figsize=self.fig_size, squeeze=False)
            
            # 绘制前3个特征
            for j in range(min(data_np.shape[2], 3)):
                ax = axes[j, 0]
                series = data_np[i, :, j]
                time_idx = np.arange(len(series))
                
                # 绘制时序线
                ax.plot(time_idx, series, 'b-', linewidth=2, label=f'Feature {j}')
                ax.fill_between(time_idx, series, alpha=0.3)
                
                # 添加网格和标签
                ax.grid(True, alpha=0.3)
                ax.set_xlabel('Time Steps')
                ax.set_ylabel('Value')
                ax.set_title(f'Sample {i} - Feature {j}')
                ax.legend()
            
            plt.tight_layout()
            
            # 转换为PIL Image
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=self.dpi, bbox_inches='tight')
            buf.seek(0)
            img = Image.open(buf)
            images.append(img)
            
            plt.close(fig)
        
        self.log_info(f"Created {len(images)} time series images")
        return images
    
    def _generate_forecast_anchors(self, data: torch.Tensor, pred_len: int, 
                                   images: List[Image.Image]) -> Dict[str, Any]:
        """
        生成预测任务的锚点
        
        锚点包括:
        1. 置信区间 (上界和下界)
        2. 关键时间点 (可能的转折点)
        3. 趋势方向
        """
        batch, seq_len, features = data.shape
        
        # 1. 基于历史数据估计置信区间
        # 计算历史波动性
        historical_mean = data.mean(dim=1, keepdim=True)  # [batch, 1, features]
        historical_std = data.std(dim=1, keepdim=True)    # [batch, 1, features]
        
        # 2. 提取趋势
        trend = self._estimate_trend(data)  # [batch, features]
        
        # 3. 生成未来的锚点
        # 使用线性外推 + 置信区间
        future_time = torch.arange(1, pred_len + 1, device=data.device).float()
        future_time = future_time.unsqueeze(0).unsqueeze(2)  # [1, pred_len, 1]
        
        # 最后一个观测值
        last_value = data[:, -1:, :]  # [batch, 1, features]
        
        # 趋势外推
        trend_extrapolation = last_value + trend.unsqueeze(1) * future_time
        
        # 置信区间（基于历史波动）
        z_score = 1.96 if self.confidence_level == 0.95 else 2.576  # 95% or 99%
        
        upper_bound = trend_extrapolation + z_score * historical_std
        lower_bound = trend_extrapolation - z_score * historical_std
        
        # 4. 识别关键锚点（均匀采样）
        num_anchors = min(5, pred_len)  # 最多5个锚点
        anchor_indices = np.linspace(0, pred_len-1, num_anchors, dtype=int)
        
        anchor_points = {
            'indices': torch.tensor(anchor_indices, device=data.device),
            'values': trend_extrapolation[:, anchor_indices, :],  # [batch, num_anchors, features]
            'upper_bounds': upper_bound[:, anchor_indices, :],
            'lower_bounds': lower_bound[:, anchor_indices, :],
        }
        
        anchors = {
            'type': 'confidence_interval',
            'full_trajectory': {
                'mean': trend_extrapolation,  # [batch, pred_len, features]
                'upper': upper_bound,
                'lower': lower_bound,
            },
            'key_points': anchor_points,
            'trend_direction': self._classify_trend(trend),  # 'increasing', 'decreasing', 'stable'
            'num_points': num_anchors
        }
        
        return anchors
    
    def _generate_classification_anchors(self, data: torch.Tensor, 
                                        images: List[Image.Image]) -> Dict[str, Any]:
        """
        生成分类任务的视觉锚点
        识别关键模式特征
        """
        # 提取关键模式特征
        pattern_features = self._extract_pattern_features(data)
        
        anchors = {
            'type': 'pattern_based',
            'features': pattern_features,
            'visual_summary': {
                'has_peaks': self._detect_peaks(data),
                'has_trend': self._has_significant_trend(data),
                'periodicity': self._estimate_periodicity(data),
                'complexity': self._estimate_complexity(data)
            }
        }
        
        return anchors
    
    def _generate_anomaly_anchors(self, data: torch.Tensor, 
                                  images: List[Image.Image]) -> Dict[str, Any]:
        """
        生成异常检测任务的视觉锚点
        标记可疑区域
        """
        # 检测可疑区域
        suspicious_regions = self._identify_suspicious_regions(data)
        
        anchors = {
            'type': 'anomaly_regions',
            'suspicious_regions': suspicious_regions,
            'baseline_statistics': {
                'mean': data.mean(dim=1),
                'std': data.std(dim=1),
                'median': data.median(dim=1)[0]
            }
        }
        
        return anchors
    
    def _generate_default_anchors(self, data: torch.Tensor, 
                                  images: List[Image.Image]) -> Dict[str, Any]:
        """生成默认锚点"""
        return {
            'type': 'default',
            'summary_statistics': {
                'mean': data.mean(dim=1),
                'std': data.std(dim=1),
                'range': (data.min(dim=1)[0], data.max(dim=1)[0])
            }
        }
    
    def _estimate_trend(self, data: torch.Tensor) -> torch.Tensor:
        """估计线性趋势斜率"""
        batch, seq_len, features = data.shape
        
        t = torch.arange(seq_len, device=data.device, dtype=data.dtype)
        t = t.unsqueeze(0).unsqueeze(2).expand(batch, seq_len, features)
        
        t_mean = t.mean(dim=1, keepdim=True)
        y_mean = data.mean(dim=1, keepdim=True)
        
        numerator = ((t - t_mean) * (data - y_mean)).sum(dim=1)
        denominator = ((t - t_mean) ** 2).sum(dim=1) + 1e-8
        
        slope = numerator / denominator
        
        return slope  # [batch, features]
    
    def _classify_trend(self, trend: torch.Tensor) -> List[str]:
        """分类趋势方向"""
        trend_np = trend.cpu().numpy()
        
        directions = []
        for i in range(trend_np.shape[0]):
            avg_trend = trend_np[i].mean()
            if avg_trend > 0.01:
                directions.append('increasing')
            elif avg_trend < -0.01:
                directions.append('decreasing')
            else:
                directions.append('stable')
        
        return directions
    
    def _extract_pattern_features(self, data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """提取模式特征"""
        features = {}
        
        # 峰值和谷值
        features['num_peaks'] = self._count_peaks(data)
        features['num_valleys'] = self._count_valleys(data)
        
        # 交叉零点次数
        features['zero_crossings'] = self._count_zero_crossings(data)
        
        # 能量
        features['energy'] = (data ** 2).sum(dim=1)
        
        return features
    
    def _detect_peaks(self, data: torch.Tensor) -> torch.Tensor:
        """检测是否有显著峰值"""
        # 简单实现：检测局部最大值
        left_shift = data[:, :-2, :]
        center = data[:, 1:-1, :]
        right_shift = data[:, 2:, :]
        
        peaks = (center > left_shift) & (center > right_shift)
        has_peaks = peaks.any(dim=1)
        
        return has_peaks
    
    def _has_significant_trend(self, data: torch.Tensor) -> torch.Tensor:
        """检测是否有显著趋势"""
        trend = self._estimate_trend(data)
        
        # 趋势显著性：|slope| > threshold
        threshold = 0.01
        significant = torch.abs(trend).mean(dim=1) > threshold
        
        return significant
    
    def _estimate_periodicity(self, data: torch.Tensor) -> torch.Tensor:
        """估计周期性强度（简化版）"""
        # 使用自相关估计
        max_lag = min(data.size(1) // 2, 50)
        
        autocorr_scores = []
        for lag in range(1, max_lag):
            autocorr = self._compute_autocorr(data, lag)
            autocorr_scores.append(autocorr.abs().mean(dim=1))
        
        if autocorr_scores:
            periodicity = torch.stack(autocorr_scores).max(dim=0)[0]
        else:
            periodicity = torch.zeros(data.size(0), device=data.device)
        
        return periodicity
    
    def _compute_autocorr(self, data: torch.Tensor, lag: int) -> torch.Tensor:
        """计算自相关"""
        if lag >= data.size(1):
            return torch.zeros(data.size(0), data.size(2), device=data.device)
        
        y1 = data[:, :-lag, :]
        y2 = data[:, lag:, :]
        
        y1_centered = y1 - y1.mean(dim=1, keepdim=True)
        y2_centered = y2 - y2.mean(dim=1, keepdim=True)
        
        numerator = (y1_centered * y2_centered).sum(dim=1)
        denominator = torch.sqrt((y1_centered ** 2).sum(dim=1) * 
                                (y2_centered ** 2).sum(dim=1)) + 1e-8
        
        return numerator / denominator
    
    def _estimate_complexity(self, data: torch.Tensor) -> torch.Tensor:
        """估计序列复杂度"""
        # 使用变化率的标准差作为复杂度度量
        diff = torch.diff(data, dim=1)
        complexity = diff.std(dim=1).mean(dim=1)
        
        return complexity
    
    def _identify_suspicious_regions(self, data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """识别可疑区域（可能的异常）"""
        # 基于z-score
        mean = data.mean(dim=1, keepdim=True)
        std = data.std(dim=1, keepdim=True) + 1e-8
        z_scores = torch.abs((data - mean) / std)
        
        # 可疑：|z-score| > 2
        suspicious_mask = z_scores > 2.0
        
        return {
            'mask': suspicious_mask,
            'scores': z_scores,
            'count': suspicious_mask.sum(dim=1)
        }
    
    def _count_peaks(self, data: torch.Tensor) -> torch.Tensor:
        """计算峰值数量"""
        if data.size(1) < 3:
            return torch.zeros(data.size(0), data.size(2), device=data.device)
        
        left = data[:, :-2, :]
        center = data[:, 1:-1, :]
        right = data[:, 2:, :]
        
        peaks = (center > left) & (center > right)
        return peaks.sum(dim=1).float()
    
    def _count_valleys(self, data: torch.Tensor) -> torch.Tensor:
        """计算谷值数量"""
        if data.size(1) < 3:
            return torch.zeros(data.size(0), data.size(2), device=data.device)
        
        left = data[:, :-2, :]
        center = data[:, 1:-1, :]
        right = data[:, 2:, :]
        
        valleys = (center < left) & (center < right)
        return valleys.sum(dim=1).float()
    
    def _count_zero_crossings(self, data: torch.Tensor) -> torch.Tensor:
        """计算过零次数"""
        # 中心化
        centered = data - data.mean(dim=1, keepdim=True)
        
        # 检测符号变化
        signs = torch.sign(centered)
        sign_changes = (signs[:, :-1, :] != signs[:, 1:, :]) & (signs[:, :-1, :] != 0) & (signs[:, 1:, :] != 0)
        
        return sign_changes.sum(dim=1).float()
    
    async def _extract_semantic_priors(self, images: List[Image.Image], 
                                      task: str) -> Dict[str, Any]:
        """
        使用VLM提取语义先验
        （需要实现VLM调用，这里提供接口）
        """
        # TODO: 实现VLM调用
        # 例如调用GPT-4V或Qwen-VL
        self.log_info("VLM-based semantic extraction not implemented, using rule-based")
        return {'method': 'vlm', 'priors': []}
    
    def _extract_rule_based_priors(self, data: torch.Tensor, task: str) -> Dict[str, str]:
        """
        基于规则的语义先验提取
        """
        priors = {}
        
        # 趋势描述
        trend = self._estimate_trend(data)
        avg_trend = trend.mean().item()
        
        if avg_trend > 0.01:
            priors['trend'] = "The time series shows an increasing trend"
        elif avg_trend < -0.01:
            priors['trend'] = "The time series shows a decreasing trend"
        else:
            priors['trend'] = "The time series is relatively stable"
        
        # 波动性描述
        volatility = data.std().item()
        if volatility > data.abs().mean().item():
            priors['volatility'] = "High volatility detected"
        else:
            priors['volatility'] = "Low to moderate volatility"
        
        # 周期性描述
        periodicity = self._estimate_periodicity(data).mean().item()
        if periodicity > 0.5:
            priors['periodicity'] = "Strong periodic pattern detected"
        elif periodicity > 0.3:
            priors['periodicity'] = "Moderate periodic pattern"
        else:
            priors['periodicity'] = "No clear periodic pattern"
        
        return priors

