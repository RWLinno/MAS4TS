"""
VisualAnchorAgent: 视觉锚定agent
将时序数据可视化为图像，使用VLM生成锚点和语义先验
这是MAS4TS的核心创新之一
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple, Optional
from PIL import Image
import io
import os
import json
from pathlib import Path
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
        
        # 从config.json读取配置
        self._load_config_from_file()
        
        # 允许传入的config覆盖
        self.use_vlm = config.get('use_vlm', self.use_vlm)
        self.use_eas = config.get('use_eas', self.use_eas)
        self.vlm_model_name = config.get('vlm_model', self.vlm_model_name)
        self.vis_save_dir = config.get('vis_save_dir', self.anchor_save_dir)
        
        # Anchor本地存储配置（新增）
        self.enable_anchor_save = config.get('save_anchors', self.enable_anchor_save)
        
        # VLM配置
        self.vlm_model = None
        self.vlm_tokenizer = None
        self.eas_client = None
        
        # 初始化VLM
        if self.use_vlm:
            if self.use_eas:
                self._init_eas_client(config)
            else:
                self._init_local_vlm()
        
        # 创建可视化和anchor存储目录
        os.makedirs(self.vis_save_dir, exist_ok=True)
        if self.enable_anchor_save:
            os.makedirs(self.anchor_save_dir, exist_ok=True)
            self.log_info(f"Anchor存储已启用，保存路径: {self.anchor_save_dir}")
    
    def _load_config_from_file(self):
        """从config.json加载visual_anchor专属配置"""
        try:
            import json
            from pathlib import Path
            
            config_path = Path(__file__).parent.parent / 'config.json'
            if config_path.exists():
                with open(config_path, 'r') as f:
                    json_config = json.load(f)
                    agent_config = json_config.get('agents_config', {}).get('visual_anchor', {})
                    
                    # 基础配置
                    self.use_vlm = agent_config.get('use_vlm', False)
                    self.use_eas = agent_config.get('use_eas', False)
                    self.vlm_model_name = agent_config.get('model_name', 'Qwen/Qwen-VL-Chat')
                    self.anchor_strategy = agent_config.get('anchor_strategy', 'confidence_interval')
                    self.confidence_level = agent_config.get('confidence_level', 0.95)
                    
                    # anchor_storage配置（新增）
                    storage_config = agent_config.get('anchor_storage', {})
                    self.enable_anchor_save = storage_config.get('enable_save', True)
                    self.anchor_save_dir = storage_config.get('save_dir', './visualizations/visual_anchors/')
                    self.anchor_save_format = storage_config.get('save_format', 'json')
                    self.update_frequency = storage_config.get('update_frequency', 'every_batch')
                    self.max_saved_batches = storage_config.get('max_saved_batches', 100)
                    self.cleanup_old = storage_config.get('cleanup_old', True)
                    self.save_anchor_images = storage_config.get('save_images', True)
                    self.anchor_image_format = storage_config.get('image_format', 'png')
                    
                    # visualization配置
                    vis_config = agent_config.get('visualization', {})
                    self.fig_size = tuple(vis_config.get('fig_size', [12, 6]))
                    self.dpi = vis_config.get('dpi', 150)
                    self.style = vis_config.get('plot_style', 'default')
                    self.remove_plot_text = vis_config.get('remove_plot_text', True)
                    self.show_grid = vis_config.get('show_grid', True)
                    self.show_legend = vis_config.get('show_legend', False)
                    self.line_width = vis_config.get('line_width', 2)
                    self.fill_alpha = vis_config.get('fill_alpha', 0.3)
                    self.color_scheme = vis_config.get('color_scheme', 'default')
                    
                    self.log_info(f"Loaded config: use_vlm={self.use_vlm}, anchor_save={self.enable_anchor_save}")
            else:
                self._set_default_config()
                
        except Exception as e:
            self.log_error(f"Failed to load config: {e}, using defaults")
            self._set_default_config()
    
    def _set_default_config(self):
        self.use_vlm = False
        self.use_eas = False
        self.vlm_model_name = 'Qwen/Qwen-VL-Chat'
        self.anchor_strategy = 'confidence_interval'
        self.confidence_level = 0.95
        
        self.enable_anchor_save = True
        self.anchor_save_dir = './visualizations/visual_anchors/'
        self.anchor_save_format = 'json'
        self.update_frequency = 'every_batch'
        self.max_saved_batches = 100
        self.cleanup_old = True
        self.save_anchor_images = True
        self.anchor_image_format = 'png'
        
        self.fig_size = (12, 6)
        self.dpi = 150
        self.style = 'default'
        self.remove_plot_text = True
        self.show_grid = True
        self.show_legend = False
        self.line_width = 2
        self.fill_alpha = 0.3
        self.color_scheme = 'default'
        
    async def process(self, input_data: Dict[str, Any]) -> AgentOutput:
        """
        处理时序数据，生成视觉锚点
        
        Args:
            input_data: {
                'plot_path': str,  # Data Analyzer生成的plot图路径
                'statistics_text': str,  # 统计信息文本
                'data_features': Dict,  # 数据特征
                'task': str,  # 任务类型
                'pred_len': int  # 预测长度
            }
        """
        try:
            task = input_data.get('task', 'forecasting')
            pred_len = input_data.get('pred_len', 96)
            plot_path = input_data.get('plot_path', None)
            statistics_text = input_data.get('statistics_text', '')
            data_features = input_data.get('data_features', {})
            batch_idx = input_data.get('batch_idx', 0)
            
            self.log_info(f"Creating visual anchors for {task} task")
            
            # 1. 如果有plot图，使用VLM分析；否则使用data_features
            if plot_path and os.path.exists(plot_path):
                # 使用VLM分析plot图
                if self.use_vlm:
                    semantic_priors = await self._extract_semantic_priors(
                        plot_path, statistics_text, task
                    )
                else:
                    semantic_priors = {'method': 'rule_based', 'priors': self._extract_rule_based_priors_from_features(data_features, task)}
            else:
                self.log_info("No plot path provided, using rule-based priors")
                semantic_priors = {'method': 'rule_based', 'priors': self._extract_rule_based_priors_from_features(data_features, task)}
            
            # 2. 基于特征生成锚点
            if task == 'forecasting':
                anchors = self._generate_forecast_anchors_from_features(data_features, pred_len)
            elif task == 'classification':
                anchors = self._generate_classification_anchors_from_features(data_features)
            elif task == 'anomaly_detection':
                anchors = self._generate_anomaly_anchors_from_features(data_features)
            else:
                anchors = self._generate_default_anchors_from_features(data_features)
            
            # 3. 保存锚点可视化
            anchor_image_path = self._save_anchor_visualization(plot_path or '', anchors, batch_idx)
            
            result = {
                'visual_anchors': anchors,
                'semantic_priors': semantic_priors,
                'anchor_image_path': anchor_image_path,
                'metadata': {
                    'task': task,
                    'use_vlm': self.use_vlm,
                    'vlm_method': semantic_priors.get('method', 'unknown')
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
    
    def _init_local_vlm(self):
        """初始化本地VLM模型（Qwen-VL）"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            self.log_info(f"Loading local VLM: {self.vlm_model_name}")
            
            self.vlm_model = AutoModelForCausalLM.from_pretrained(
                self.vlm_model_name,
                device_map="auto",
                trust_remote_code=True
            ).eval()
            
            self.vlm_tokenizer = AutoTokenizer.from_pretrained(
                self.vlm_model_name,
                trust_remote_code=True
            )
            
            self.log_info("Local VLM loaded successfully")
            
        except Exception as e:
            self.log_error(f"Failed to load local VLM: {e}")
            self.use_vlm = False
    
    def _init_eas_client(self, config: Dict[str, Any]):
        """初始化EAS客户端 - 支持visual_anchor专用的EAS配置"""
        try:
            from src.utils.eas_client import EASClient
            import json
            from pathlib import Path
            import os
            
            # 优先级1: 环境变量 VLM_EAS_BASE_URL 和 VLM_EAS_TOKEN（专用）
            eas_base_url = os.environ.get('VLM_EAS_BASE_URL', '')
            eas_token = os.environ.get('VLM_EAS_TOKEN', '')
            
            # 优先级2: 通用环境变量
            if not eas_base_url:
                eas_base_url = os.environ.get('EAS_BASE_URL', '')
            if not eas_token:
                eas_token = os.environ.get('EAS_TOKEN', '')
            
            # 优先级3: 从config.json的agent专属配置读取
            if not eas_base_url or not eas_token:
                config_path = Path(__file__).parent.parent / 'config.json'
                if config_path.exists():
                    with open(config_path, 'r') as f:
                        json_config = json.load(f)
                        agent_config = json_config.get('agents_config', {}).get('visual_anchor', {})
                        eas_config = agent_config.get('eas_config', {})
                        if not eas_base_url:
                            eas_base_url = eas_config.get('base_url', eas_config.get('EAS_BASE_URL', ''))
                        if not eas_token:
                            eas_token = eas_config.get('token', eas_config.get('EAS_TOKEN', ''))
            
            # 优先级4: 从传入的config读取
            if not eas_base_url:
                eas_base_url = config.get('eas_base_url', config.get('eas_endpoint', ''))
            if not eas_token:
                eas_token = config.get('eas_token', '')
            
            if not eas_base_url or not eas_token:
                self.log_error("VLM EAS credentials not provided. Set VLM_EAS_BASE_URL/VLM_EAS_TOKEN env vars or configure in config.json -> agents_config.visual_anchor.eas_config")
                self.use_vlm = False
                return
            
            self.eas_client = EASClient(eas_base_url, eas_token)
            self.log_info(f"VLM EAS client initialized: {eas_base_url}")
            
        except Exception as e:
            self.log_error(f"Failed to initialize EAS client: {e}")
            self.use_vlm = False
    
    async def _extract_semantic_priors(self, plot_path: str, statistics_text: str, 
                                      task: str) -> Dict[str, Any]:
        """
        使用VLM提取语义先验
        
        Args:
            plot_path: 时序plot图路径
            statistics_text: 统计信息文本
            task: 任务类型
            
        Returns:
            语义先验字典
        """
        if not self.use_vlm:
            self.log_info("VLM disabled, using rule-based extraction")
            return {'method': 'rule_based', 'priors': {}}
        
        # 构建VLM prompt
        prompt = self._build_vlm_prompt(statistics_text, task)
        
        try:
            if self.use_eas:
                # 使用EAS服务
                response = await self.eas_client.call_vlm(plot_path, prompt)
            else:
                # 使用本地VLM
                response = await self._call_local_vlm(plot_path, prompt)
            
            # 解析VLM响应
            priors = self._parse_vlm_response(response, task)
            
            self.log_info(f"VLM semantic extraction completed: {priors.keys()}")
            return {'method': 'vlm', 'priors': priors}
            
        except Exception as e:
            self.log_error(f"VLM extraction failed: {e}, falling back to rule-based")
            return {'method': 'rule_based_fallback', 'priors': {}}
    
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
    
    def _build_vlm_prompt(self, statistics_text: str, task: str, pred_len: int = 96) -> str:
        """
        构建VLM prompt - 针对视觉推理优化
        使用详细的视觉描述引导，适合Qwen-VL等多模态模型
        """
        if task == 'forecasting':
            prompt = f"""You are an expert in visual time series analysis. Carefully examine the time series plot image.

STATISTICAL CONTEXT:
{statistics_text}

TASK: Forecast next {pred_len} time steps based on visual patterns

ANALYSIS REQUIREMENTS:
1. Visual Trend Analysis
   - Identify trend direction from plot curvature
   - Detect trend strength from slope steepness
   - Note any trend changes or inflection points

2. Pattern Recognition
   - Identify periodic/seasonal patterns visually
   - Detect peaks, troughs, and cycles
   - Recognize anomalies or outliers

3. Prediction Anchors
   - Define confidence intervals based on historical volatility (visible spread)
   - Identify 3-5 key anchor points for predictions
   - Estimate value ranges for each anchor

OUTPUT FORMAT (JSON):
{{
  "visual_observations": {{
    "trend": "increasing/decreasing/stable",
    "trend_strength": "weak/moderate/strong",
    "periodicity": "none/weak/strong",
    "volatility": "low/medium/high"
  }},
  "prediction_anchors": [
    {{"timestep": t1, "value": v1, "confidence": c1}},
    {{"timestep": t2, "value": v2, "confidence": c2}}
  ],
  "confidence_interval": {{"lower": L, "upper": U}},
  "overall_confidence": 0.0-1.0
}}

Focus on VISUAL cues from the plot. Be concise."""
        elif task == 'classification':
            prompt = f"""You are an expert in visual pattern recognition for time series classification.

STATISTICAL CONTEXT:
{statistics_text}

TASK: Identify visual patterns for classification

Analyze:
1. Shape characteristics (smooth/jagged, monotonic/oscillating)
2. Dominant frequencies and periodicities (visually)
3. Amplitude variations and volatility patterns
4. Unique visual signatures

OUTPUT (JSON):
{{
  "visual_features": {{
    "shape": "...",
    "periodicity": "...",
    "volatility_pattern": "..."
  }},
  "distinguishing_patterns": ["..."],
  "confidence": 0.0-1.0
}}"""
        else:
            prompt = f"""Analyze this time series plot visually.

{statistics_text}

Describe:
1. Main visual patterns
2. Trend and seasonality
3. Notable features

Output concise JSON with observations."""
        
        return prompt
    
    async def _call_local_vlm(self, image_path: str, prompt: str) -> Dict[str, Any]:
        """调用本地VLM模型"""
        try:
            image = Image.open(image_path)
            
            query = self.vlm_tokenizer.from_list_format([
                {'image': image_path},
                {'text': prompt}
            ])
            
            response, history = self.vlm_model.chat(
                self.vlm_tokenizer,
                query=query,
                history=None
            )
            
            return {'response': response, 'status': 'success'}
            
        except Exception as e:
            self.log_error(f"Local VLM call failed: {e}")
            return {'error': str(e), 'status': 'failed'}
    
    def _parse_vlm_response(self, response: Dict[str, Any], task: str) -> Dict[str, Any]:
        """解析VLM响应"""
        if 'error' in response or response.get('status') == 'failed':
            return {}
        
        try:
            # 尝试从响应中提取JSON
            response_text = response.get('response', '')
            
            # 查找JSON部分
            import re
            json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                parsed = json.loads(json_str)
                return parsed
            else:
                # 无法解析JSON，返回文本
                return {'raw_response': response_text}
                
        except Exception as e:
            self.log_error(f"Failed to parse VLM response: {e}")
            return {'raw_response': response.get('response', '')}
    
    def _extract_rule_based_priors_from_features(self, features: Optional[Dict], task: str) -> Dict[str, str]:
        """从特征提取rule-based语义先验"""
        priors = {}
        
        if features is None:
            return priors
        
        if 'trend' in features:
            trend_val = features['trend'].mean().item()
            if trend_val > 0.001:
                priors['trend'] = "The time series shows an increasing trend"
            elif trend_val < -0.001:
                priors['trend'] = "The time series shows a decreasing trend"
            else:
                priors['trend'] = "The time series is relatively stable"
        
        if 'std' in features:
            volatility = features['std'].mean().item()
            mean_val = features.get('mean', features['std']).mean().item()
            if volatility > abs(mean_val) * 0.5:
                priors['volatility'] = "High volatility detected"
            else:
                priors['volatility'] = "Low to moderate volatility"
        
        if 'autocorr_1' in features:
            autocorr = features['autocorr_1'].mean().item()
            if autocorr > 0.5:
                priors['periodicity'] = "Strong periodic pattern detected"
            elif autocorr > 0.3:
                priors['periodicity'] = "Moderate periodic pattern"
            else:
                priors['periodicity'] = "No clear periodic pattern"
        
        return priors
    
    def _generate_forecast_anchors_from_features(self, features: Dict, pred_len: int) -> Dict[str, Any]:
        """
        从特征生成预测锚点，包含预测区间和预测点
        """
        # 提取需要的统计量
        mean = features.get('mean', torch.tensor(0.0))
        std = features.get('std', torch.tensor(1.0))
        trend = features.get('trend', torch.tensor(0.0))
        
        # 生成锚点
        batch_size = mean.size(0) if mean.dim() > 0 else 1
        n_features = mean.size(1) if mean.dim() > 1 else 1
        device = mean.device if isinstance(mean, torch.Tensor) else 'cpu'
        
        # 生成未来时间步的预测
        # 创建时间索引
        future_time = torch.arange(1, pred_len + 1, device=device).float()
        future_time = future_time.unsqueeze(0).unsqueeze(2)  # [1, pred_len, 1]
        
        # 趋势外推
        if isinstance(trend, torch.Tensor) and trend.numel() > 0:
            # 处理不同维度的trend
            if trend.dim() == 0:  # 标量
                trend_expanded = trend.view(1, 1, 1).expand(batch_size, 1, n_features)
            elif trend.dim() == 1:  # [features]
                trend_expanded = trend.unsqueeze(0).unsqueeze(1)  # [1, 1, features]
                trend_expanded = trend_expanded.expand(batch_size, 1, n_features)
            elif trend.dim() == 2:  # [batch, features]
                trend_expanded = trend.unsqueeze(1)  # [batch, 1, features]
            else:
                trend_expanded = trend
            
            # 处理mean的维度
            if mean.dim() == 0:
                mean_expanded = mean.view(1, 1, 1).expand(batch_size, 1, n_features)
            elif mean.dim() == 1:  # [features]
                mean_expanded = mean.unsqueeze(0).unsqueeze(1).expand(batch_size, 1, n_features)
            elif mean.dim() == 2:  # [batch, features]
                mean_expanded = mean.unsqueeze(1)
            else:
                mean_expanded = mean
            
            # 线性趋势预测
            predictions = mean_expanded + trend_expanded * future_time  # [batch, pred_len, features]
        else:
            # 如果没有趋势，使用均值
            if mean.dim() == 0:
                mean_expanded = mean.view(1, 1, 1).expand(batch_size, pred_len, n_features)
            elif mean.dim() == 1:  # [features]
                mean_expanded = mean.unsqueeze(0).unsqueeze(1).expand(batch_size, pred_len, n_features)
            elif mean.dim() == 2:  # [batch, features]
                mean_expanded = mean.unsqueeze(1).expand(batch_size, pred_len, n_features)
            else:
                mean_expanded = mean
            predictions = mean_expanded
        
        # 置信区间（95%置信度，约1.96倍标准差）
        if std.dim() == 0:
            std_expanded = std.view(1, 1, 1).expand(batch_size, 1, n_features)
        elif std.dim() == 1:  # [features]
            std_expanded = std.unsqueeze(0).unsqueeze(1).expand(batch_size, 1, n_features)
        elif std.dim() == 2:  # [batch, features]
            std_expanded = std.unsqueeze(1)
        else:
            std_expanded = std
        z_score = 1.96  # 95% confidence
        
        # 随时间增长的不确定性
        time_factor = torch.sqrt(future_time / pred_len)  # 时间越远，不确定性越大
        upper_bound = predictions + z_score * std_expanded * time_factor
        lower_bound = predictions - z_score * std_expanded * time_factor
        
        # 生成关键锚点（均匀采样）
        num_anchor_points = min(5, pred_len)
        anchor_indices = torch.linspace(0, pred_len - 1, num_anchor_points, dtype=torch.long)
        
        anchor_predictions = predictions[:, anchor_indices, :]  # [batch, num_anchors, features]
        anchor_upper = upper_bound[:, anchor_indices, :]
        anchor_lower = lower_bound[:, anchor_indices, :]
        
        return {
            'type': 'confidence_interval',
            'predictions': {
                'point_forecast': predictions,  # [batch, pred_len, features]
                'upper_bound': upper_bound,      # [batch, pred_len, features]
                'lower_bound': lower_bound,      # [batch, pred_len, features]
            },
            'anchor_points': {
                'indices': anchor_indices,       # [num_anchors]
                'values': anchor_predictions,    # [batch, num_anchors, features]
                'upper': anchor_upper,           # [batch, num_anchors, features]
                'lower': anchor_lower,           # [batch, num_anchors, features]
            },
            'trend_direction': self._classify_trend_from_value(trend),
            'confidence': 0.85,
            'num_anchor_points': num_anchor_points
        }
    
    def _generate_classification_anchors_from_features(self, features: Dict) -> Dict[str, Any]:
        """从特征生成分类锚点"""
        return {
            'type': 'pattern_based',
            'features': features,
            'confidence': 0.80
        }
    
    def _generate_anomaly_anchors_from_features(self, features: Dict) -> Dict[str, Any]:
        """从特征生成异常检测锚点"""
        return {
            'type': 'anomaly_regions',
            'baseline_mean': features.get('mean', torch.tensor(0.0)),
            'baseline_std': features.get('std', torch.tensor(1.0)),
            'confidence': 0.75
        }
    
    def _generate_default_anchors_from_features(self, features: Dict) -> Dict[str, Any]:
        """从特征生成默认锚点"""
        return {
            'type': 'default',
            'summary': features,
            'confidence': 0.70
        }
    
    def _classify_trend_from_value(self, trend: torch.Tensor) -> str:
        """从trend值判断趋势方向"""
        if not isinstance(trend, torch.Tensor):
            return 'stable'
        trend_val = trend.mean().item() if trend.numel() > 0 else 0.0
        if trend_val > 0.001:
            return 'increasing'
        elif trend_val < -0.001:
            return 'decreasing'
        else:
            return 'stable'
    
    def _save_anchor_visualization(self, original_plot: str, anchors: Dict, batch_idx: int) -> str:
        """
        保存锚点可视化 - 支持本地存储和定期更新
        
        Args:
            original_plot: 原始plot路径
            anchors: 锚点数据
            batch_idx: 批次索引
            
        Returns:
            保存的图片路径
        """
        if not self.enable_anchor_save:
            self.log_info("Anchor存储未启用，跳过保存")
            return ""
        
        try:
            save_dir = Path(self.anchor_save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # 1. 保存anchor数据（JSON格式）
            if self.anchor_save_format == 'json':
                anchors_json_path = save_dir / f'batch_{batch_idx}_anchors.json'
                serializable_anchors = self._make_json_serializable(anchors)
                
                # 添加元数据
                anchor_data = {
                    'batch_idx': batch_idx,
                    'timestamp': self._get_timestamp(),
                    'anchors': serializable_anchors,
                    'config': {
                        'strategy': self.anchor_strategy,
                        'confidence_level': self.confidence_level
                    }
                }
                
                with open(anchors_json_path, 'w') as f:
                    json.dump(anchor_data, f, indent=2)
            
                self.log_info(f"Anchors保存至: {anchors_json_path}")
            
            # 2. 保存可视化图片（可选）
            image_path = ""
            if self.save_anchor_images:
                image_path = save_dir / f'batch_{batch_idx}_anchors.{self.anchor_image_format}'
                self._generate_anchor_plot(anchors, original_plot, str(image_path))
                self.log_info(f"Anchor可视化保存至: {image_path}")
            
            # 3. 定期清理旧文件
            if self.cleanup_old:
                self._cleanup_old_anchors(save_dir, batch_idx)
            
            return str(image_path) if image_path else str(save_dir / f'batch_{batch_idx}_anchors.json')
            
        except Exception as e:
            self.log_error(f"Failed to save anchor visualization: {e}")
            import traceback
            traceback.print_exc()
            return ""
    
    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    def _generate_anchor_plot(self, anchors: Dict, original_plot: str, save_path: str):
        """生成带锚点标注的可视化图"""
        try:
            # 创建图形
            fig, ax = plt.subplots(figsize=self.fig_size)
            
            # 如果有原始plot，加载并显示
            if original_plot and os.path.exists(original_plot):
                from PIL import Image
                img = Image.open(original_plot)
                ax.imshow(img)
                ax.axis('off')
            else:
                # 绘制锚点预测
                if anchors.get('type') == 'confidence_interval':
                    predictions = anchors.get('predictions', {})
                    point_forecast = predictions.get('point_forecast', None)
                    
                    if isinstance(point_forecast, list) and len(point_forecast) > 0:
                        # 绘制第一个batch的第一个特征
                        forecast = point_forecast[0]  # 假设是list格式
                        if isinstance(forecast, list) and len(forecast) > 0:
                            time_steps = list(range(len(forecast)))
                            values = [f[0] if isinstance(f, list) else f for f in forecast]
                            
                            ax.plot(time_steps, values, 'b-', linewidth=2, label='Point Forecast')
                            
                            # 绘制置信区间
                            upper = predictions.get('upper_bound', None)
                            lower = predictions.get('lower_bound', None)
                            if upper and lower and isinstance(upper, list) and isinstance(lower, list):
                                upper_vals = [u[0][0] if isinstance(u, list) else u for u in upper]
                                lower_vals = [l[0][0] if isinstance(l, list) else l for l in lower]
                                ax.fill_between(time_steps, lower_vals, upper_vals, alpha=0.3)
                            
                            ax.set_xlabel('Time Steps')
                            ax.set_ylabel('Value')
                            ax.set_title('Visual Anchors')
                            ax.legend()
                            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)
            
        except Exception as e:
            self.log_error(f"Failed to generate anchor plot: {e}")
    
    def _cleanup_old_anchors(self, save_dir: Path, current_batch: int):
        """清理旧的anchor文件"""
        try:
            if current_batch <= self.max_saved_batches:
                return
            
            # 删除batch索引小于 current_batch - max_saved_batches 的文件
            threshold_batch = current_batch - self.max_saved_batches
            
            for file_path in save_dir.glob('batch_*_anchors.*'):
                try:
                    # 提取batch索引
                    filename = file_path.stem
                    batch_num = int(filename.split('_')[1])
                    
                    if batch_num < threshold_batch:
                        file_path.unlink()
                        self.log_info(f"清理旧文件: {file_path}")
                        
                except (ValueError, IndexError):
                    # 文件名格式不匹配，跳过
                    continue
                    
        except Exception as e:
            self.log_error(f"Failed to cleanup old anchors: {e}")
    
    def _make_json_serializable(self, obj):
        """递归将对象转换为JSON可序列化格式"""
        if isinstance(obj, torch.Tensor):
            return obj.cpu().detach().tolist()
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            # 对于其他类型，尝试转换为字符串
            return str(obj)

