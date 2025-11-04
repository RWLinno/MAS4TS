"""
NumerologicAdapterAgent: 数值推理适配器
结合视觉锚点、语义先验和原始时序数据，进行精确的数值推理
这是MAS4TS的另一个核心创新
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from .base_agent_ts import BaseAgentTS, AgentOutput
import logging

logger = logging.getLogger(__name__)


class NumerologicAdapterAgent(BaseAgentTS):
    """
    数值逻辑适配器Agent：融合多模态信息进行数值推理
    
    核心创新:
    1. 将视觉锚点转换为数值约束
    2. 将语义先验编码为推理引导
    3. 结合原始数据进行精确的数值预测
    4. 使用注意力机制融合多源信息
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("NumerologicAdapterAgent", config)
        
        self.device = config.get('device', 'cpu')
        self.hidden_dim = config.get('hidden_dim', 128)
        self.num_layers = config.get('num_layers', 2)
        
        # 融合策略
        self.fusion_strategy = config.get('fusion_strategy', 'attention')  # 'attention', 'concat', 'weighted'
        
        # 初始化融合网络
        self._init_fusion_networks()
        
    def _init_fusion_networks(self):
        """初始化融合网络"""
        # 注意力融合模块
        self.anchor_attention = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1)
        ).to(self.device)
        
        # 特征融合MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU()
        ).to(self.device)
        
        self.log_info("Fusion networks initialized")
    
    async def process(self, input_data: Dict[str, Any]) -> AgentOutput:
        """
        处理多模态输入，生成融合特征
        
        Args:
            input_data: {
                'data': torch.Tensor,  # 原始时序数据 [batch, seq_len, features]
                'anchors': Dict,  # 视觉锚点
                'semantic_priors': Dict,  # 语义先验
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
            
            data = input_data['data']
            anchors = input_data.get('anchors', {})
            semantic_priors = input_data.get('semantic_priors', {})
            
            self.log_info(f"Adapting features with fusion strategy: {self.fusion_strategy}")
            
            # 1. 编码原始数据特征
            data_features = self._encode_data_features(data)
            
            # 2. 编码锚点约束
            anchor_features = self._encode_anchor_constraints(anchors, data.shape)
            
            # 3. 编码语义先验
            semantic_features = self._encode_semantic_priors(semantic_priors, data.shape)
            
            # 4. 多模态融合
            fused_features = self._fuse_multimodal_features(
                data_features, anchor_features, semantic_features
            )
            
            # 5. 生成数值推理结果
            adapted_output = self._generate_numerical_output(fused_features, data)
            
            result = {
                'adapted_features': fused_features,
                'numerical_constraints': adapted_output['constraints'],
                'confidence_weights': adapted_output['confidence'],
                'metadata': {
                    'fusion_strategy': self.fusion_strategy,
                    'feature_dim': fused_features.shape
                }
            }
            
            return AgentOutput(
                agent_name=self.name,
                success=True,
                result=result,
                confidence=0.92,
                metadata={'fusion_strategy': self.fusion_strategy}
            )
            
        except Exception as e:
            self.log_error(f"Error in numerical adaptation: {e}")
            import traceback
            traceback.print_exc()
            return AgentOutput(
                agent_name=self.name,
                success=False,
                result=None,
                confidence=0.0,
                metadata={'error': str(e)}
            )
    
    def _encode_data_features(self, data: torch.Tensor) -> torch.Tensor:
        """
        编码原始数据特征
        
        Args:
            data: [batch, seq_len, features]
            
        Returns:
            encoded_features: [batch, hidden_dim]
        """
        # 提取统计特征
        mean = data.mean(dim=1)  # [batch, features]
        std = data.std(dim=1)
        min_val = data.min(dim=1)[0]
        max_val = data.max(dim=1)[0]
        
        # 最后几个时间步的值（用于趋势）
        last_values = data[:, -5:, :].mean(dim=1)  # [batch, features]
        
        # 拼接特征
        stat_features = torch.cat([mean, std, min_val, max_val, last_values], dim=1)
        
        # 投影到hidden_dim
        if stat_features.size(1) != self.hidden_dim:
            # 简单的线性投影
            if not hasattr(self, 'data_projection'):
                self.data_projection = nn.Linear(stat_features.size(1), self.hidden_dim).to(self.device)
            encoded = self.data_projection(stat_features)
        else:
            encoded = stat_features
        
        return encoded  # [batch, hidden_dim]
    
    def _encode_anchor_constraints(self, anchors: Dict, data_shape: tuple) -> torch.Tensor:
        """
        编码锚点约束为特征向量
        
        Args:
            anchors: 锚点字典
            data_shape: (batch, seq_len, features)
            
        Returns:
            anchor_features: [batch, hidden_dim]
        """
        batch_size = data_shape[0]
        
        if not anchors or anchors.get('type') is None:
            # 没有锚点，返回零向量
            return torch.zeros(batch_size, self.hidden_dim, device=self.device)
        
        anchor_type = anchors['type']
        
        if anchor_type == 'confidence_interval':
            # 提取置信区间信息
            full_traj = anchors.get('full_trajectory', {})
            if 'mean' in full_traj:
                mean_traj = full_traj['mean']  # [batch, pred_len, features]
                upper_traj = full_traj.get('upper', mean_traj)
                lower_traj = full_traj.get('lower', mean_traj)
                
                # 提取关键统计量
                anchor_mean = mean_traj.mean(dim=(1, 2))  # [batch]
                anchor_range = (upper_traj - lower_traj).mean(dim=(1, 2))  # [batch]
                
                # 趋势方向编码
                trend_directions = anchors.get('trend_direction', [])
                trend_encoding = self._encode_trend_direction(trend_directions, batch_size)
                
                # 拼接
                anchor_stats = torch.stack([anchor_mean, anchor_range, trend_encoding], dim=1)  # [batch, 3]
            else:
                anchor_stats = torch.zeros(batch_size, 3, device=self.device)
        
        elif anchor_type == 'pattern_based':
            # 模式特征
            pattern_features = anchors.get('features', {})
            feature_list = []
            for key in ['num_peaks', 'num_valleys', 'zero_crossings']:
                if key in pattern_features:
                    feat = pattern_features[key].mean(dim=1) if pattern_features[key].dim() > 1 else pattern_features[key]
                    feature_list.append(feat)
            
            if feature_list:
                anchor_stats = torch.stack(feature_list, dim=1)  # [batch, num_features]
            else:
                anchor_stats = torch.zeros(batch_size, 3, device=self.device)
        
        else:
            anchor_stats = torch.zeros(batch_size, 3, device=self.device)
        
        # 投影到hidden_dim
        if not hasattr(self, 'anchor_projection'):
            self.anchor_projection = nn.Linear(anchor_stats.size(1), self.hidden_dim).to(self.device)
        
        anchor_features = self.anchor_projection(anchor_stats)
        
        return anchor_features  # [batch, hidden_dim]
    
    def _encode_semantic_priors(self, semantic_priors: Dict, data_shape: tuple) -> torch.Tensor:
        """
        编码语义先验为特征向量
        
        Args:
            semantic_priors: 语义先验字典
            data_shape: (batch, seq_len, features)
            
        Returns:
            semantic_features: [batch, hidden_dim]
        """
        batch_size = data_shape[0]
        
        if not semantic_priors:
            return torch.zeros(batch_size, self.hidden_dim, device=self.device)
        
        # 简单的规则编码
        semantic_vector = []
        
        # 趋势编码
        trend_text = semantic_priors.get('trend', '')
        if 'increasing' in trend_text.lower():
            trend_val = 1.0
        elif 'decreasing' in trend_text.lower():
            trend_val = -1.0
        else:
            trend_val = 0.0
        semantic_vector.append(trend_val)
        
        # 波动性编码
        volatility_text = semantic_priors.get('volatility', '')
        if 'high' in volatility_text.lower():
            vol_val = 1.0
        elif 'low' in volatility_text.lower():
            vol_val = 0.0
        else:
            vol_val = 0.5
        semantic_vector.append(vol_val)
        
        # 周期性编码
        periodicity_text = semantic_priors.get('periodicity', '')
        if 'strong' in periodicity_text.lower():
            period_val = 1.0
        elif 'moderate' in periodicity_text.lower():
            period_val = 0.5
        else:
            period_val = 0.0
        semantic_vector.append(period_val)
        
        # 转换为tensor并扩展到batch
        semantic_tensor = torch.tensor(semantic_vector, device=self.device, dtype=torch.float32)
        semantic_tensor = semantic_tensor.unsqueeze(0).expand(batch_size, -1)  # [batch, 3]
        
        # 投影到hidden_dim
        if not hasattr(self, 'semantic_projection'):
            self.semantic_projection = nn.Linear(3, self.hidden_dim).to(self.device)
        
        semantic_features = self.semantic_projection(semantic_tensor)
        
        return semantic_features  # [batch, hidden_dim]
    
    def _encode_trend_direction(self, trend_directions: list, batch_size: int) -> torch.Tensor:
        """编码趋势方向"""
        if not trend_directions:
            return torch.zeros(batch_size, device=self.device)
        
        encoding = []
        for direction in trend_directions[:batch_size]:
            if direction == 'increasing':
                encoding.append(1.0)
            elif direction == 'decreasing':
                encoding.append(-1.0)
            else:
                encoding.append(0.0)
        
        # 填充到batch_size
        while len(encoding) < batch_size:
            encoding.append(0.0)
        
        return torch.tensor(encoding[:batch_size], device=self.device, dtype=torch.float32)
    
    def _fuse_multimodal_features(self, data_features: torch.Tensor, 
                                   anchor_features: torch.Tensor,
                                   semantic_features: torch.Tensor) -> torch.Tensor:
        """
        融合多模态特征
        
        Args:
            data_features: [batch, hidden_dim]
            anchor_features: [batch, hidden_dim]
            semantic_features: [batch, hidden_dim]
            
        Returns:
            fused_features: [batch, hidden_dim]
        """
        if self.fusion_strategy == 'attention':
            # 注意力融合
            # 计算每个模态的注意力权重
            data_attn = self.anchor_attention(data_features)  # [batch, 1]
            anchor_attn = self.anchor_attention(anchor_features)
            semantic_attn = self.anchor_attention(semantic_features)
            
            # Softmax归一化
            attn_weights = torch.softmax(
                torch.cat([data_attn, anchor_attn, semantic_attn], dim=1),
                dim=1
            )  # [batch, 3]
            
            # 加权求和
            fused = (data_features * attn_weights[:, 0:1] + 
                    anchor_features * attn_weights[:, 1:2] + 
                    semantic_features * attn_weights[:, 2:3])
            
        elif self.fusion_strategy == 'concat':
            # 拼接后MLP
            concatenated = torch.cat([data_features, anchor_features, semantic_features], dim=1)
            fused = self.fusion_mlp(concatenated)
            
        elif self.fusion_strategy == 'weighted':
            # 简单加权平均
            weights = torch.tensor([0.5, 0.3, 0.2], device=self.device)
            fused = (data_features * weights[0] + 
                    anchor_features * weights[1] + 
                    semantic_features * weights[2])
        else:
            # 默认：简单平均
            fused = (data_features + anchor_features + semantic_features) / 3.0
        
        return fused  # [batch, hidden_dim]
    
    def _generate_numerical_output(self, fused_features: torch.Tensor, 
                                   original_data: torch.Tensor) -> Dict[str, Any]:
        """
        基于融合特征生成数值推理输出
        
        Args:
            fused_features: [batch, hidden_dim]
            original_data: [batch, seq_len, features]
            
        Returns:
            Dict包含约束和置信度
        """
        batch_size = fused_features.size(0)
        
        # 生成数值约束
        # 这里可以训练一个小网络来预测约束
        # 暂时使用简单的启发式方法
        
        # 基于原始数据的统计
        data_std = original_data.std(dim=1)  # [batch, features]
        data_mean = original_data.mean(dim=1)  # [batch, features]
        
        # 约束范围：均值 ± 2*std
        upper_bound = data_mean + 2 * data_std
        lower_bound = data_mean - 2 * data_std
        
        constraints = {
            'upper_bound': upper_bound,
            'lower_bound': lower_bound,
            'expected_value': data_mean
        }
        
        # 置信度（基于特征的范数）
        confidence = torch.sigmoid(fused_features.norm(dim=1) / 10.0)  # [batch]
        
        return {
            'constraints': constraints,
            'confidence': confidence
        }

