"""
NumerologicAdapterAgent: 数值推理适配器
结合视觉锚点、语义先验和原始时序数据，进行精确的数值推理
这是MAS4TS的另一个核心创新
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from .base_agent_ts import BaseAgentTS, AgentOutput
import asyncio
import json
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
        self.fusion_strategy = config.get('fusion_strategy', 'attention')
        
        # LLM配置
        self.use_llm = config.get('use_llm', False)
        self.use_eas = config.get('use_eas', False)
        self.num_llm_models = config.get('num_llm_models', 3)
        
        # 初始化融合网络
        self._init_fusion_networks()
        
        # 初始化EAS客户端
        if self.use_llm and self.use_eas:
            self._init_eas_client(config)
        
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
            
            # 5. 如果启用LLM，进行并发数值推理
            if input_data.get('use_parallel_llm', False) and self.use_llm:
                statistics_text = input_data.get('statistics_text', '')
                num_models = input_data.get('num_llm_models', self.num_llm_models)
                llm_output = await self._parallel_llm_inference(
                    data, anchors, statistics_text, num_models
                )
                adapted_output = self._generate_numerical_output(fused_features, data)
                adapted_output['llm_reasoning'] = llm_output
            else:
                # 不使用LLM，直接生成数值推理结果
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
    
    def _init_eas_client(self, config: Dict[str, Any]):
        """初始化EAS客户端 - 支持多个LLM模型的独立EAS配置"""
        try:
            from src.utils.eas_client import EASClient
            import json
            from pathlib import Path
            import os
            
            # 读取config.json中的llm_ensemble配置
            config_path = Path(__file__).parent.parent / 'config.json'
            llm_ensemble = []
            
            if config_path.exists():
                with open(config_path, 'r') as f:
                    json_config = json.load(f)
                    agent_config = json_config.get('agents_config', {}).get('numerologic_adapter', {})
                    llm_ensemble = agent_config.get('llm_ensemble', [])
            
            # 为每个LLM模型创建EAS客户端
            self.llm_clients = []
            
            for llm_config in llm_ensemble:
                model_name = llm_config.get('model_name', '')
                eas_config = llm_config.get('eas_config', {})
                
                # 优先从环境变量读取（支持模型专用环境变量）
                # 例如: QWEN_7B_EAS_BASE_URL, QWEN_7B_EAS_TOKEN
                env_prefix = model_name.split('/')[-1].upper().replace('.', '_').replace('-', '_')
                eas_base_url = os.environ.get(f'{env_prefix}_EAS_BASE_URL', '')
                eas_token = os.environ.get(f'{env_prefix}_EAS_TOKEN', '')
                
                # 如果没有专用环境变量，使用通用环境变量
                if not eas_base_url:
                    eas_base_url = os.environ.get('LLM_EAS_BASE_URL', '')
                if not eas_token:
                    eas_token = os.environ.get('LLM_EAS_TOKEN', '')
                
                # 从配置文件读取
                if not eas_base_url:
                    eas_base_url = eas_config.get('base_url', '')
                if not eas_token:
                    eas_token = eas_config.get('token', '')
                
                if eas_base_url and eas_token:
                    client = EASClient(eas_base_url, eas_token)
                    self.llm_clients.append({
                        'model_name': model_name,
                        'client': client,
                        'max_tokens': llm_config.get('max_tokens', 512),
                        'temperature': llm_config.get('temperature', 0.5)
                    })
                    self.log_info(f"LLM EAS client initialized for {model_name}: {eas_base_url}")
            else:
                    self.log_warning(f"EAS credentials not provided for {model_name}, skipping")
            
            if not self.llm_clients:
                self.log_error("No LLM EAS clients initialized. Configure in config.json -> agents.numerologic_adapter.llm_ensemble")
                self.use_llm = False
            else:
                self.log_info(f"Initialized {len(self.llm_clients)} LLM EAS clients")
                
        except Exception as e:
            self.log_error(f"Failed to init EAS clients: {e}")
            self.use_llm = False
    
    async def _parallel_llm_inference(self, data_features: Dict, visual_anchors: Dict, 
                                     statistics_text: str, num_models: int = 3) -> Dict[str, Any]:
        """
        并发调用多个LLM模型进行数值推理
        
        Args:
            data_features: 数据特征
            visual_anchors: 视觉锚点
            statistics_text: 统计文本
            num_models: 并发模型数量
            
        Returns:
            ensemble后的预测结果
        """
        if not hasattr(self, 'llm_clients') or not self.llm_clients:
            self.log_error("No LLM clients available, using rule-based fallback")
            return self._rule_based_numerical_reasoning(data_features, visual_anchors)
        
        # 使用配置的LLM客户端数量，但不超过num_models
        num_models = min(num_models, len(self.llm_clients))
        self.log_info(f"Starting parallel LLM inference with {num_models} models")
        
        # 构建prompt
        prompt = self._build_numerical_reasoning_prompt(
            data_features, visual_anchors, statistics_text
        )
        
        # 并发调用多个模型
        tasks = []
        for i in range(num_models):
            client_info = self.llm_clients[i]
            tasks.append(self._call_single_llm_with_client(client_info, prompt))
        
        # 收集结果
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 过滤有效结果
        valid_results = [r for r in results if not isinstance(r, Exception) and 'error' not in r]
        
        if valid_results:
            self.log_info(f"Got {len(valid_results)} valid LLM responses")
            return self._ensemble_llm_results(valid_results)
        else:
            self.log_error("All LLM calls failed, using rule-based fallback")
            return self._rule_based_numerical_reasoning(data_features, visual_anchors)
    
    def _build_numerical_reasoning_prompt(self, data_features: Dict, 
                                         visual_anchors: Dict, 
                                         statistics_text: str) -> str:
        """
        构建LLM数值推理prompt - 针对数值推理优化
        使用Qwen等数学推理能力强的模型
        """
        # 提取关键数值统计
        mean_val = data_features.get('mean', torch.tensor(0)).mean().item() if 'mean' in data_features else 0
        std_val = data_features.get('std', torch.tensor(0)).mean().item() if 'std' in data_features else 0
        trend_val = data_features.get('trend', torch.tensor(0)).mean().item() if 'trend' in data_features else 0
        
        prompt = f"""You are an expert in numerical reasoning for time series forecasting.

STATISTICAL SUMMARY:
{statistics_text}

NUMERICAL DATA:
- Historical Mean: {mean_val:.6f}
- Standard Deviation: {std_val:.6f}
- Linear Trend Slope: {trend_val:.8f}
- Autocorrelation(1): {data_features.get('autocorr_1', torch.tensor(0)).mean().item() if 'autocorr_1' in data_features else 0:.4f}

VISUAL ANCHOR ANALYSIS:
- Value Range: {visual_anchors.get('value_range', 'Not available')}
- Trend Direction: {visual_anchors.get('trend_direction', 'Not available')}
- Visual Confidence: {visual_anchors.get('confidence', 'N/A')}

TASK: Perform rigorous numerical reasoning to:

1. VALIDATE visual predictions using statistics
   - Check if visual range aligns with mean ± k*std
   - Verify trend direction matches slope sign
   - Assess if confidence is justified by volatility

2. REFINE value range predictions
   - Apply statistical bounds (e.g., 95% CI)
   - Consider trend extrapolation
   - Account for autocorrelation effects

3. COMPUTE anchor values
   - Provide 3-5 specific numerical predictions
   - Include uncertainty quantification
   - Consider both point estimates and intervals

OUTPUT (JSON):
{{
  "validated_range": {{"lower": L, "upper": U, "method": "statistical/visual/hybrid"}},
  "point_predictions": [v1, v2, v3, v4, v5],
  "confidence_scores": [c1, c2, c3, c4, c5],
  "statistical_validity": {{"visual_alignment": "good/moderate/poor", "notes": "..."}},
  "overall_confidence": 0.0-1.0
}}

Use mathematical reasoning. Be precise with numbers."""
        return prompt
    
    async def _call_single_llm_with_client(self, client_info: Dict, prompt: str) -> Dict[str, Any]:
        """使用指定的客户端调用单个LLM模型"""
        model_name = client_info['model_name']
        client = client_info['client']
        max_tokens = client_info.get('max_tokens', 512)
        temperature = client_info.get('temperature', 0.5)
        
        try:
            # 使用EAS客户端调用
            response = await client.call_llm(
                prompt=prompt,
                model_name=model_name,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            self.log_info(f"LLM {model_name} responded")
            return response
            
        except Exception as e:
            self.log_error(f"LLM {model_name} call failed: {e}")
            return {'error': str(e)}
    
    async def _call_single_llm(self, model_name: str, prompt: str) -> Dict[str, Any]:
        """调用单个LLM模型（保留兼容性）"""
        try:
            if self.use_eas and hasattr(self, 'eas_client'):
                # 使用EAS服务
                response = await self.eas_client.call_llm(prompt, model_name)
            else:
                # 本地LLM推理（简化实现）
                response = await self._call_local_llm(model_name, prompt)
            
            self.log_info(f"LLM {model_name} responded")
            return response
            
        except Exception as e:
            self.log_error(f"LLM {model_name} call failed: {e}")
            return {'error': str(e)}
    
    async def _call_local_llm(self, model_name: str, prompt: str) -> Dict[str, Any]:
        """本地LLM调用（简化版）"""
        # 简化：不实际加载大模型，返回基于规则的结果
        self.log_info(f"Local LLM call not implemented, using rule-based for {model_name}")
        return {
            'response': json.dumps({
                'refined_range': [0.0, 1.0],
                'confidence': 0.75,
                'anchor_values': [0.25, 0.5, 0.75],
                'reasoning': 'Rule-based fallback'
            }),
            'status': 'success'
        }
    
    def _ensemble_llm_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """ensemble多个LLM的结果"""
        try:
            # 解析所有响应
            parsed_results = []
            for result in results:
                response_text = result.get('response', '')
                try:
                    import re
                    json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
                    if json_match:
                        parsed = json.loads(json_match.group())
                        parsed_results.append(parsed)
                except:
                    pass
            
            if not parsed_results:
                return {'confidence': 0.5}
            
            # 简单平均
            avg_confidence = sum(r.get('confidence', 0.5) for r in parsed_results) / len(parsed_results)
            
            return {
                'ensemble_confidence': avg_confidence,
                'num_models': len(parsed_results),
                'individual_results': parsed_results
            }
            
        except Exception as e:
            self.log_error(f"Ensemble failed: {e}")
            return {'confidence': 0.5}
    
    def _rule_based_numerical_reasoning(self, data_features: Dict, 
                                       visual_anchors: Dict) -> Dict[str, Any]:
        """基于规则的数值推理（LLM fallback）"""
        return {
            'ensemble_confidence': 0.70,
            'method': 'rule_based',
            'reasoning': 'LLM not available, using statistical methods'
        }

