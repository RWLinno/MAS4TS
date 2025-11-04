"""
KnowledgeRetrieverAgent: 知识检索agent
维护时序模式的向量库，支持相似样本检索和领域知识查询
"""

import torch
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from .base_agent_ts import BaseAgentTS, AgentOutput
import logging

logger = logging.getLogger(__name__)


class KnowledgeRetrieverAgent(BaseAgentTS):
    """
    知识检索Agent：检索相似的历史模式和领域知识
    
    职责:
    1. 维护时序模式的向量库
    2. 基于相似度检索历史案例
    3. 提供领域知识和专家经验
    4. 支持Few-shot和Zero-shot学习
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("KnowledgeRetrieverAgent", config)
        
        self.use_vector_db = config.get('use_vector_db', False)
        self.k_neighbors = config.get('k_neighbors', 5)
        self.similarity_threshold = config.get('similarity_threshold', 0.7)
        
        # 知识库
        self.knowledge_base = {
            'patterns': [],  # 存储模式特征
            'embeddings': None,  # 特征embedding
            'metadata': []  # 元数据（任务类型、数据集等）
        }
        
        # 领域知识
        self.domain_knowledge = self._load_domain_knowledge()
        
    def _load_domain_knowledge(self) -> Dict[str, str]:
        """加载领域知识"""
        return {
            'forecasting': {
                'trend_following': "For time series with strong trends, trend-following methods often work well",
                'seasonal': "Seasonal patterns require models that can capture periodicity",
                'volatile': "High volatility series benefit from ensemble methods",
            },
            'classification': {
                'pattern_matching': "Classification relies on identifying discriminative patterns",
                'feature_importance': "Peak detection and frequency features are often important",
            },
            'anomaly_detection': {
                'statistical': "Statistical methods work well for point anomalies",
                'contextual': "Contextual anomalies require understanding of normal patterns",
            }
        }
    
    async def process(self, input_data: Dict[str, Any]) -> AgentOutput:
        """
        处理知识检索请求
        
        Args:
            input_data: {
                'data': torch.Tensor,  # [batch, seq_len, features]
                'task': str,  # 任务类型
                'query_type': str,  # 'similar_patterns', 'domain_knowledge'
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
            query_type = input_data.get('query_type', 'both')
            
            self.log_info(f"Retrieving knowledge for task: {task}")
            
            result = {}
            
            # 检索相似模式
            if query_type in ['similar_patterns', 'both']:
                similar_patterns = self._retrieve_similar_patterns(data, task)
                result['similar_patterns'] = similar_patterns
            
            # 检索领域知识
            if query_type in ['domain_knowledge', 'both']:
                domain_info = self._retrieve_domain_knowledge(data, task)
                result['domain_knowledge'] = domain_info
            
            # 生成建议
            recommendations = self._generate_recommendations(data, task, result)
            result['recommendations'] = recommendations
            
            return AgentOutput(
                agent_name=self.name,
                success=True,
                result=result,
                confidence=0.85,
                metadata={'task': task, 'query_type': query_type}
            )
            
        except Exception as e:
            self.log_error(f"Error in knowledge retrieval: {e}")
            import traceback
            traceback.print_exc()
            return AgentOutput(
                agent_name=self.name,
                success=False,
                result=None,
                confidence=0.0,
                metadata={'error': str(e)}
            )
    
    def _retrieve_similar_patterns(self, data: torch.Tensor, task: str) -> List[Dict[str, Any]]:
        """
        检索相似的历史模式
        
        Args:
            data: [batch, seq_len, features]
            task: 任务类型
            
        Returns:
            List of similar patterns
        """
        if not self.knowledge_base['patterns']:
            self.log_info("Knowledge base is empty, no similar patterns found")
            return []
        
        # 提取查询特征
        query_features = self._extract_pattern_features(data)  # [batch, feature_dim]
        
        # 计算相似度
        if self.knowledge_base['embeddings'] is not None:
            kb_embeddings = self.knowledge_base['embeddings']  # [num_patterns, feature_dim]
            
            # 余弦相似度
            similarities = self._compute_cosine_similarity(query_features, kb_embeddings)
            
            # Top-K检索
            top_k_indices = torch.topk(similarities, k=min(self.k_neighbors, similarities.size(1)), dim=1)
            
            # 构建结果
            similar_patterns = []
            for i in range(top_k_indices.indices.size(0)):
                batch_patterns = []
                for j in range(top_k_indices.indices.size(1)):
                    idx = top_k_indices.indices[i, j].item()
                    similarity = top_k_indices.values[i, j].item()
                    
                    if similarity >= self.similarity_threshold:
                        batch_patterns.append({
                            'pattern_id': idx,
                            'similarity': similarity,
                            'metadata': self.knowledge_base['metadata'][idx] if idx < len(self.knowledge_base['metadata']) else {}
                        })
                similar_patterns.append(batch_patterns)
            
            return similar_patterns
        else:
            return []
    
    def _retrieve_domain_knowledge(self, data: torch.Tensor, task: str) -> Dict[str, Any]:
        """
        检索领域知识
        
        Args:
            data: [batch, seq_len, features]
            task: 任务类型
            
        Returns:
            Domain knowledge dict
        """
        domain_info = {}
        
        # 获取任务相关的领域知识
        if task in self.domain_knowledge:
            domain_info['task_knowledge'] = self.domain_knowledge[task]
        else:
            domain_info['task_knowledge'] = {}
        
        # 基于数据特性提供建议
        data_characteristics = self._analyze_data_characteristics(data)
        domain_info['data_characteristics'] = data_characteristics
        
        # 匹配相关知识
        relevant_knowledge = []
        for key, desc in domain_info.get('task_knowledge', {}).items():
            if self._is_knowledge_relevant(key, data_characteristics):
                relevant_knowledge.append({
                    'topic': key,
                    'description': desc,
                    'relevance': 'high'
                })
        
        domain_info['relevant_knowledge'] = relevant_knowledge
        
        return domain_info
    
    def _extract_pattern_features(self, data: torch.Tensor) -> torch.Tensor:
        """
        提取模式特征用于相似度计算
        
        Args:
            data: [batch, seq_len, features]
            
        Returns:
            features: [batch, feature_dim]
        """
        # 统计特征
        mean = data.mean(dim=1)  # [batch, features]
        std = data.std(dim=1)
        skew = self._compute_skewness(data)
        
        # 形状特征
        trend = self._compute_trend(data)
        autocorr = self._compute_autocorr_features(data)
        
        # 拼接所有特征
        features = torch.cat([
            mean, std, skew, trend, autocorr
        ], dim=1)  # [batch, feature_dim]
        
        # L2归一化
        features = torch.nn.functional.normalize(features, p=2, dim=1)
        
        return features
    
    def _compute_skewness(self, data: torch.Tensor) -> torch.Tensor:
        """计算偏度"""
        mean = data.mean(dim=1, keepdim=True)
        std = data.std(dim=1, keepdim=True) + 1e-8
        
        centered = (data - mean) / std
        skewness = (centered ** 3).mean(dim=1)
        
        return skewness
    
    def _compute_trend(self, data: torch.Tensor) -> torch.Tensor:
        """计算趋势"""
        batch, seq_len, features = data.shape
        
        t = torch.arange(seq_len, device=data.device, dtype=data.dtype)
        t = t.unsqueeze(0).unsqueeze(2).expand(batch, seq_len, features)
        
        t_mean = t.mean(dim=1, keepdim=True)
        y_mean = data.mean(dim=1, keepdim=True)
        
        numerator = ((t - t_mean) * (data - y_mean)).sum(dim=1)
        denominator = ((t - t_mean) ** 2).sum(dim=1) + 1e-8
        
        return numerator / denominator
    
    def _compute_autocorr_features(self, data: torch.Tensor, lags: List[int] = [1, 5, 10]) -> torch.Tensor:
        """计算多个lag的自相关特征"""
        autocorr_list = []
        
        for lag in lags:
            if lag < data.size(1):
                y1 = data[:, :-lag, :]
                y2 = data[:, lag:, :]
                
                y1_centered = y1 - y1.mean(dim=1, keepdim=True)
                y2_centered = y2 - y2.mean(dim=1, keepdim=True)
                
                numerator = (y1_centered * y2_centered).sum(dim=1)
                denominator = torch.sqrt((y1_centered ** 2).sum(dim=1) * 
                                        (y2_centered ** 2).sum(dim=1)) + 1e-8
                
                autocorr = numerator / denominator
                autocorr_list.append(autocorr)
            else:
                autocorr_list.append(torch.zeros(data.size(0), data.size(2), device=data.device))
        
        return torch.cat(autocorr_list, dim=1)  # [batch, num_lags * features]
    
    def _compute_cosine_similarity(self, query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """
        计算余弦相似度
        
        Args:
            query: [batch, feature_dim]
            keys: [num_keys, feature_dim]
            
        Returns:
            similarities: [batch, num_keys]
        """
        # 归一化
        query_norm = torch.nn.functional.normalize(query, p=2, dim=1)
        keys_norm = torch.nn.functional.normalize(keys, p=2, dim=1)
        
        # 计算相似度
        similarities = torch.matmul(query_norm, keys_norm.T)  # [batch, num_keys]
        
        return similarities
    
    def _analyze_data_characteristics(self, data: torch.Tensor) -> Dict[str, Any]:
        """分析数据特性"""
        characteristics = {}
        
        # 趋势
        trend = self._compute_trend(data).mean().item()
        if abs(trend) > 0.01:
            characteristics['has_trend'] = True
            characteristics['trend_direction'] = 'increasing' if trend > 0 else 'decreasing'
        else:
            characteristics['has_trend'] = False
        
        # 波动性
        volatility = data.std().item()
        data_range = (data.max() - data.min()).item()
        if volatility > 0.5 * data_range:
            characteristics['volatility'] = 'high'
        elif volatility > 0.2 * data_range:
            characteristics['volatility'] = 'medium'
        else:
            characteristics['volatility'] = 'low'
        
        # 周期性（简化检测）
        autocorr_1 = self._compute_autocorr_features(data, lags=[1]).mean().item()
        if autocorr_1 > 0.5:
            characteristics['has_seasonality'] = True
        else:
            characteristics['has_seasonality'] = False
        
        return characteristics
    
    def _is_knowledge_relevant(self, knowledge_key: str, characteristics: Dict[str, Any]) -> bool:
        """判断知识是否相关"""
        # 简单的启发式规则
        if 'trend' in knowledge_key and characteristics.get('has_trend', False):
            return True
        if 'seasonal' in knowledge_key and characteristics.get('has_seasonality', False):
            return True
        if 'volatile' in knowledge_key and characteristics.get('volatility') == 'high':
            return True
        
        return False
    
    def _generate_recommendations(self, data: torch.Tensor, task: str, 
                                  retrieval_results: Dict[str, Any]) -> List[str]:
        """生成推荐建议"""
        recommendations = []
        
        # 基于相似模式的建议
        similar_patterns = retrieval_results.get('similar_patterns', [])
        if similar_patterns and len(similar_patterns) > 0:
            num_similar = sum(len(patterns) for patterns in similar_patterns)
            recommendations.append(f"Found {num_similar} similar historical patterns")
        
        # 基于领域知识的建议
        domain_knowledge = retrieval_results.get('domain_knowledge', {})
        relevant_knowledge = domain_knowledge.get('relevant_knowledge', [])
        
        for knowledge in relevant_knowledge[:3]:  # 最多3条
            recommendations.append(f"{knowledge['topic']}: {knowledge['description']}")
        
        # 基于数据特性的建议
        characteristics = domain_knowledge.get('data_characteristics', {})
        if characteristics.get('has_trend'):
            recommendations.append("Consider using trend-aware models")
        if characteristics.get('has_seasonality'):
            recommendations.append("Seasonal decomposition may improve performance")
        if characteristics.get('volatility') == 'high':
            recommendations.append("High volatility detected, ensemble methods recommended")
        
        return recommendations
    
    def add_to_knowledge_base(self, patterns: torch.Tensor, metadata: List[Dict]):
        """
        向知识库添加新模式
        
        Args:
            patterns: [num_patterns, seq_len, features]
            metadata: List of metadata dicts
        """
        # 提取特征
        features = self._extract_pattern_features(patterns)
        
        # 添加到知识库
        if self.knowledge_base['embeddings'] is None:
            self.knowledge_base['embeddings'] = features
        else:
            self.knowledge_base['embeddings'] = torch.cat([
                self.knowledge_base['embeddings'], features
            ], dim=0)
        
        self.knowledge_base['patterns'].extend(patterns.cpu().numpy())
        self.knowledge_base['metadata'].extend(metadata)
        
        self.log_info(f"Added {len(metadata)} patterns to knowledge base")

