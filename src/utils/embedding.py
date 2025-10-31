from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import os
import json

logger = logging.getLogger(__name__)

class EmbeddingConfig(BaseModel):
    """嵌入配置"""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 32
    max_length: int = 512
    normalize: bool = True
    pooling: str = "mean"  # mean, max, cls

class Embedding:
    """嵌入模块"""
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self.model = None
        self._initialize()
    
    def _initialize(self) -> None:
        """初始化"""
        # 加载模型
        self.model = SentenceTransformer(
            self.config.model_name,
            device=self.config.device
        )
        
        logger.info(f"嵌入模块初始化完成，使用设备: {self.config.device}")
    
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: Optional[int] = None,
        show_progress_bar: bool = False
    ) -> np.ndarray:
        """编码文本"""
        if isinstance(texts, str):
            texts = [texts]
        
        # 使用配置的批处理大小或指定的批处理大小
        batch_size = batch_size or self.config.batch_size
        
        # 编码文本
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True
        )
        
        # 归一化
        if self.config.normalize:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        return embeddings
    
    def encode_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress_bar: bool = False
    ) -> List[np.ndarray]:
        """批量编码文本"""
        embeddings = self.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar
        )
        return [emb for emb in embeddings]
    
    def similarity(
        self,
        text1: str,
        text2: str,
        batch_size: Optional[int] = None
    ) -> float:
        """计算文本相似度"""
        # 编码文本
        emb1 = self.encode(text1, batch_size=batch_size)
        emb2 = self.encode(text2, batch_size=batch_size)
        
        # 计算余弦相似度
        similarity = np.dot(emb1, emb2.T)
        
        return float(similarity)
    
    def similarity_batch(
        self,
        texts1: List[str],
        texts2: List[str],
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """批量计算文本相似度"""
        # 编码文本
        emb1 = self.encode(texts1, batch_size=batch_size)
        emb2 = self.encode(texts2, batch_size=batch_size)
        
        # 计算余弦相似度矩阵
        similarity_matrix = np.dot(emb1, emb2.T)
        
        return similarity_matrix
    
    def most_similar(
        self,
        query: str,
        texts: List[str],
        top_k: int = 5,
        batch_size: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """查找最相似的文本"""
        # 计算相似度矩阵
        similarity_matrix = self.similarity_batch([query], texts, batch_size)
        
        # 获取最相似的结果
        top_indices = np.argsort(similarity_matrix[0])[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                "text": texts[idx],
                "similarity": float(similarity_matrix[0][idx])
            })
        
        return results
    
    def save_model(self, path: str) -> None:
        """保存模型"""
        os.makedirs(path, exist_ok=True)
        
        # 保存模型
        self.model.save(path)
        
        # 保存配置
        config_path = os.path.join(path, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.config.dict(), f, ensure_ascii=False, indent=2)
        
        logger.info(f"模型保存到: {path}")
    
    @classmethod
    def load_model(cls, path: str) -> "Embedding":
        """加载模型"""
        # 加载配置
        config_path = os.path.join(path, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config = EmbeddingConfig(**json.load(f))
        
        # 创建实例
        embedding = cls(config)
        
        # 加载模型
        embedding.model = SentenceTransformer(path, device=config.device)
        
        logger.info(f"从 {path} 加载模型")
        return embedding

# 创建全局嵌入实例
embedding = Embedding() 