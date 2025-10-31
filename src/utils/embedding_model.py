"""
嵌入模型模块
提供文本向量化功能，支持使用不同的模型获取文本嵌入
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union, Type
import numpy as np

# 配置日志
logger = logging.getLogger(__name__)

class EmbeddingModel:
    """
    嵌入模型基类
    提供文本向量化的基本接口
    """
    
    def __init__(self, model_name: str = "", **kwargs):
        """
        初始化嵌入模型
        
        Args:
            model_name: 模型名称
            **kwargs: 额外参数
        """
        self.model_name = model_name
        self.embedding_dimension = 0  # 子类需要设置实际的向量维度
    
    def encode(self, text: str) -> np.ndarray:
        """
        将单个文本编码为向量
        
        Args:
            text: 要编码的文本
            
        Returns:
            文本的向量表示
        """
        raise NotImplementedError("子类必须实现encode方法")
    
    def encode_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        将多个文本编码为向量
        
        Args:
            texts: 要编码的文本列表
            
        Returns:
            文本向量的列表
        """
        return [self.encode(text) for text in texts]
    
    def encode_documents(self, documents: List[str]) -> List[np.ndarray]:
        """
        为文档编码，可能与查询编码方式不同
        
        Args:
            documents: 要编码的文档列表
            
        Returns:
            文档向量的列表
        """
        return self.encode_batch(documents)
    
    def encode_query(self, query: str) -> np.ndarray:
        """
        为查询编码，可能与文档编码方式不同
        
        Args:
            query: 要编码的查询文本
            
        Returns:
            查询的向量表示
        """
        return self.encode(query)

class SentenceTransformerEmbeddings(EmbeddingModel):
    """
    使用sentence-transformers进行文本向量化
    支持各种预训练的文本嵌入模型
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        **kwargs
    ):
        """
        初始化SentenceTransformer嵌入模型
        
        Args:
            model_name: 模型名称或路径
            device: 使用的设备，如'cpu'或'cuda'
            **kwargs: 传递给SentenceTransformer的额外参数
        """
        super().__init__(model_name=model_name, **kwargs)
        
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "使用SentenceTransformerEmbeddings需要安装sentence-transformers库，"
                "请使用 pip install sentence-transformers"
            )
        
        self.device = device
        logger.info(f"正在加载sentence-transformer模型: {model_name}")
        
        model_kwargs = kwargs.copy()
        if device is not None:
            model_kwargs["device"] = device
        
        self.model = SentenceTransformer(model_name, **model_kwargs)
        self.embedding_dimension = self.model.get_sentence_embedding_dimension()
        
        logger.info(f"模型加载完成，嵌入维度: {self.embedding_dimension}")
    
    def encode(self, text: str) -> np.ndarray:
        """
        将单个文本编码为向量
        
        Args:
            text: 要编码的文本
            
        Returns:
            文本的向量表示
        """
        return self.model.encode(text, convert_to_numpy=True)
    
    def encode_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        将多个文本编码为向量
        
        Args:
            texts: 要编码的文本列表
            
        Returns:
            文本向量的列表
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return [embedding for embedding in embeddings]

class OpenAIEmbeddings(EmbeddingModel):
    """
    使用OpenAI API进行文本向量化
    支持各种OpenAI嵌入模型
    """
    
    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        初始化OpenAI嵌入模型
        
        Args:
            model_name: 模型名称，如'text-embedding-3-small'或'text-embedding-3-large'
            api_key: OpenAI API密钥，如果为None则从环境变量获取
            **kwargs: 额外参数
        """
        super().__init__(model_name=model_name, **kwargs)
        
        try:
            import openai
        except ImportError:
            raise ImportError(
                "使用OpenAIEmbeddings需要安装openai库，"
                "请使用 pip install openai"
            )
        
        self.openai = openai
        
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key is None:
                raise ValueError(
                    "未提供OpenAI API密钥，请通过参数提供或设置OPENAI_API_KEY环境变量"
                )
        
        self.client = openai.OpenAI(api_key=api_key)
        
        # 根据模型设置维度
        dimension_map = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        
        self.embedding_dimension = dimension_map.get(model_name, 1536)
        self.kwargs = kwargs
        
        logger.info(f"初始化OpenAI嵌入模型: {model_name}，嵌入维度: {self.embedding_dimension}")
    
    def encode(self, text: str) -> np.ndarray:
        """
        将单个文本编码为向量
        
        Args:
            text: 要编码的文本
            
        Returns:
            文本的向量表示
        """
        if not text.strip():
            return np.zeros(self.embedding_dimension)
        
        response = self.client.embeddings.create(
            input=[text],
            model=self.model_name,
            **self.kwargs
        )
        
        embedding = response.data[0].embedding
        return np.array(embedding)
    
    def encode_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        将多个文本编码为向量
        
        Args:
            texts: 要编码的文本列表
            
        Returns:
            文本向量的列表
        """
        # 过滤空文本
        valid_texts = []
        empty_indices = []
        
        for i, text in enumerate(texts):
            if text.strip():
                valid_texts.append(text)
            else:
                empty_indices.append(i)
        
        if not valid_texts:
            return [np.zeros(self.embedding_dimension) for _ in texts]
        
        # 获取有效文本的嵌入
        response = self.client.embeddings.create(
            input=valid_texts,
            model=self.model_name,
            **self.kwargs
        )
        
        embeddings = [np.array(data.embedding) for data in response.data]
        
        # 为空文本插入零向量
        for idx in empty_indices:
            embeddings.insert(idx, np.zeros(self.embedding_dimension))
        
        return embeddings

class HuggingFaceEmbeddings(EmbeddingModel):
    """
    使用HuggingFace模型进行文本向量化
    支持各种Transformer模型
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        device: Optional[str] = None,
        **kwargs
    ):
        """
        初始化HuggingFace嵌入模型
        
        Args:
            model_name: 模型名称或路径
            device: 使用的设备，如'cpu'或'cuda'
            **kwargs: 传递给AutoModel的额外参数
        """
        super().__init__(model_name=model_name, **kwargs)
        
        try:
            import torch
            from transformers import AutoTokenizer, AutoModel
        except ImportError:
            raise ImportError(
                "使用HuggingFaceEmbeddings需要安装torch和transformers库，"
                "请使用 pip install torch transformers"
            )
        
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"正在加载HuggingFace模型: {model_name}，设备: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, **kwargs)
        self.model.to(self.device)
        
        # 获取模型隐藏层维度
        self.embedding_dimension = self.model.config.hidden_size
        logger.info(f"模型加载完成，嵌入维度: {self.embedding_dimension}")
    
    def encode(self, text: str) -> np.ndarray:
        """
        将单个文本编码为向量
        
        Args:
            text: 要编码的文本
            
        Returns:
            文本的向量表示
        """
        # 设置为评估模式
        self.model.eval()
        
        # 对输入进行编码
        inputs = self.tokenizer(
            text,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        # 将张量移动到指定设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 进行推理
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        
        # 获取[CLS]令牌的嵌入作为句子表示
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embeddings[0]
    
    def encode_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        将多个文本编码为向量
        
        Args:
            texts: 要编码的文本列表
            
        Returns:
            文本向量的列表
        """
        # 设置为评估模式
        self.model.eval()
        
        # 对输入批量进行编码
        inputs = self.tokenizer(
            texts,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        # 将张量移动到指定设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 进行推理
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        
        # 获取[CLS]令牌的嵌入作为句子表示
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return [embedding for embedding in embeddings]

# 注册可用的嵌入模型
_EMBEDDING_MODELS: Dict[str, Type[EmbeddingModel]] = {
    "sentencetransformer": SentenceTransformerEmbeddings,
    "sentence-transformer": SentenceTransformerEmbeddings,
    "openai": OpenAIEmbeddings,
    "huggingface": HuggingFaceEmbeddings,
}

def get_embedding_model(model_type: str, **kwargs) -> EmbeddingModel:
    """
    获取指定类型的嵌入模型
    
    Args:
        model_type: 模型类型名称
        **kwargs: 传递给模型构造函数的参数
    
    Returns:
        嵌入模型实例
    
    Raises:
        ValueError: 如果指定的模型类型不存在
    """
    model_type = model_type.lower()
    
    if model_type not in _EMBEDDING_MODELS:
        raise ValueError(
            f"未知的嵌入模型类型: {model_type}, 可用类型: {list(_EMBEDDING_MODELS.keys())}"
        )
    
    model_class = _EMBEDDING_MODELS[model_type]
    return model_class(**kwargs) 