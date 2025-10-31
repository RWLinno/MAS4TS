import os
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class FeatureConfig(BaseModel):
    use_rag: bool = True
    use_visual_analysis: bool = True
    use_metrics_analysis: bool = True
    use_log_analysis: bool = True
    use_knowledge_base: bool = True
    use_comprehensive_agent: bool = True

class AgentConfig(BaseModel):
    enabled: bool = True
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    max_tokens: int = 512
    temperature: float = 0.7
    confidence_threshold: float = 0.7

class RAGConfig(BaseModel):
    enabled: bool = True
    retrieval: Dict[str, Any] = Field(default_factory=dict)
    embedding: Dict[str, Any] = Field(default_factory=dict)
    knowledge_base: Dict[str, Any] = Field(default_factory=dict)
    document_processing: Dict[str, Any] = Field(default_factory=dict)
    multimodal: Dict[str, Any] = Field(default_factory=dict)

class OnCallAgentConfig(BaseModel):
    """OnCallAgent主配置"""
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    agents: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    api_keys: Dict[str, Any] = Field(default_factory=dict)
    directories: Dict[str, str] = Field(default_factory=dict)
    models: Dict[str, Any] = Field(default_factory=dict)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    privacy: Dict[str, Any] = Field(default_factory=dict)
    logging: Dict[str, Any] = Field(default_factory=dict)
    api_service: Dict[str, Any] = Field(default_factory=dict)
    web_app: Dict[str, Any] = Field(default_factory=dict)
    caching: Dict[str, Any] = Field(default_factory=dict)
    advanced: Dict[str, Any] = Field(default_factory=dict)
    
    def get(self, key: str, default: Any = None) -> Any:
        """字典式访问方法，用于向后兼容"""
        try:
            # 首先尝试作为属性访问
            if hasattr(self, key):
                value = getattr(self, key)
                # 如果是Pydantic模型，转换为字典
                if hasattr(value, 'dict'):
                    return value.dict()
                return value
            
            # 尝试从字典表示中获取
            config_dict = self.dict()
            return config_dict.get(key, default)
        except Exception:
            return default

def load_config(config_path: Optional[str] = None) -> OnCallAgentConfig:
    if config_path is None:
        # 计算项目根目录路径
        project_root = Path(__file__).parent.parent.parent
        
        # 确保项目根目录存在
        if not project_root.exists():
            raise FileNotFoundError(f"Project root directory not found: {project_root}")
        
        config_paths = [
            project_root / "config.json",
            project_root / "config.example.json"
        ]
        
        config_path = None
        for path in config_paths:
            if path.exists() and path.is_file():
                config_path = path
                logger.info(f"Found config file: {path}")
                break
        
        if config_path is None:
            logger.error(f"Config file not found in project root: {project_root}")
            logger.error(f"Searched paths: {[str(p) for p in config_paths]}")
            raise FileNotFoundError("Config file not found: config.json or config.example.json")
    else:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file does not exist: {config_path}")
        if not config_path.is_file():
            raise ValueError(f"Config path is not a file: {config_path}")
    
    logger.info(f"load config file: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"config file json format error: {e}")
    except Exception as e:
        raise Exception(f"read config file failed: {e}")
    
    try:
        config = OnCallAgentConfig(**config_data)
        logger.info("config file validate success")
        return config
    except Exception as e:
        logger.error(f"config file validate failed: {e}")
        config = OnCallAgentConfig()
        
        if "features" in config_data:
            for key, value in config_data["features"].items():
                if hasattr(config.features, key):
                    setattr(config.features, key, value)
        
        if "rag" in config_data:
            config.rag = RAGConfig(**config_data["rag"])
        
        for key in ["agents", "api_keys", "directories", "models", "privacy", 
                   "logging", "api_service", "web_app", "caching", "advanced"]:
            if key in config_data:
                setattr(config, key, config_data[key])
        
        logger.warning("use partial config, some config items that failed to validate will use default values")
        return config

class ConfigLoader:
    """配置加载器类，用于向后兼容"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config = None
    
    def load_config(self) -> OnCallAgentConfig:
        """加载配置文件"""
        if self.config is None:
            self.config = load_config(self.config_path)
        return self.config
    
    def get_config(self) -> OnCallAgentConfig:
        """获取配置对象"""
        return self.load_config()

# 为了向后兼容，保留原有的函数名（虽然不推荐使用）
def get_agent_config(config: OnCallAgentConfig, agent_name: str) -> Dict[str, Any]:
    """
    获取指定智能体的配置
    
    Args:
        config: 主配置对象
        agent_name: 智能体名称
    
    Returns:
        智能体配置字典
    """
    agent_configs = config.agents
    
    agent_key = agent_name.lower().replace("agent", "_agent")
    if agent_key not in agent_configs:
        agent_key = agent_name.lower()
    
    if agent_key in agent_configs:
        return agent_configs[agent_key]
    
    logger.warning(f"agent {agent_name} not found, use default config")
    return {
        "enabled": True,
        "model_name": config.models.get("local_models", {}).get("model_name", "Qwen/Qwen2.5-VL-7B-Instruct"),
        "max_tokens": 512,
        "temperature": 0.7,
        "confidence_threshold": 0.7
    }

def should_use_rag(config: OnCallAgentConfig) -> bool:
    """
    检查是否应该使用RAG功能
    
    Args:
        config: 配置对象
    
    Returns:
        是否使用RAG
    """
    return config.features.use_rag and config.rag.enabled

def get_rag_config(config: OnCallAgentConfig) -> Dict[str, Any]:
    """
    获取RAG配置
    
    Args:
        config: 主配置对象
    
    Returns:
        RAG配置字典
    """
    rag_config = config.rag.dict()
    
    if "models" in config.dict():
        models_config = config.models
        if "embedding_model" in models_config:
            rag_config.setdefault("embedding", {})["model_name"] = models_config["embedding_model"]
    
    return rag_config 