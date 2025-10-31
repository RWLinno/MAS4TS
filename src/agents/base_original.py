from typing import Dict, List, Optional, Any, Union, Type, Set
from pydantic import BaseModel, Field
import logging
import asyncio

logger = logging.getLogger(__name__)

class AgentConfig(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    version: str = "1.0.0"
    max_retries: int = 3
    timeout: float = 30.0
    extra_params: Dict[str, Any] = Field(default_factory=dict)

class AgentInput(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)

class AgentOutput(BaseModel):
    result: Any
    confidence: float = 0.0
    context: Dict[str, Any] = Field(default_factory=dict)

class Agent:
    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        if not self.config.name:
            self.config.name = self.__class__.__name__
        self._dependencies: Set[str] = set()
    
    def add_dependency(self, agent_name: str) -> None:
        self._dependencies.add(agent_name)
        logger.debug(f"智能体 {self.config.name} 添加依赖: {agent_name}")
    
    def get_dependencies(self) -> Set[str]:
        return self._dependencies
    
    async def execute(self, input_data: AgentInput) -> AgentOutput:
        raise NotImplementedError("Agent subclasses must implement execute method")
    
    async def run(self, input_data: AgentInput, retries: int = None) -> AgentOutput:
        max_retries = retries if retries is not None else self.config.max_retries
        attempt = 0
        
        while attempt <= max_retries:
            try:
                # 设置超时
                return await asyncio.wait_for(
                    self.execute(input_data),
                    timeout=self.config.timeout
                )
            except asyncio.TimeoutError:
                attempt += 1
                if attempt <= max_retries:
                    logger.warning(f"智能体 {self.config.name} 执行超时，进行第 {attempt} 次重试")
                else:
                    logger.error(f"智能体 {self.config.name} 执行超时，已达到最大重试次数")
                    return AgentOutput(
                        result="执行超时",
                        confidence=0.0,
                        context={"error": "timeout"}
                    )
            except Exception as e:
                attempt += 1
                if attempt <= max_retries:
                    logger.warning(f"智能体 {self.config.name} 执行出错: {str(e)}，进行第 {attempt} 次重试")
                else:
                    logger.error(f"智能体 {self.config.name} 执行出错: {str(e)}，已达到最大重试次数")
                    return AgentOutput(
                        result=f"执行出错: {str(e)}",
                        confidence=0.0,
                        context={"error": str(e)}
                    )

class AgentRegistry:
    """智能体注册中心"""
    
    _agents: Dict[str, Type[Agent]] = {}
    
    @classmethod
    def register(cls, name: Optional[str] = None):
        """
        注册智能体的装饰器
        
        Args:
            name: 智能体名称，None表示使用类名
        
        Returns:
            装饰器函数
        """
        def decorator(agent_class: Type[Agent]):
            agent_name = name or agent_class.__name__
            cls._agents[agent_name] = agent_class
            logger.info(f"注册智能体: {agent_name}")
            return agent_class
        
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Optional[Type[Agent]]:
        """
        获取智能体类
        
        Args:
            name: 智能体名称
        
        Returns:
            智能体类，如果不存在则返回None
        """
        return cls._agents.get(name)
    
    @classmethod
    def create(cls, name: str, config: Optional[AgentConfig] = None) -> Optional[Agent]:
        """
        创建智能体实例
        
        Args:
            name: 智能体名称
            config: 智能体配置
        
        Returns:
            智能体实例，如果不存在则返回None
        """
        agent_class = cls.get(name)
        if agent_class:
            return agent_class(config)
        return None
    
    @classmethod
    def list_agents(cls) -> List[str]:
        """
        列出所有已注册的智能体
        
        Returns:
            智能体名称列表
        """
        return list(cls._agents.keys())

# ===== 兼容旧架构的支持 =====

class BaseAgent:
    """
    旧架构智能体基类（兼容性支持）
    """
    
    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.status = "idle"
        self.capabilities = []
    
    async def initialize(self) -> None:
        """初始化代理"""
        self.status = "ready"
        logger.info(f"代理 {self.name} 初始化完成")
    
    async def _process(self, input_data: AgentInput) -> AgentOutput:
        """处理逻辑（子类需要实现）"""
        raise NotImplementedError("BaseAgent subclasses must implement _process method")
    
    async def process(self, input_data: AgentInput) -> AgentOutput:
        """处理请求"""
        if self.status != "ready":
            await self.initialize()
        
        self.status = "processing"
        try:
            result = await self._process(input_data)
            self.status = "ready"
            return result
        except Exception as e:
            self.status = "error"
            logger.exception(f"代理 {self.name} 处理失败: {e}")
            return AgentOutput(
                result=f"处理失败: {str(e)}",
                confidence=0.0,
                context={"error": str(e)}
            )

# 全局智能体注册表（兼容旧架构）
_legacy_agents: Dict[str, BaseAgent] = {}

def register_agent(agent: BaseAgent) -> None:
    """
    注册智能体（兼容旧架构）
    
    Args:
        agent: 智能体实例
    """
    _legacy_agents[agent.name] = agent
    logger.info(f"注册旧架构智能体: {agent.name}")

def get_agent(name: str) -> Optional[BaseAgent]:
    """
    获取智能体实例（兼容旧架构）
    
    Args:
        name: 智能体名称
    
    Returns:
        智能体实例，如果不存在则返回None
    """
    return _legacy_agents.get(name)

def list_legacy_agents() -> List[str]:
    """
    列出所有已注册的旧架构智能体
    
    Returns:
        智能体名称列表
    """
    return list(_legacy_agents.keys())
