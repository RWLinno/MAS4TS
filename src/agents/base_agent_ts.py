"""
Base Agent for Time Series Analysis
所有时序分析agents的基类
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class AgentMessage:
    """Agent间通信的消息格式"""
    sender: str  # 发送者agent名称
    receiver: str  # 接收者agent名称
    message_type: str  # 消息类型: 'request', 'response', 'notification'
    content: Dict[str, Any]  # 消息内容
    metadata: Optional[Dict[str, Any]] = None  # 元数据


@dataclass
class AgentOutput:
    """Agent的输出结果"""
    agent_name: str
    success: bool
    result: Any
    confidence: float = 1.0
    metadata: Optional[Dict[str, Any]] = None
    messages: Optional[List[AgentMessage]] = None  # 发送给其他agents的消息


class BaseAgentTS(ABC):
    """时序分析Agent基类"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.device = config.get('device', 'cpu')
        self.messages_queue = []  # 消息队列
        
        logger.info(f"Agent '{self.name}' initialized")
    
    @abstractmethod
    async def process(self, input_data: Dict[str, Any]) -> AgentOutput:
        """
        处理输入数据并返回结果
        
        Args:
            input_data: 输入数据，包含任务相关信息
            
        Returns:
            AgentOutput: 处理结果
        """
        pass
    
    def send_message(self, receiver: str, message_type: str, content: Dict[str, Any]) -> AgentMessage:
        """
        发送消息给其他agent
        
        Args:
            receiver: 接收者agent名称
            message_type: 消息类型
            content: 消息内容
            
        Returns:
            AgentMessage: 创建的消息
        """
        message = AgentMessage(
            sender=self.name,
            receiver=receiver,
            message_type=message_type,
            content=content
        )
        return message
    
    def receive_message(self, message: AgentMessage):
        """
        接收来自其他agent的消息
        
        Args:
            message: 接收的消息
        """
        self.messages_queue.append(message)
        logger.debug(f"Agent '{self.name}' received message from '{message.sender}'")
    
    def get_pending_messages(self) -> List[AgentMessage]:
        """获取待处理的消息"""
        messages = self.messages_queue.copy()
        self.messages_queue.clear()
        return messages
    
    def log_info(self, message: str):
        """记录信息日志"""
        logger.info(f"[{self.name}] {message}")
    
    def log_error(self, message: str):
        """记录错误日志"""
        logger.error(f"[{self.name}] {message}")
    
    def validate_input(self, input_data: Dict[str, Any], required_keys: List[str]) -> bool:
        """
        验证输入数据是否包含必需的键
        
        Args:
            input_data: 输入数据
            required_keys: 必需的键列表
            
        Returns:
            bool: 验证是否通过
        """
        missing_keys = [key for key in required_keys if key not in input_data]
        if missing_keys:
            self.log_error(f"Missing required keys: {missing_keys}")
            return False
        return True

