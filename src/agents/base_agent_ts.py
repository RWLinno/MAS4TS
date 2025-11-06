from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class AgentMessage:
    sender: str
    receiver: str
    message_type: str
    content: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class AgentOutput:
    agent_name: str
    success: bool
    result: Any
    confidence: float = 1.0
    metadata: Optional[Dict[str, Any]] = None
    messages: Optional[List[AgentMessage]] = None


class BaseAgentTS(ABC):
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.device = config.get('device', 'cpu')
        self.messages_queue = []
        logger.info(f"Agent '{self.name}' initialized")
    
    @abstractmethod
    async def process(self, input_data: Dict[str, Any]) -> AgentOutput:
        pass
    
    def send_message(self, receiver: str, message_type: str, content: Dict[str, Any]) -> AgentMessage:
        return AgentMessage(sender=self.name, receiver=receiver, message_type=message_type, content=content)
    
    def receive_message(self, message: AgentMessage):
        self.messages_queue.append(message)
    
    def get_pending_messages(self) -> List[AgentMessage]:
        messages = self.messages_queue.copy()
        self.messages_queue.clear()
        return messages
    
    def log_info(self, message: str):
        logger.info(f"[{self.name}] {message}")
    
    def log_error(self, message: str):
        logger.error(f"[{self.name}] {message}")
    
    def validate_input(self, input_data: Dict[str, Any], required_keys: List[str]) -> bool:
        missing_keys = [key for key in required_keys if key not in input_data]
        if missing_keys:
            self.log_error(f"Missing required keys: {missing_keys}")
            return False
        return True
