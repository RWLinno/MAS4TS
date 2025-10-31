#!/usr/bin/env python3
"""
Unified Base Agent Architecture
Consolidated interface for all OnCallAgent specialized agents
Combines the best features from base.py and base_agent.py
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Type, Union
from pydantic import BaseModel, Field
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

# ===== Enums and Status =====

class AgentStatus(Enum):
    """Agent execution status"""
    IDLE = "idle"
    PROCESSING = "processing"
    ERROR = "error"
    DISABLED = "disabled"

# ===== Configuration Models =====

class AgentConfig(BaseModel):
    """Unified configuration for all agents"""
    name: Optional[str] = None
    description: Optional[str] = None
    version: str = "1.0.0"
    enabled: bool = True
    
    # Model configuration
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    max_tokens: int = 512
    temperature: float = 0.7
    
    # Performance configuration
    confidence_threshold: float = 0.8
    timeout_seconds: int = 30
    retry_attempts: int = 2
    max_retries: int = 3
    timeout: float = 30.0
    
    # Agent-specific configurations (unified from both systems)
    extra_params: Dict[str, Any] = Field(default_factory=dict)
    specialized_config: Dict[str, Any] = Field(default_factory=dict)

# ===== Input/Output Models =====

@dataclass
class AgentInput:
    """Standardized input for all agents"""
    query: str
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Multi-modal inputs
    image_path: Optional[str] = None
    document_references: List[str] = field(default_factory=list)
    
    # Execution context
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: Optional[float] = None

@dataclass
class AgentOutput:
    """Standardized output from all agents"""
    # Unified response fields (supports both 'result' and 'response' for compatibility)
    result: Any = None
    response: str = ""
    confidence: float = 0.0
    
    # Execution metadata
    agent_name: str = ""
    processing_time: float = 0.0
    status: AgentStatus = AgentStatus.IDLE
    
    # Additional context and results
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Multi-agent coordination
    suggested_next_agents: List[str] = field(default_factory=list)
    coordination_needed: bool = False
    
    def __post_init__(self):
        """Ensure compatibility between result and response fields"""
        if self.result and not self.response:
            self.response = str(self.result)
        elif self.response and not self.result:
            self.result = self.response

# ===== Base Agent Classes =====

class BaseAgent(ABC):
    """
    Unified abstract base class for all OnCallAgent specialized agents
    Combines functionality from both base.py and base_agent.py
    """
    
    def __init__(self, config: Union[AgentConfig, Dict[str, Any]], global_config: Optional[Dict[str, Any]] = None):
        # Handle both AgentConfig objects and dict configurations
        if isinstance(config, dict):
            self.config = AgentConfig(**config)
        else:
            self.config = config
            
        # Set default name if not provided
        if not self.config.name:
            self.config.name = self.__class__.__name__
            
        # Store global configuration
        self.global_config = global_config or {}
        
        # Agent state
        self.status = AgentStatus.IDLE
        self.dependencies: Set[str] = set()
        
        # Performance metrics
        self.total_executions = 0
        self.total_processing_time = 0.0
        self.success_count = 0
        self.error_count = 0
        
        # Initialize agent-specific components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize agent-specific components (override in subclasses)"""
        pass
    
    @abstractmethod
    async def _process_query(self, input_data: AgentInput) -> AgentOutput:
        """
        Core query processing logic (must be implemented by subclasses)
        
        Args:
            input_data: Standardized agent input
            
        Returns:
            AgentOutput with response and metadata
        """
        pass
    
    async def execute(self, input_data: AgentInput) -> AgentOutput:
        """
        Main execution method with error handling, retries, and performance tracking
        Unified implementation from both base architectures
        """
        if not self.config.enabled:
            return AgentOutput(
                response="Agent is disabled",
                result="Agent is disabled",
                confidence=0.0,
                agent_name=self.config.name,
                status=AgentStatus.DISABLED
            )
        
        start_time = time.time()
        self.status = AgentStatus.PROCESSING
        self.total_executions += 1
        
        # Validate input
        if not input_data.query.strip():
            return self._create_error_output("Empty query provided", start_time)
        
        # Execute with retry logic
        max_retries = max(self.config.retry_attempts, self.config.max_retries)
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                # Apply timeout
                result = await asyncio.wait_for(
                    self._process_query(input_data),
                    timeout=max(self.config.timeout_seconds, self.config.timeout)
                )
                
                # Update performance metrics
                processing_time = time.time() - start_time
                self.total_processing_time += processing_time
                self.success_count += 1
                self.status = AgentStatus.IDLE
                
                # Enhance result with execution metadata
                result.agent_name = self.config.name
                result.processing_time = processing_time
                result.status = AgentStatus.IDLE
                result.metadata.update({
                    "execution_attempt": attempt + 1,
                    "total_executions": self.total_executions,
                    "success_rate": self.success_count / self.total_executions
                })
                
                logger.debug(f"✓ Agent {self.config.name} completed in {processing_time:.2f}s")
                return result
            
            except asyncio.TimeoutError:
                last_exception = f"Agent execution timeout after {max(self.config.timeout_seconds, self.config.timeout)}s"
                logger.warning(f"⏱️ Agent {self.config.name} timeout on attempt {attempt + 1}")
            
            except Exception as e:
                last_exception = str(e)
                logger.warning(f"❌ Agent {self.config.name} error on attempt {attempt + 1}: {e}")
            
            # Wait before retry (exponential backoff)
            if attempt < max_retries:
                await asyncio.sleep(0.5 * (2 ** attempt))
        
        # All attempts failed
        self.error_count += 1
        return self._create_error_output(f"All attempts failed. Last error: {last_exception}", start_time)
    
    def _create_error_output(self, error_message: str, start_time: float) -> AgentOutput:
        """Create standardized error output"""
        self.status = AgentStatus.ERROR
        processing_time = time.time() - start_time
        
        return AgentOutput(
            response=f"Agent {self.config.name} failed: {error_message}",
            result=f"Agent {self.config.name} failed: {error_message}",
            confidence=0.0,
            agent_name=self.config.name,
            processing_time=processing_time,
            status=AgentStatus.ERROR,
            metadata={"error": error_message}
        )
    
    def add_dependency(self, agent_name: str) -> None:
        """Add dependency on another agent"""
        self.dependencies.add(agent_name)
        logger.debug(f"Agent {self.config.name} added dependency: {agent_name}")
    
    def get_dependencies(self) -> Set[str]:
        """Get agent dependencies"""
        return self.dependencies
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics"""
        avg_processing_time = (
            self.total_processing_time / self.total_executions 
            if self.total_executions > 0 else 0.0
        )
        
        success_rate = (
            self.success_count / self.total_executions 
            if self.total_executions > 0 else 0.0
        )
        
        return {
            "agent_name": self.config.name,
            "total_executions": self.total_executions,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "success_rate": success_rate,
            "average_processing_time": avg_processing_time,
            "total_processing_time": self.total_processing_time,
            "current_status": self.status.value
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform agent health check"""
        try:
            test_input = AgentInput(
                query="health check",
                context={"health_check": True},
                metadata={"test": True}
            )
            
            start_time = time.time()
            result = await asyncio.wait_for(
                self._process_query(test_input),
                timeout=5.0
            )
            
            return {
                "agent": self.config.name,
                "status": "healthy",
                "response_time": time.time() - start_time,
                "confidence": result.confidence
            }
        
        except Exception as e:
            return {
                "agent": self.config.name,
                "status": "unhealthy",
                "error": str(e)
            }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.config.name}, enabled={self.config.enabled}, status={self.status.value})"

# ===== Legacy Compatibility Layer =====

class Agent(BaseAgent):
    """
    Legacy compatibility class for old base.py architecture
    Provides backward compatibility for existing agents
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        # Convert old-style initialization to new unified format
        if config is None:
            config = AgentConfig()
        super().__init__(config)
    
    async def _process_query(self, input_data: AgentInput) -> AgentOutput:
        """Delegate to the old execute method for compatibility"""
        # This will be overridden by subclasses that implement the old interface
        raise NotImplementedError("Agent subclasses must implement _process_query method")

# ===== Agent Registry =====

class AgentRegistry:
    """
    Unified registry for agent classes, enabling dynamic discovery and instantiation
    Combines functionality from both registry systems
    """
    
    _registry: Dict[str, Type[BaseAgent]] = {}
    _legacy_agents: Dict[str, BaseAgent] = {}
    
    @classmethod
    def register(cls, name: Optional[str] = None):
        """Decorator to register agent classes"""
        def decorator(agent_class: Type[BaseAgent]):
            agent_name = name or agent_class.__name__.lower()
            cls._registry[agent_name] = agent_class
            logger.debug(f"Registered agent class: {agent_name}")
            return agent_class
        return decorator
    
    @classmethod
    def get_agent_class(cls, name: str) -> Optional[Type[BaseAgent]]:
        """Get agent class by name"""
        return cls._registry.get(name)
    
    @classmethod
    def list_registered_agents(cls) -> List[str]:
        """List all registered agent names"""
        return list(cls._registry.keys())
    
    @classmethod
    def create_agent(cls, name: str, config: Union[AgentConfig, Dict[str, Any]], 
                    global_config: Optional[Dict[str, Any]] = None) -> Optional[BaseAgent]:
        """Create agent instance by name"""
        agent_class = cls.get_agent_class(name)
        if agent_class:
            return agent_class(config, global_config)
        return None
    
    # Legacy compatibility methods
    @classmethod
    def get(cls, name: str) -> Optional[Type[BaseAgent]]:
        """Legacy method for getting agent class"""
        return cls.get_agent_class(name)
    
    @classmethod
    def create(cls, name: str, config: Optional[AgentConfig] = None) -> Optional[BaseAgent]:
        """Legacy method for creating agent instance"""
        return cls.create_agent(name, config or AgentConfig())
    
    @classmethod
    def list_agents(cls) -> List[str]:
        """Legacy method for listing agents"""
        return cls.list_registered_agents()

# ===== Legacy Support Functions =====

def register_agent(agent: BaseAgent) -> None:
    """
    Register agent instance (legacy compatibility)
    
    Args:
        agent: Agent instance
    """
    AgentRegistry._legacy_agents[agent.config.name] = agent
    logger.info(f"Registered legacy agent: {agent.config.name}")

def get_agent(name: str) -> Optional[BaseAgent]:
    """
    Get agent instance (legacy compatibility)
    
    Args:
        name: Agent name
    
    Returns:
        Agent instance, if exists
    """
    return AgentRegistry._legacy_agents.get(name)

def list_legacy_agents() -> List[str]:
    """
    List all registered legacy agents
    
    Returns:
        Agent name list
    """
    return list(AgentRegistry._legacy_agents.keys())

# ===== Type Aliases for Backward Compatibility =====

# Support both naming conventions
BaseAgentInput = AgentInput
BaseAgentOutput = AgentOutput
BaseAgentConfig = AgentConfig

# Export all public classes and functions
__all__ = [
    'AgentStatus',
    'AgentConfig', 'BaseAgentConfig',
    'AgentInput', 'BaseAgentInput', 
    'AgentOutput', 'BaseAgentOutput',
    'BaseAgent', 'Agent',
    'AgentRegistry',
    'register_agent', 'get_agent', 'list_legacy_agents'
]