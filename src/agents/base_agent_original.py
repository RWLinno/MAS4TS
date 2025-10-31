#!/usr/bin/env python3
"""
Base Agent Architecture
Standardized interface for all OnCallAgent specialized agents
Enhanced with Eigent-inspired design patterns
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

class AgentStatus(Enum):
    """Agent execution status"""
    IDLE = "idle"
    PROCESSING = "processing"
    ERROR = "error"
    DISABLED = "disabled"

@dataclass
class AgentConfig:
    """Configuration for individual agents"""
    name: str
    description: str
    enabled: bool = True
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    max_tokens: int = 512
    temperature: float = 0.7
    confidence_threshold: float = 0.8
    timeout_seconds: int = 30
    retry_attempts: int = 2
    
    # Agent-specific configurations
    specialized_config: Dict[str, Any] = field(default_factory=dict)

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
    response: str
    confidence: float
    
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

class BaseAgent(ABC):
    """
    Abstract base class for all OnCallAgent specialized agents
    Provides common functionality and standardized interface
    """
    
    def __init__(self, config: AgentConfig, global_config: Dict[str, Any]):
        self.config = config
        self.global_config = global_config
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
        
        Args:
            input_data: Standardized agent input
            
        Returns:
            AgentOutput with response and execution metadata
        """
        if not self.config.enabled:
            return AgentOutput(
                response="Agent is disabled",
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
        last_exception = None
        for attempt in range(self.config.retry_attempts + 1):
            try:
                # Apply timeout
                result = await asyncio.wait_for(
                    self._process_query(input_data),
                    timeout=self.config.timeout_seconds
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
                last_exception = f"Agent execution timeout after {self.config.timeout_seconds}s"
                logger.warning(f"⏱️ Agent {self.config.name} timeout on attempt {attempt + 1}")
            
            except Exception as e:
                last_exception = str(e)
                logger.warning(f"❌ Agent {self.config.name} error on attempt {attempt + 1}: {e}")
            
            # Wait before retry (exponential backoff)
            if attempt < self.config.retry_attempts:
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
            confidence=0.0,
            agent_name=self.config.name,
            processing_time=processing_time,
            status=AgentStatus.ERROR,
            metadata={"error": error_message}
        )
    
    def add_dependency(self, agent_name: str):
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

class AgentRegistry:
    """
    Registry for agent classes, enabling dynamic discovery and instantiation
    Inspired by Eigent's agent management pattern
    """
    
    _registry: Dict[str, type[BaseAgent]] = {}
    
    @classmethod
    def register(cls, name: Optional[str] = None):
        """Decorator to register agent classes"""
        def decorator(agent_class: type[BaseAgent]):
            agent_name = name or agent_class.__name__.lower()
            cls._registry[agent_name] = agent_class
            logger.debug(f"Registered agent class: {agent_name}")
            return agent_class
        return decorator
    
    @classmethod
    def get_agent_class(cls, name: str) -> Optional[type[BaseAgent]]:
        """Get agent class by name"""
        return cls._registry.get(name)
    
    @classmethod
    def list_registered_agents(cls) -> List[str]:
        """List all registered agent names"""
        return list(cls._registry.keys())
    
    @classmethod
    def create_agent(cls, name: str, config: AgentConfig, 
                    global_config: Dict[str, Any]) -> Optional[BaseAgent]:
        """Create agent instance by name"""
        agent_class = cls.get_agent_class(name)
        if agent_class:
            return agent_class(config, global_config)
        return None
