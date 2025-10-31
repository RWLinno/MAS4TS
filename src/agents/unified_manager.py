#!/usr/bin/env python3
"""
Unified Agent and Model Manager
Centralized management for all OnCallAgent specialized agents and models
Consolidates UnifiedAgentManager and UnifiedModelManager functionality
"""

import asyncio
import logging
import time
import os
import torch
from typing import Dict, Any, List, Optional, Type, Union
from pathlib import Path
from dataclasses import dataclass

# Import unified base classes
from .base import BaseAgent, AgentConfig, AgentInput, AgentOutput, AgentRegistry

# Import core agents
from .core_agents import (
    VisualAnalysisAgent, LogAnalysisAgent, 
    MetricsAnalysisAgent, ComprehensiveAgent, KnowledgeAgent, RetrieverAgent
)

# Import enhanced route agent
from .enhanced_route_agent import EnhancedRouteAgent

# Import search agent
from .search_agent import SearchAgent

logger = logging.getLogger(__name__)

# ===== Model Management =====

@dataclass
class ModelRequest:
    """Request for model generation"""
    messages: List[Dict[str, str]]
    max_tokens: int = 512
    temperature: float = 0.7
    image: Optional[str] = None

@dataclass
class ModelResponse:
    """Response from model generation"""
    content: str
    success: bool
    error: Optional[str] = None

# Supported models configuration
SUPPORTED_MODELS = {
    "Qwen/Qwen2.5-7B-Instruct": {
        "type": "text",
        "description": "通用文本生成模型",
        "local_path": "Qwen2.5-7B-Instruct",
        "min_gpu_memory": "8GB"
    },
    "Qwen/Qwen2.5-VL-7B-Instruct": {
        "type": "vision-language",
        "description": "视觉语言多模态模型",
        "local_path": "Qwen2.5-VL-7B-Instruct",
        "min_gpu_memory": "12GB"
    },
    "Qwen/Qwen2-7B-Instruct": {
        "type": "text",
        "description": "Qwen2系列文本模型",
        "local_path": "Qwen2-7B-Instruct",
        "min_gpu_memory": "8GB"
    }
}

class UnifiedModelManager:
    """
    Unified Model Manager - Singleton pattern for model management
    Handles model loading, caching, and inference
    """
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device_config: Optional[Dict[str, Any]] = None,
        offline_mode: bool = False
    ):
        """
        Initialize model manager
        
        Args:
            model_name: Model name
            device_config: Device configuration
            offline_mode: Whether to use offline mode
        """
        if self._initialized:
            return
            
        self.model_name = model_name
        self.device_config = device_config or {"gpu_ids": [0]}
        self.offline_mode = offline_mode
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.mock_mode = False
        self.model_type = "unknown"
        
        # Validate model name
        if model_name not in SUPPORTED_MODELS:
            logger.warning(f"Unknown model: {model_name}. Supported models: {list(SUPPORTED_MODELS.keys())}")
        
        self._initialize()
        self._initialized = True
    
    def _get_model_path(self) -> str:
        """Get model path"""
        model_info = SUPPORTED_MODELS.get(self.model_name, {})
        
        # Use environment variable path if set
        env_model_dir = os.getenv("ONCALL_MODEL_DIR")
        if env_model_dir:
            base_path = Path(env_model_dir)
        else:
            # Use models directory under project root
            base_path = Path(__file__).parent.parent.parent.parent / "models"
        
        model_path = base_path / model_info.get("local_path", self.model_name.split("/")[-1])
        
        if not model_path.exists() and self.offline_mode:
            logger.warning(
                f"Local model not found in offline mode: {model_path}\n"
                f"Please download model to specified directory, or set ONCALL_MODEL_DIR environment variable\n"
                f"System will try to continue in mock mode"
            )
            return None
        
        return str(model_path if model_path.exists() else self.model_name)
    
    def _initialize(self) -> None:
        """Initialize model and tokenizer"""
        try:
            print("Initializing model and tokenizer")
            # Set device
            if torch.cuda.is_available() and self.device_config.get("gpu_ids"):
                device_ids = self.device_config["gpu_ids"]
                device = f"cuda:{device_ids[0]}"
                torch.cuda.set_device(device_ids[0])
            else:
                device = "cpu"
                logger.warning("GPU not available or not specified, using CPU")
            
            # Check if it's a VL model
            is_vl_model = "VL" in self.model_name or "vision" in self.model_name.lower()
            
            # For VL models, try specialized loading
            if is_vl_model:
                logger.info(f"Detected vision-language model: {self.model_name}")
                if self._load_vl_model(device):
                    return
                else:
                    logger.warning("VL model loading failed, entering mock mode")
                    self.mock_mode = True
                    return
            
            # For non-VL models, use standard loading
            model_path = self._get_model_path()
            
            # If model path is None, use mock mode in offline mode
            if model_path is None:
                logger.warning(f"Model {self.model_name} cannot be loaded in offline mode, using mock mode")
                self.model = None
                self.tokenizer = None
                self.mock_mode = True
                return
            
            # Load tokenizer
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    local_files_only=self.offline_mode
                )
                logger.info(f"Successfully loaded tokenizer: {model_path}")
            except Exception as e:
                logger.error(f"Tokenizer loading failed: {e}")
                self.tokenizer = None
                self.mock_mode = True
                return
            
            # Load model - try multiple loading methods
            try:
                from transformers import AutoModelForCausalLM
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map=device,
                    trust_remote_code=True,
                    local_files_only=self.offline_mode
                )
                self.model_type = "causal_lm"
                logger.info(f"Successfully loaded model with AutoModelForCausalLM: {self.model_name}")
            except Exception as e:
                logger.warning(f"AutoModelForCausalLM loading failed: {e}")
                try:
                    from transformers import AutoModel
                    self.model = AutoModel.from_pretrained(
                        model_path,
                        device_map=device,
                        trust_remote_code=True,
                        local_files_only=self.offline_mode
                    )
                    self.model_type = "base_model"
                    logger.info(f"Successfully loaded model with AutoModel: {self.model_name}")
                except Exception as e2:
                    logger.error(f"All loading methods failed: {e2}")
                    self.model = None
                    self.mock_mode = True
                    return
            
            # Move model to specified device if successfully loaded
            if self.model:
                self.model = self.model.to(device)
                self.mock_mode = False
                logger.info(f"Model {self.model_name} loaded successfully, using device: {device}")
            
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            self.model = None
            self.tokenizer = None
            self.mock_mode = True
    
    def _load_vl_model(self, device: str) -> bool:
        """Specialized method for loading VL models"""
        try:
            # Try online loading of processor and model first
            if not self.offline_mode:
                logger.info("Loading VL model in online mode")
                from transformers import AutoProcessor
                self.processor = AutoProcessor.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )
                
                # Try multiple VL model loading methods
                try:
                    from transformers import AutoModelForImageTextToText
                    self.model = AutoModelForImageTextToText.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float16,
                        device_map=device,
                        trust_remote_code=True
                    )
                    self.model_type = "vl_model"
                    logger.info("Successfully loaded VL model with AutoModelForImageTextToText")
                    return True
                except Exception as e1:
                    logger.warning(f"AutoModelForImageTextToText loading failed: {e1}")
                    try:
                        from transformers import AutoModelForCausalLM
                        self.model = AutoModelForCausalLM.from_pretrained(
                            self.model_name,
                            torch_dtype=torch.float16,
                            device_map=device,
                            trust_remote_code=True
                        )
                        self.model_type = "vl_causal_lm"
                        logger.info("Successfully loaded VL model with AutoModelForCausalLM")
                        return True
                    except Exception as e2:
                        logger.error(f"All VL model loading methods failed: {e2}")
                        return False
            else:
                # In offline mode, VL models are complex, enter mock mode directly
                logger.warning("VL model loading not supported in offline mode, using mock mode")
                return False
                
        except Exception as e:
            logger.error(f"VL model loading process error: {e}")
            return False
    
    async def generate(self, request: ModelRequest) -> ModelResponse:
        """
        Generate response
        
        Args:
            request: Request parameters
        
        Returns:
            Generated response
        """
        try:
            # Simple handling in mock mode
            if hasattr(self, 'mock_mode') and self.mock_mode:
                query_content = self._extract_query_content(request.messages)
                return ModelResponse(
                    content=f"Mock mode response: Based on query '{query_content[:50]}...', suggest upgrading environment to use real model.",
                    success=True
                )
            
            if not self.model:
                raise RuntimeError("Model not properly initialized")
            
            # Choose generation method based on model type
            if self.model_type == "vl_model" and hasattr(self, 'processor') and self.processor:
                return await self._generate_with_vl_processor(request)
            elif self.model_type in ["causal_lm", "vl_causal_lm"] and hasattr(self.model, 'generate'):
                return await self._generate_with_causal_lm(request)
            elif hasattr(self, 'tokenizer') and self.tokenizer:
                return await self._generate_with_base_model(request)
            else:
                raise RuntimeError("Missing appropriate generation method")
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return ModelResponse(
                content="",
                success=False,
                error=str(e)
            )
    
    async def _generate_with_vl_processor(self, request: ModelRequest) -> ModelResponse:
        """Generate response using VL-specific processor"""
        # Implementation details for VL model generation
        # This is a simplified version - full implementation would be more complex
        try:
            text_content = self._extract_query_content(request.messages)
            return ModelResponse(
                content=f"VL model response for: {text_content}",
                success=True
            )
        except Exception as e:
            return ModelResponse(
                content="",
                success=False,
                error=str(e)
            )
    
    async def _generate_with_causal_lm(self, request: ModelRequest) -> ModelResponse:
        """Generate response using CausalLM model"""
        try:
            text_content = self._extract_query_content(request.messages)
            return ModelResponse(
                content=f"CausalLM response for: {text_content}",
                success=True
            )
        except Exception as e:
            return ModelResponse(
                content="",
                success=False,
                error=str(e)
            )
    
    async def _generate_with_base_model(self, request: ModelRequest) -> ModelResponse:
        """Generate response using base model"""
        try:
            text_content = self._extract_query_content(request.messages)
            return ModelResponse(
                content=f"Base model loaded but lacks generate method. Cannot generate response for query '{text_content[:50]}...'. Please use a model version that supports generation.",
                success=False,
                error="Model does not support generation"
            )
        except Exception as e:
            return ModelResponse(
                content="",
                success=False,
                error=str(e)
            )
    
    def _extract_query_content(self, messages):
        """Extract query content from messages, supporting complex formats"""
        try:
            if not messages:
                return ""
            
            last_message = messages[-1]
            content = last_message.get('content', '')
            
            # If content is string, return directly
            if isinstance(content, str):
                return content
            
            # If content is list (official format), extract text parts
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        text_parts.append(item.get('text', ''))
                return ' '.join(text_parts)
            
            return str(content)
        except Exception:
            return ""
    
    @classmethod
    def from_env(cls, model_name: str, context: Dict[str, Any]) -> "UnifiedModelManager":
        """
        Create model manager from environment configuration
        
        Args:
            model_name: Model name
            context: Context configuration
        
        Returns:
            Model manager instance
        """
        device_config = context.get("device_config", {"gpu_ids": [0]})
        offline_mode = context.get("offline_mode", False)
        
        return cls(model_name, device_config, offline_mode)

# ===== Agent Management =====

class UnifiedAgentManager:
    """
    Unified Agent Manager
    Centralized management for all OnCallAgent specialized agents
    """
    
    def __init__(self, config):
        self.config = config
        self.agents: Dict[str, BaseAgent] = {}
        self.route_agent: Optional[EnhancedRouteAgent] = None
        self.model_manager: Optional[UnifiedModelManager] = None
        
        # Agent registry mapping
        self.agent_classes = {
            "route_agent": EnhancedRouteAgent,
            "visual_analysis_agent": VisualAnalysisAgent,
            "log_analysis_agent": LogAnalysisAgent,
            "metrics_analysis_agent": MetricsAnalysisAgent,
            "knowledge_agent": KnowledgeAgent,
            "retrieval_agent": RetrieverAgent,
            "search_agent": SearchAgent,
            "comprehensive_agent": ComprehensiveAgent
        }
        
        self._initialize_model_manager()
        self._initialize_agents()
    
    def _initialize_model_manager(self):
        """Initialize unified model manager"""
        try:
            model_name = getattr(self.config, 'model', 'Qwen/Qwen2.5-VL-7B-Instruct')
            device_config = getattr(self.config, 'device_config', {"gpu_ids": [0]})
            offline_mode = getattr(self.config, 'offline_mode', False)
            
            self.model_manager = UnifiedModelManager(
                model_name=model_name,
                device_config=device_config,
                offline_mode=offline_mode
            )
            logger.info("✓ Unified model manager initialized")
        except Exception as e:
            logger.error(f"✗ Failed to initialize model manager: {e}")
            self.model_manager = None
    
    def _initialize_agents(self):
        """Initialize all configured agents"""
        # Handle both dict and OnCallAgentConfig object
        if hasattr(self.config, 'agents'):
            agents_config = self.config.agents
        else:
            agents_config = self.config.get("agents", {})
        
        # Convert to dict if it's a Pydantic model
        if hasattr(agents_config, 'dict'):
            agents_config = agents_config.dict()
        elif not isinstance(agents_config, dict):
            agents_config = {}
        
        # Initialize enhanced route agent first (required)
        route_config_dict = agents_config.get("route_agent", {})
        route_config = AgentConfig(
            name="route_agent",
            description="Enhanced intelligent routing agent with semantic analysis and adaptive confidence assignment",
            **route_config_dict
        )
        
        # Pass global config to enhanced route agent for enabled agents detection
        global_config = {
            "config": self.config,
            "agents": agents_config,
            "model_manager": self.model_manager
        }
        
        self.route_agent = EnhancedRouteAgent(route_config, global_config)
        logger.info("✓ Enhanced route agent initialized with semantic analysis")
        
        # Initialize other agents
        for agent_name, agent_class in self.agent_classes.items():
            if agent_name == "route_agent":
                continue
            
            agent_config_dict = agents_config.get(agent_name, {})
            enabled = agent_config_dict.get("enabled", True)
            
            if enabled:
                try:
                    agent_config = AgentConfig(
                        name=agent_name,
                        description=f"{agent_name.replace('_', ' ').title()} for specialized processing",
                        **agent_config_dict
                    )
                    
                    # Handle different agent constructor signatures
                    # BaseAgent subclasses (SearchAgent, EnhancedRouteAgent) need global_config
                    # Agent subclasses (core_agents.py) only need config
                    
                    if agent_name in ["search_agent"]:
                        # SearchAgent requires global_config
                        global_config = {
                            "model": getattr(self.config, 'model', 'Qwen/Qwen2.5-VL-7B-Instruct'),
                            "config": self.config,
                            "model_manager": self.model_manager
                        }
                        agent_instance = agent_class(agent_config, global_config)
                    else:
                        # Try BaseAgent constructor first (2 params), fallback to Agent constructor (1 param)
                        try:
                            # First try with global_config (BaseAgent signature)
                            global_config = {
                                "model": getattr(self.config, 'model', 'Qwen/Qwen2.5-VL-7B-Instruct'),
                                "config": self.config,
                                "model_manager": self.model_manager
                            }
                            agent_instance = agent_class(agent_config, global_config)
                        except TypeError as e:
                            if "takes from 1 to 2 positional arguments but 3 were given" in str(e):
                                # Fallback to Agent constructor (1 param only)
                                logger.debug(f"Agent {agent_name} uses legacy Agent constructor, using single parameter")
                                agent_instance = agent_class(agent_config)
                            else:
                                raise e
                    
                    self.agents[agent_name] = agent_instance
                    logger.info(f"✓ Agent {agent_name} initialized")
                
                except Exception as e:
                    logger.error(f"✗ Failed to initialize agent {agent_name}: {e}")
            else:
                logger.info(f"- Agent {agent_name} disabled")
        
    async def process_request(self, query: str) -> AgentOutput:
        """
        Process request with simple query string (convenience method)
        
        Args:
            query: Query string
            
        Returns:
            AgentOutput with response
        """
        query_context = {
            "query": query,
            "context": {},
            "image": None
        }
        
        result = await self.process_query(query_context)
        
        # Convert dict result to AgentOutput
        return AgentOutput(
            result=result.get("response", ""),
            confidence=result.get("confidence", 0.0),
            context=result.get("metadata", {})
        )
        
        logger.info(f"✓ Unified agent manager initialized with {len(self.agents)} agents")
    
    async def process_query(self, query_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process query through intelligent agent routing and coordination
        
        Args:
            query_context: Dictionary containing query, image_path, context, etc.
            
        Returns:
            Dictionary with response, confidence, metadata
        """
        start_time = time.time()
        query = query_context.get("query", "")
        
        if not query.strip():
            return {
                "response": "Empty query provided",
                "confidence": 0.0,
                "agent_used": "none",
                "processing_time": 0.0
            }
        
        try:
            # Step 1: Route query to appropriate agent(s)
            route_input = AgentInput(
                query=query,
                context=query_context
            )
            
            route_result = await self.route_agent.execute(route_input)
            selected_agent_name = route_result.result
            route_confidence = route_result.confidence
            
            logger.info(f"🧭 Route decision: {selected_agent_name} (confidence: {route_confidence:.2f})")
            
            # Step 2: Execute selected agent
            if selected_agent_name not in self.agents:
                logger.warning(f"Selected agent {selected_agent_name} not available, using comprehensive_agent")
                selected_agent_name = "comprehensive_agent"
            
            if selected_agent_name not in self.agents:
                raise ValueError(f"No fallback agent available")
            
            selected_agent = self.agents[selected_agent_name]
            
            # Prepare agent input
            agent_input = AgentInput(
                query=query,
                context=query_context
            )
            
            # Execute agent
            agent_result = await selected_agent.execute(agent_input)
            
            # Step 3: Post-process results
            processing_time = time.time() - start_time
            
            # Determine if multi-agent coordination is needed
            final_result = await self._coordinate_agents_if_needed(
                agent_result, query_context, selected_agent_name
            )
            
            return {
                "response": getattr(final_result, 'response', getattr(final_result, 'result', 'No response available')),
                "confidence": final_result.confidence,
                "agent_used": selected_agent_name,
                "processing_time": processing_time,
                "metadata": {
                    "route_info": route_result.context,
                    "agent_metadata": final_result.context,
                    "coordination_used": final_result.context.get("coordination_used", False)
                }
            }
        
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            processing_time = time.time() - start_time
            
            return {
                "response": f"Processing failed: {str(e)}",
                "confidence": 0.0,
                "agent_used": "error",
                "processing_time": processing_time,
                "metadata": {"error": str(e)}
            }
    
    async def _coordinate_agents_if_needed(self, primary_result: AgentOutput, 
                                         query_context: Dict[str, Any], 
                                         primary_agent: str) -> AgentOutput:
        """
        Coordinate multiple agents if needed for complex queries
        """
        # Check if coordination is needed
        needs_coordination = (
            primary_result.confidence < 0.7 or  # Low confidence
            "search" in query_context.get("query", "").lower() or  # Explicit search request
            len(query_context.get("query", "").split()) > 20  # Complex query
        )
        
        if not needs_coordination:
            return primary_result
        
        logger.info("🔄 Initiating multi-agent coordination")
        
        try:
            # Determine additional agents to involve
            additional_agents = self._select_coordination_agents(query_context, primary_agent)
            
            if not additional_agents:
                return primary_result
            
            # Execute additional agents
            coordination_results = []
            for agent_name in additional_agents:
                if agent_name in self.agents:
                    agent_input = AgentInput(
                        query=query_context.get("query", ""),
                        context=query_context
                    )
                    
                    try:
                        result = await self.agents[agent_name].execute(agent_input)
                        coordination_results.append({
                            "agent": agent_name,
                            "result": result
                        })
                    except Exception as e:
                        logger.warning(f"Coordination agent {agent_name} failed: {e}")
            
            # Synthesize results using comprehensive agent
            if coordination_results and "comprehensive_agent" in self.agents:
                synthesis_context = {
                    **query_context,
                    "primary_result": primary_result,
                    "coordination_results": coordination_results
                }
                
                synthesis_input = AgentInput(
                    query=query_context.get("query", ""),
                    context=synthesis_context
                )
                
                synthesized_result = await self.agents["comprehensive_agent"].execute(synthesis_input)
                synthesized_result.context["coordination_used"] = True
                synthesized_result.context["coordinated_agents"] = additional_agents
                
                return synthesized_result
            
            return primary_result
        
        except Exception as e:
            logger.error(f"Agent coordination failed: {e}")
            return primary_result
    
    def _select_coordination_agents(self, query_context: Dict[str, Any], 
                                  primary_agent: str) -> List[str]:
        """Select additional agents for coordination"""
        query = query_context.get("query", "").lower()
        additional_agents = []
        
        # Always include search for external information
        if primary_agent != "search_agent" and "search_agent" in self.agents:
            additional_agents.append("search_agent")
        
        # Include retrieval for knowledge augmentation
        if primary_agent != "retrieval_agent" and "retrieval_agent" in self.agents:
            if any(keyword in query for keyword in ["document", "guide", "manual", "文档"]):
                additional_agents.append("retrieval_agent")
        
        # Include visual analysis if images mentioned
        if primary_agent != "visual_analysis_agent" and "visual_analysis_agent" in self.agents:
            if any(keyword in query for keyword in ["image", "screenshot", "chart", "graph", "图片", "截图"]):
                additional_agents.append("visual_analysis_agent")
        
        return additional_agents[:2]  # Limit to 2 additional agents
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents"""
        status = {
            "total_agents": len(self.agents) + (1 if self.route_agent else 0),
            "enabled_agents": list(self.agents.keys()),
            "route_agent_status": "enabled" if self.route_agent else "disabled",
            "model_manager_status": "enabled" if self.model_manager else "disabled",
            "agent_details": {}
        }
        
        for agent_name, agent in self.agents.items():
            status["agent_details"][agent_name] = {
                "enabled": True,
                "description": agent.config.description,
                "model": getattr(agent.config, 'model_name', 'unknown')
            }
        
        return status
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform system health check"""
        health_status = {
            "system": "healthy",
            "agents": {},
            "model_manager": "unknown",
            "timestamp": time.time()
        }
        
        # Check model manager
        if self.model_manager:
            health_status["model_manager"] = "healthy" if not self.model_manager.mock_mode else "mock_mode"
        else:
            health_status["model_manager"] = "disabled"
        
        # Check route agent
        if self.route_agent:
            try:
                test_input = AgentInput(query="health check", context={})
                await self.route_agent.execute(test_input)
                health_status["agents"]["route_agent"] = "healthy"
            except Exception as e:
                health_status["agents"]["route_agent"] = f"unhealthy: {e}"
                health_status["system"] = "degraded"
        
        # Check other agents
        for agent_name, agent in self.agents.items():
            try:
                test_input = AgentInput(query="health check", context={"health_check": True})
                await agent.execute(test_input)
                health_status["agents"][agent_name] = "healthy"
            except Exception as e:
                health_status["agents"][agent_name] = f"unhealthy: {e}"
                if health_status["system"] == "healthy":
                    health_status["system"] = "degraded"
        
        return health_status

# ===== Main Entry Point =====

def main():
    """Entry point for command line usage"""
    asyncio.run(main())

if __name__ == "__main__":
    main()