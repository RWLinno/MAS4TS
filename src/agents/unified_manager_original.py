#!/usr/bin/env python3
"""
Unified Agent Manager
Centralized management for all OnCallAgent specialized agents
Inspired by Eigent's multi-agent workforce architecture
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Type
from pathlib import Path

from .base import Agent, AgentInput as BaseAgentInput, AgentOutput as BaseAgentOutput, AgentConfig as BaseAgentConfig
from .core_agents import RouteAgent, VisualAnalysisAgent, LogAnalysisAgent, MetricsAnalysisAgent, ComprehensiveAgent, KnowledgeAgent, RetrieverAgent
#from .visual_agent import VisualAnalysisAgent
#from .route_agent import RouteAgent
#from .log_agent import LogAnalysisAgent
#from .metrics_agent import MetricsAnalysisAgent
#from .retrieval_agent import RetrievalAgent
#from .comprehensive_agent import ComprehensiveAgent
#from .knowledge_agent import KnowledgeAgent
from .search_agent import SearchAgent

logger = logging.getLogger(__name__)

class UnifiedAgentManager:
    """
    Unified manager for all OnCallAgent specialized agents
    Handles agent initialization, routing, and coordination
    """
    def __init__(self, config):
        self.config = config
        self.agents: Dict[str, BaseAgent] = {}
        self.route_agent: Optional[RouteAgent] = None
        
        # Agent registry mapping
        self.agent_classes = {
            "route_agent": RouteAgent,
            "visual_analysis_agent": VisualAnalysisAgent,
            "log_analysis_agent": LogAnalysisAgent,
            "metrics_analysis_agent": MetricsAnalysisAgent,
            "knowledge_agent": KnowledgeAgent,
            "retrieval_agent": RetrieverAgent,
            "search_agent": SearchAgent,
            "comprehensive_agent": ComprehensiveAgent
        }
        
        self._initialize_agents()
    
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
        
        # Initialize route agent first (required)
        route_config_dict = agents_config.get("route_agent", {})
        # Separate standard AgentConfig fields from specialized config
        standard_fields = {'name', 'description', 'enabled', 'model_name', 'max_tokens', 'temperature', 'confidence_threshold', 'timeout_seconds', 'retry_attempts'}
        route_standard = {k: v for k, v in route_config_dict.items() if k in standard_fields}
        route_specialized = {k: v for k, v in route_config_dict.items() if k not in standard_fields}
        
        route_config = BaseAgentConfig(
            name="route_agent",
            description="Intelligent routing agent for query analysis and agent selection",
            version="1.0.0",
            extra_params=route_specialized
        )
        
        self.route_agent = RouteAgent(route_config)
        logger.info("✓ Route agent initialized")
        
        # Initialize other agents
        for agent_name, agent_class in self.agent_classes.items():
            if agent_name == "route_agent":
                continue
            
            agent_config_dict = agents_config.get(agent_name, {})
            enabled = agent_config_dict.get("enabled", True)
            
            if enabled:
                try:
                    agent_config_dict = agents_config.get(agent_name, {})
                    # Separate standard AgentConfig fields from specialized config
                    standard_fields = {'name', 'description', 'enabled', 'model_name', 'max_tokens', 'temperature', 'confidence_threshold', 'timeout_seconds', 'retry_attempts'}
                    agent_standard = {k: v for k, v in agent_config_dict.items() if k in standard_fields}
                    agent_specialized = {k: v for k, v in agent_config_dict.items() if k not in standard_fields}
                    
                    agent_config = BaseAgentConfig(
                        name=agent_name,
                        description=f"{agent_name.replace('_', ' ').title()} for specialized processing",
                        version="1.0.0",
                        extra_params=agent_specialized
                    )
                    
                    # Special handling for SearchAgent which requires global_config
                    if agent_name == "search_agent":
                        # Convert BaseAgentConfig to AgentConfig for SearchAgent
                        from .base import AgentConfig as SearchAgentConfig
                        search_config = SearchAgentConfig(
                            name=agent_name,
                            description=f"{agent_name.replace('_', ' ').title()} for specialized processing",
                            specialized_config=agent_specialized
                        )
                        # Prepare global_config
                        global_config = {
                            "model": getattr(self.config, 'model', 'Qwen/Qwen2.5-VL-7B-Instruct'),
                            "config": self.config
                        }
                        agent_instance = agent_class(search_config, global_config)
                    else:
                        agent_instance = agent_class(agent_config)
                    self.agents[agent_name] = agent_instance
                    logger.info(f"✓ Agent {agent_name} initialized")
                
                except Exception as e:
                    logger.error(f"✗ Failed to initialize agent {agent_name}: {e}")
            else:
                logger.info(f"- Agent {agent_name} disabled")
        
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
            route_input = BaseAgentInput(
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
            agent_input = BaseAgentInput(
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
    
    async def _coordinate_agents_if_needed(self, primary_result: BaseAgentOutput, 
                                         query_context: Dict[str, Any], 
                                         primary_agent: str) -> BaseAgentOutput:
        """
        Coordinate multiple agents if needed for complex queries
        Inspired by Eigent's agent coordination mechanism
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
                    agent_input = BaseAgentInput(
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
                
                synthesis_input = BaseAgentInput(
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
            "timestamp": time.time()
        }
        
        # Check route agent
        if self.route_agent:
            try:
                test_input = BaseAgentInput(query="health check", context={})
                await self.route_agent.execute(test_input)
                health_status["agents"]["route_agent"] = "healthy"
            except Exception as e:
                health_status["agents"]["route_agent"] = f"unhealthy: {e}"
                health_status["system"] = "degraded"
        
        # Check other agents
        for agent_name, agent in self.agents.items():
            try:
                test_input = BaseAgentInput(query="health check", context={"health_check": True})
                await agent.execute(test_input)
                health_status["agents"][agent_name] = "healthy"
            except Exception as e:
                health_status["agents"][agent_name] = f"unhealthy: {e}"
                if health_status["system"] == "healthy":
                    health_status["system"] = "degraded"
        
        return health_status

def main():
    """Entry point for command line usage"""
    asyncio.run(main())

if __name__ == "__main__":
    main()
