#!/usr/bin/env python3
"""
Route Agent
Intelligent query analysis and agent selection
Enhanced with Eigent-inspired routing strategies
"""

import logging
import re
from typing import Dict, Any, Tuple, List
from .base import BaseAgent, AgentInput, AgentOutput, AgentConfig, AgentRegistry

logger = logging.getLogger(__name__)

@AgentRegistry.register()
class RouteAgent(BaseAgent):
    """
    Intelligent routing agent for query analysis and agent selection
    Inspired by Eigent's sophisticated routing mechanisms
    """
    
    def __init__(self, config: AgentConfig, global_config: Dict[str, Any]):
        super().__init__(config, global_config)
        
        # Routing strategy configuration
        self.routing_strategy = config.specialized_config.get("routing_strategy", "intelligent")
        self.confidence_threshold = config.confidence_threshold
        
        # Query pattern definitions
        self._initialize_patterns()
    
    def _initialize_patterns(self):
        """Initialize query analysis patterns"""
        # Multi-modal detection patterns
        self.image_keywords = [
            "image", "screenshot", "picture", "photo", "chart", "graph", "dashboard",
            "monitor", "visual", "display", "diagram", "plot"
        ]
        
        self.document_reference_pattern = r'@([a-zA-Z0-9_.-]+\.(?:md|txt|pdf|doc))'
        
        # Technical domain patterns
        self.domain_patterns = {
            "database": ["redis", "mysql", "mongodb", "postgres", "database", "db", "sql"],
            "infrastructure": ["kubernetes", "docker", "k8s", "container", "deployment", "cluster"],
            "monitoring": ["prometheus", "grafana", "alert", "monitoring", "metrics", "dashboard"],
            "logging": ["log", "logging", "error", "exception", "trace", "debug"],
            "networking": ["network", "nginx", "load balancer", "dns", "ssl", "tcp", "http"],
            "messaging": ["kafka", "rabbitmq", "message queue", "event", "stream"],
            "search": ["search", "find", "lookup", "latest", "current", "recent", "documentation"]
        }
        
        # Urgency indicators
        self.urgency_keywords = [
            "urgent", "critical", "emergency", "down", "outage", "broken", "failed"
        ]
        
        # Complexity indicators
        self.complexity_keywords = [
            "analyze", "investigate", "troubleshoot", "diagnose", "comprehensive",
            "detailed", "complex", "multiple", "correlation"
        ]
    
    def _detect_query_characteristics(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze query characteristics for intelligent routing
        Enhanced pattern detection inspired by Eigent's query analysis
        """
        query_lower = query.lower()
        
        characteristics = {
            # Basic modalities
            "has_text": bool(query.strip()),
            "has_image": self._detect_image_requirement(query, context),
            "has_document_reference": bool(re.search(self.document_reference_pattern, query)),
            
            # Technical domains
            "primary_domain": self._detect_primary_domain(query_lower),
            "secondary_domains": self._detect_secondary_domains(query_lower),
            
            # Query intent
            "intent_type": self._classify_intent(query_lower),
            "urgency_level": self._assess_urgency(query_lower),
            "complexity_level": self._assess_complexity(query_lower),
            
            # Processing requirements
            "needs_real_time_info": self._needs_real_time_info(query_lower),
            "needs_multi_agent": self._needs_multi_agent_coordination(query_lower),
            "needs_external_search": self._needs_external_search(query_lower)
        }
        
        return characteristics
    
    def _detect_image_requirement(self, query: str, context: Dict[str, Any]) -> bool:
        """Detect if query requires image processing"""
        # Check for actual image in context
        if context.get("image_path") or context.get("image"):
            return True
        
        # Check for image-related keywords in query
        return any(keyword in query.lower() for keyword in self.image_keywords)
    
    def _detect_primary_domain(self, query_lower: str) -> str:
        """Detect primary technical domain"""
        domain_scores = {}
        
        for domain, keywords in self.domain_patterns.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        return "general"
    
    def _detect_secondary_domains(self, query_lower: str) -> List[str]:
        """Detect secondary technical domains"""
        domains = []
        primary_domain = self._detect_primary_domain(query_lower)
        
        for domain, keywords in self.domain_patterns.items():
            if domain != primary_domain:
                if any(keyword in query_lower for keyword in keywords):
                    domains.append(domain)
        
        return domains[:2]  # Limit to 2 secondary domains
    
    def _classify_intent(self, query_lower: str) -> str:
        """Classify query intent"""
        intent_patterns = {
            "troubleshooting": ["error", "problem", "issue", "broken", "failed", "timeout", "exception"],
            "information": ["what is", "how to", "explain", "describe", "definition"],
            "analysis": ["analyze", "investigate", "examine", "review", "assess"],
            "configuration": ["configure", "setup", "install", "deploy", "settings"],
            "monitoring": ["monitor", "alert", "metrics", "performance", "status"],
            "search": ["search", "find", "lookup", "locate", "discover"]
        }
        
        for intent, keywords in intent_patterns.items():
            if any(keyword in query_lower for keyword in keywords):
                return intent
        
        return "general"
    
    def _assess_urgency(self, query_lower: str) -> str:
        """Assess query urgency level"""
        if any(keyword in query_lower for keyword in self.urgency_keywords):
            return "high"
        
        # Medium urgency indicators
        medium_indicators = ["slow", "performance", "timeout", "delay", "issue"]
        if any(indicator in query_lower for indicator in medium_indicators):
            return "medium"
        
        return "low"
    
    def _assess_complexity(self, query_lower: str) -> str:
        """Assess query complexity level"""
        complexity_score = 0
        
        # Count complexity indicators
        complexity_score += sum(1 for keyword in self.complexity_keywords if keyword in query_lower)
        
        # Multiple domains increase complexity
        domain_count = sum(1 for domain, keywords in self.domain_patterns.items() 
                          if any(keyword in query_lower for keyword in keywords))
        complexity_score += max(0, domain_count - 1)
        
        # Query length as complexity indicator
        word_count = len(query_lower.split())
        if word_count > 20:
            complexity_score += 2
        elif word_count > 10:
            complexity_score += 1
        
        if complexity_score >= 4:
            return "high"
        elif complexity_score >= 2:
            return "medium"
        return "low"
    
    def _needs_real_time_info(self, query_lower: str) -> bool:
        """Determine if query needs real-time information"""
        realtime_indicators = [
            "latest", "current", "recent", "update", "new", "today"
        ]
        return any(indicator in query_lower for indicator in realtime_indicators)
    
    def _needs_multi_agent_coordination(self, query_lower: str) -> bool:
        """Determine if query needs multi-agent coordination"""
        coordination_indicators = [
            "and", "also", "both", "multiple", "various", "different",
            "correlate", "combine", "integrate", "comprehensive"
        ]
        return any(indicator in query_lower for indicator in coordination_indicators)
    
    def _needs_external_search(self, query_lower: str) -> bool:
        """Determine if query needs external web search"""
        search_indicators = [
            "search", "find online", "web search", "google", "latest documentation"
        ]
        return any(indicator in query_lower for indicator in search_indicators)
    
    def _select_optimal_agent(self, characteristics: Dict[str, Any]) -> Tuple[str, str, float]:
        """
        Select optimal agent based on query characteristics
        Enhanced routing logic inspired by Eigent's agent selection
        """
        # Priority-based agent selection
        
        # 1. External search requirements (highest priority for real-time info)
        if characteristics["needs_external_search"] or characteristics["needs_real_time_info"]:
            return "search_agent", "Real-time information or external search required", 0.9
        
        # 2. Document reference (explicit document retrieval)
        if characteristics["has_document_reference"]:
            return "retrieval_agent", "Document reference detected, using RAG retrieval", 0.95
        
        # 3. Image processing requirements
        if characteristics["has_image"]:
            return "visual_analysis_agent", "Image processing required", 0.9
        
        # 4. Domain-specific routing
        primary_domain = characteristics["primary_domain"]
        domain_agent_mapping = {
            "logging": "log_analysis_agent",
            "monitoring": "metrics_analysis_agent",
            "database": "knowledge_agent",  # Can be enhanced with specialized DB agent
            "infrastructure": "comprehensive_agent",
            "search": "search_agent"
        }
        
        if primary_domain in domain_agent_mapping:
            agent_name = domain_agent_mapping[primary_domain]
            return agent_name, f"Primary domain '{primary_domain}' mapped to specialized agent", 0.8
        
        # 5. Intent-based routing
        intent = characteristics["intent_type"]
        intent_agent_mapping = {
            "troubleshooting": "comprehensive_agent",
            "analysis": "comprehensive_agent", 
            "information": "knowledge_agent",
            "search": "search_agent"
        }
        
        if intent in intent_agent_mapping:
            agent_name = intent_agent_mapping[intent]
            return agent_name, f"Intent '{intent}' mapped to appropriate agent", 0.75
        
        # 6. Complexity-based fallback
        complexity = characteristics["complexity_level"]
        if complexity == "high" or characteristics["needs_multi_agent"]:
            return "comprehensive_agent", "High complexity requires comprehensive analysis", 0.7
        
        # 7. Default fallback
        return "knowledge_agent", "Default routing for general queries", 0.6
    
    async def _process_query(self, input_data: AgentInput) -> AgentOutput:
        """Process routing query and select optimal agent"""
        query = input_data.query
        context = input_data.context
        
        try:
            # Analyze query characteristics
            characteristics = self._detect_query_characteristics(query, context)
            
            # Select optimal agent
            selected_agent, reasoning, confidence = self._select_optimal_agent(characteristics)
            
            # Prepare routing result
            routing_metadata = {
                "selected_agent": selected_agent,
                "routing_reasoning": reasoning,
                "query_characteristics": characteristics,
                "routing_strategy": self.routing_strategy,
                "alternative_agents": self._get_alternative_agents(characteristics)
            }
            
            logger.info(f"🧭 Routing decision: {selected_agent} (confidence: {confidence:.2f})")
            logger.debug(f"Routing reasoning: {reasoning}")
            
            return AgentOutput(
                response=selected_agent,
                confidence=confidence,
                context=routing_metadata,
                coordination_needed=characteristics["needs_multi_agent"]
            )
        
        except Exception as e:
            logger.error(f"Routing failed: {e}")
            return AgentOutput(
                response="comprehensive_agent",  # Safe fallback
                confidence=0.5,
                context={"error": str(e), "fallback_used": True}
            )
    
    def _get_alternative_agents(self, characteristics: Dict[str, Any]) -> List[str]:
        """Get alternative agents that could handle the query"""
        alternatives = []
        
        # Based on characteristics, suggest alternative agents
        if characteristics["has_image"]:
            alternatives.append("visual_analysis_agent")
        
        if characteristics["primary_domain"] in ["logging", "monitoring"]:
            alternatives.extend(["log_analysis_agent", "metrics_analysis_agent"])
        
        if characteristics["needs_real_time_info"]:
            alternatives.append("search_agent")
        
        if characteristics["complexity_level"] == "high":
            alternatives.append("comprehensive_agent")
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(alternatives))
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get routing performance statistics"""
        base_metrics = self.get_performance_metrics()
        
        # Add routing-specific metrics
        routing_metrics = {
            **base_metrics,
            "routing_strategy": self.routing_strategy,
            "confidence_threshold": self.confidence_threshold,
            "average_confidence": base_metrics.get("success_rate", 0.0)  # Approximation
        }
        
        return routing_metrics
