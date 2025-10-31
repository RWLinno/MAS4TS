#!/usr/bin/env python3
"""
Enhanced Route Agent with Semantic Analysis and Adaptive Confidence Assignment
Implements intelligent query analysis and dynamic agent selection based on semantic matching
"""

import logging
import re
import math
from typing import Dict, Any, Tuple, List, Set
from dataclasses import dataclass
from enum import Enum

from .base import BaseAgent, AgentInput, AgentOutput, AgentConfig, AgentRegistry

logger = logging.getLogger(__name__)

class QueryType(Enum):
    """Query type classification"""
    VISUAL = "visual"
    SEARCH = "search"
    KNOWLEDGE = "knowledge"
    LOGS = "logs"
    METRICS = "metrics"
    TROUBLESHOOTING = "troubleshooting"
    CONFIGURATION = "configuration"
    MONITORING = "monitoring"
    RETRIEVAL = "retrieval"
    COMPREHENSIVE = "comprehensive"

@dataclass
class SemanticPattern:
    """Semantic pattern definition for agent matching"""
    keywords: List[str]
    phrases: List[str]
    weight: float
    boost_factor: float = 1.0
    context_requirements: List[str] = None

@dataclass
class AgentCapability:
    """Agent capability definition with semantic matching"""
    agent_name: str
    primary_domains: List[str]
    semantic_patterns: List[SemanticPattern]
    base_confidence: float
    enabled: bool = True

@dataclass
class ConfidenceScore:
    """Confidence score with detailed breakdown"""
    agent_name: str
    total_score: float
    semantic_match_score: float
    domain_match_score: float
    context_boost: float
    reasoning: str

@AgentRegistry.register()
class EnhancedRouteAgent(BaseAgent):
    """
    Enhanced routing agent with semantic analysis and adaptive confidence assignment
    Features:
    1. Semantic-based query analysis
    2. Dynamic confidence calculation based on semantic matching
    3. Config-driven agent filtering (only enabled agents participate)
    4. Adaptive routing strategies
    """
    
    def __init__(self, config: AgentConfig, global_config: Dict[str, Any] = None):
        super().__init__(config, global_config)
        
        # Routing configuration
        self.routing_strategy = config.specialized_config.get("routing_strategy", "semantic_adaptive")
        self.confidence_threshold = max(0.3, min(config.confidence_threshold * 0.5, 0.6))  # Lower threshold for semantic routing
        self.min_confidence_gap = 0.1  # Minimum gap between top choices
        
        # Initialize semantic patterns and agent capabilities
        self._initialize_semantic_patterns()
        self._initialize_agent_capabilities()
        
        # Get enabled agents from global config
        self.enabled_agents = self._get_enabled_agents(global_config)
        
        logger.info(f"Enhanced route agent initialized with {len(self.enabled_agents)} enabled agents")
    
    def _get_enabled_agents(self, global_config: Dict[str, Any]) -> Set[str]:
        """Get list of enabled agents from configuration"""
        enabled_agents = set()
        
        if not global_config:
            # Default to all agents if no config provided
            logger.warning("No global config provided, using default agents")
            return {"visual_analysis_agent", "search_agent", "knowledge_agent", 
                   "log_analysis_agent", "metrics_analysis_agent", "retrieval_agent", 
                   "comprehensive_agent"}
        
        # Check global config structure
        config_obj = global_config.get("config")
        if hasattr(config_obj, 'agents'):
            agents_config = config_obj.agents
        elif isinstance(global_config.get("agents"), dict):
            agents_config = global_config["agents"]
        else:
            agents_config = {}
        
        # Convert to dict if it's a Pydantic model
        if hasattr(agents_config, 'dict'):
            agents_config = agents_config.dict()
        
        logger.info(f"Found agents config: {list(agents_config.keys()) if agents_config else 'None'}")
        
        # Check each agent's enabled status
        for agent_name, agent_config in agents_config.items():
            if agent_name == "route_agent":  # Skip self
                continue
                
            if isinstance(agent_config, dict):
                enabled = agent_config.get("enabled", True)
            else:
                enabled = getattr(agent_config, 'enabled', True)
            
            logger.info(f"Agent {agent_name}: enabled={enabled}")
            
            if enabled:
                enabled_agents.add(agent_name)
                logger.debug(f"Agent {agent_name} is enabled")
            else:
                logger.debug(f"Agent {agent_name} is disabled")
        
        logger.info(f"Final enabled agents: {enabled_agents}")
        return enabled_agents
    
    def _initialize_semantic_patterns(self):
        """Initialize semantic patterns for different query types"""
        
        # Visual analysis patterns
        self.visual_patterns = [
            SemanticPattern(
                keywords=["image", "screenshot", "picture", "photo", "chart", "graph", 
                         "dashboard", "monitor", "visual", "display", "diagram", "plot",
                         "图片", "截图", "图像", "图表", "可视化"],
                phrases=["analyze the image", "explain the picture", "what's in the", 
                        "describe the", "in the picture", "in the image", "图片中", "图像中"],
                weight=1.0,
                boost_factor=1.2
            )
        ]
        
        # Search patterns
        self.search_patterns = [
            SemanticPattern(
                keywords=["search", "find", "lookup", "latest", "current", "recent", 
                         "update", "online", "web", "google", "搜索", "查找", "最新"],
                phrases=["search for", "find online", "web search", "latest documentation",
                        "current status", "recent updates", "在线查找", "网络搜索"],
                weight=1.0,
                boost_factor=1.1
            )
        ]
        
        # Knowledge patterns
        self.knowledge_patterns = [
            SemanticPattern(
                keywords=["what is", "how to", "explain", "describe", "definition", 
                         "concept", "principle", "什么是", "如何", "解释", "概念"],
                phrases=["what is", "how to", "can you explain", "tell me about",
                        "什么是", "如何做", "请解释", "介绍一下"],
                weight=0.9,
                boost_factor=1.0
            )
        ]
        
        # Troubleshooting patterns
        self.troubleshooting_patterns = [
            SemanticPattern(
                keywords=["error", "problem", "issue", "broken", "failed", "timeout", 
                         "exception", "bug", "troubleshoot", "fix", "solve", "错误", 
                         "问题", "故障", "异常", "修复", "解决"],
                phrases=["how to fix", "troubleshoot", "solve the problem", 
                        "error occurred", "如何修复", "故障排查", "解决问题"],
                weight=1.1,
                boost_factor=1.3
            )
        ]
        
        # Logs patterns
        self.logs_patterns = [
            SemanticPattern(
                keywords=["log", "logs", "logging", "trace", "debug", "exception", 
                         "stack trace", "日志", "记录", "追踪"],
                phrases=["analyze logs", "log analysis", "check logs", "log file",
                        "日志分析", "查看日志", "日志文件"],
                weight=1.0,
                boost_factor=1.2
            )
        ]
        
        # Metrics patterns
        self.metrics_patterns = [
            SemanticPattern(
                keywords=["metrics", "monitoring", "performance", "cpu", "memory", 
                         "disk", "network", "latency", "throughput", "qps", "指标", 
                         "监控", "性能", "延迟"],
                phrases=["system metrics", "performance monitoring", "resource usage",
                        "系统指标", "性能监控", "资源使用"],
                weight=1.0,
                boost_factor=1.2
            )
        ]
        
        # Retrieval patterns - 修复检索功能异常
        self.retrieval_patterns = [
            SemanticPattern(
                keywords=["document", "documentation", "manual", "guide", "reference",
                         "文档", "手册", "指南", "参考", "@"],
                phrases=["@", "document reference", "check documentation", 
                        "refer to", "文档引用", "查看文档", "@文档"],
                weight=1.2,  # 增加权重
                boost_factor=1.6,  # 增加boost因子
                context_requirements=["document_reference", "has_doc_reference"]
            )
        ]
        
        # Technical domain patterns
        self.domain_patterns = {
            "database": ["redis", "mysql", "mongodb", "postgres", "database", "db", "sql",
                        "数据库", "缓存", "存储"],
            "infrastructure": ["kubernetes", "docker", "k8s", "container", "deployment", 
                             "cluster", "容器", "集群", "部署"],
            "monitoring": ["prometheus", "grafana", "alert", "monitoring", "metrics", 
                          "dashboard", "告警", "监控", "仪表板"],
            "networking": ["network", "nginx", "load balancer", "dns", "ssl", "tcp", 
                          "http", "网络", "负载均衡"],
            "messaging": ["kafka", "rabbitmq", "message queue", "event", "stream",
                         "消息队列", "事件", "流"]
        }
    
    def _initialize_agent_capabilities(self):
        """Initialize agent capabilities with semantic matching"""
        
        self.agent_capabilities = {
            "visual_analysis_agent": AgentCapability(
                agent_name="visual_analysis_agent",
                primary_domains=["visual", "image", "chart", "dashboard"],
                semantic_patterns=self.visual_patterns,
                base_confidence=0.85
            ),
            
            "search_agent": AgentCapability(
                agent_name="search_agent",
                primary_domains=["search", "online", "latest", "current"],
                semantic_patterns=self.search_patterns,
                base_confidence=0.80
            ),
            
            "knowledge_agent": AgentCapability(
                agent_name="knowledge_agent",
                primary_domains=["knowledge", "concept", "definition", "explanation"],
                semantic_patterns=self.knowledge_patterns,
                base_confidence=0.75
            ),
            
            "log_analysis_agent": AgentCapability(
                agent_name="log_analysis_agent",
                primary_domains=["logs", "logging", "trace", "debug"],
                semantic_patterns=self.logs_patterns,
                base_confidence=0.80
            ),
            
            "metrics_analysis_agent": AgentCapability(
                agent_name="metrics_analysis_agent",
                primary_domains=["metrics", "monitoring", "performance", "system"],
                semantic_patterns=self.metrics_patterns,
                base_confidence=0.80
            ),
            
            "retrieval_agent": AgentCapability(
                agent_name="retrieval_agent",
                primary_domains=["document", "documentation", "reference", "manual"],
                semantic_patterns=self.retrieval_patterns,
                base_confidence=0.90
            ),
            
            "comprehensive_agent": AgentCapability(
                agent_name="comprehensive_agent",
                primary_domains=["complex", "comprehensive", "analysis", "troubleshooting"],
                semantic_patterns=self.troubleshooting_patterns,
                base_confidence=0.70
            )
        }
    
    def _analyze_query_semantics(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive semantic analysis of the query
        Returns detailed semantic characteristics for intelligent routing
        """
        query_lower = query.lower()
        
        # Basic query characteristics
        characteristics = {
            "query_length": len(query.split()),
            "has_text": bool(query.strip()),
            "has_image": self._detect_image_requirement(query, context),
            "has_document_reference": self._detect_document_reference(query),
            "query_type": self._classify_query_type(query_lower),
            "urgency_level": self._assess_urgency(query_lower),
            "complexity_level": self._assess_complexity(query_lower),
            "technical_domains": self._detect_technical_domains(query_lower),
            "semantic_features": self._extract_semantic_features(query_lower)
        }
        
        return characteristics
    
    def _detect_image_requirement(self, query: str, context: Dict[str, Any]) -> bool:
        """Enhanced image detection with context awareness"""
        # Check for actual image in context
        if context.get("image_path") or context.get("image"):
            return True
        
        # Check for image-related keywords with weighted scoring
        image_indicators = [
            ("image", 1.0), ("screenshot", 1.0), ("picture", 1.0), ("photo", 0.8),
            ("chart", 0.9), ("graph", 0.9), ("dashboard", 0.8), ("visual", 0.7),
            ("diagram", 0.8), ("plot", 0.8), ("图片", 1.0), ("截图", 1.0), ("图像", 1.0)
        ]
        
        query_lower = query.lower()
        image_score = sum(weight for keyword, weight in image_indicators 
                         if keyword in query_lower)
        
        return image_score >= 0.8
    
    def _detect_document_reference(self, query: str) -> bool:
        """Detect document reference patterns - 修复检索功能异常"""
        # Check for @document.md pattern
        doc_pattern = r'@([a-zA-Z0-9_.-]+\.(?:md|txt|pdf|doc))'
        if re.search(doc_pattern, query):
            return True
        
        # Check for @文档 pattern (Chinese)
        chinese_doc_pattern = r'@\s*文档'
        if re.search(chinese_doc_pattern, query):
            return True
            
        # Check for general @ symbol followed by text (broader detection)
        general_doc_pattern = r'@\s*[a-zA-Z0-9_\u4e00-\u9fff]'
        if re.search(general_doc_pattern, query):
            return True
            
        return False
    
    def _classify_query_type(self, query_lower: str) -> QueryType:
        """Classify query into primary type using semantic analysis"""
        
        # Define type indicators with weights
        type_indicators = {
            QueryType.VISUAL: [
                ("image", 1.0), ("screenshot", 1.0), ("picture", 1.0), ("photo", 0.9),
                ("chart", 0.9), ("visual", 0.8), ("diagram", 0.8), ("graph", 0.9)
            ],
            QueryType.SEARCH: [
                ("search", 1.0), ("find", 0.8), ("latest", 0.9), ("current", 0.8)
            ],
            QueryType.TROUBLESHOOTING: [
                ("error", 1.0), ("problem", 1.0), ("issue", 0.9), ("failed", 0.9),
                ("broken", 0.9), ("troubleshoot", 1.0)
            ],
            QueryType.LOGS: [
                ("log", 1.0), ("logging", 1.0), ("trace", 0.9), ("exception", 0.9)
            ],
            QueryType.METRICS: [
                ("metrics", 1.0), ("monitoring", 1.0), ("performance", 0.9), ("cpu", 0.8)
            ],
            QueryType.KNOWLEDGE: [
                ("what is", 1.0), ("how to", 1.0), ("explain", 0.9), ("definition", 0.8)
            ],
            QueryType.RETRIEVAL: [
                ("document", 1.0), ("@", 1.2), ("reference", 0.9), ("manual", 0.8)
            ]
        }
        
        # Calculate scores for each type
        type_scores = {}
        for query_type, indicators in type_indicators.items():
            score = sum(weight for keyword, weight in indicators 
                       if keyword in query_lower)
            if score > 0:
                type_scores[query_type] = score
        
        if type_scores:
            return max(type_scores, key=type_scores.get)
        
        return QueryType.COMPREHENSIVE
    
    def _assess_urgency(self, query_lower: str) -> str:
        """Assess query urgency with weighted scoring"""
        urgency_indicators = {
            "high": [("urgent", 1.0), ("critical", 1.0), ("emergency", 1.0), 
                    ("down", 1.0), ("outage", 1.0), ("broken", 0.9), ("failed", 0.8)],
            "medium": [("slow", 0.7), ("performance", 0.6), ("timeout", 0.8), 
                      ("delay", 0.7), ("issue", 0.5)]
        }
        
        high_score = sum(weight for keyword, weight in urgency_indicators["high"] 
                        if keyword in query_lower)
        medium_score = sum(weight for keyword, weight in urgency_indicators["medium"] 
                          if keyword in query_lower)
        
        if high_score >= 0.8:
            return "high"
        elif medium_score >= 0.5 or high_score >= 0.3:
            return "medium"
        return "low"
    
    def _assess_complexity(self, query_lower: str) -> str:
        """Assess query complexity using multiple factors"""
        complexity_score = 0
        
        # Complexity keywords
        complexity_keywords = [
            "analyze", "investigate", "troubleshoot", "diagnose", "comprehensive",
            "detailed", "complex", "multiple", "correlation", "integrate"
        ]
        complexity_score += sum(1 for keyword in complexity_keywords if keyword in query_lower)
        
        # Multiple domains increase complexity
        domain_count = sum(1 for domain, keywords in self.domain_patterns.items() 
                          if any(keyword in query_lower for keyword in keywords))
        complexity_score += max(0, domain_count - 1)
        
        # Query length factor
        word_count = len(query_lower.split())
        if word_count > 25:
            complexity_score += 3
        elif word_count > 15:
            complexity_score += 2
        elif word_count > 10:
            complexity_score += 1
        
        # Coordination indicators
        coordination_words = ["and", "also", "both", "multiple", "various", "different"]
        if any(word in query_lower for word in coordination_words):
            complexity_score += 1
        
        if complexity_score >= 5:
            return "high"
        elif complexity_score >= 3:
            return "medium"
        return "low"
    
    def _detect_technical_domains(self, query_lower: str) -> List[str]:
        """Detect technical domains with confidence scoring"""
        domain_scores = {}
        
        for domain, keywords in self.domain_patterns.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                domain_scores[domain] = score
        
        # Return domains sorted by relevance
        return [domain for domain, score in sorted(domain_scores.items(), 
                                                  key=lambda x: x[1], reverse=True)]
    
    def _extract_semantic_features(self, query_lower: str) -> Dict[str, float]:
        """Extract semantic features for advanced matching"""
        features = {}
        
        # Intent features
        intent_patterns = {
            "question": ["what", "how", "why", "when", "where", "which"],
            "action": ["fix", "solve", "resolve", "configure", "setup", "install"],
            "analysis": ["analyze", "examine", "investigate", "review", "check"],
            "information": ["show", "display", "list", "get", "find", "search"]
        }
        
        for intent, keywords in intent_patterns.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            features[f"intent_{intent}"] = min(score / len(keywords), 1.0)
        
        # Technical depth features
        tech_depth_indicators = [
            "configuration", "architecture", "implementation", "optimization",
            "troubleshooting", "debugging", "monitoring", "analysis"
        ]
        features["technical_depth"] = sum(1 for indicator in tech_depth_indicators 
                                        if indicator in query_lower) / len(tech_depth_indicators)
        
        return features
    
    def _calculate_semantic_confidence(self, agent_name: str, characteristics: Dict[str, Any]) -> ConfidenceScore:
        """
        Calculate confidence score for an agent based on semantic matching
        Uses advanced semantic analysis and pattern matching
        """
        
        if agent_name not in self.agent_capabilities:
            return ConfidenceScore(
                agent_name=agent_name,
                total_score=0.0,
                semantic_match_score=0.0,
                domain_match_score=0.0,
                context_boost=0.0,
                reasoning="Agent not found in capabilities"
            )
        
        capability = self.agent_capabilities[agent_name]
        
        # Skip disabled agents
        if agent_name not in self.enabled_agents:
            return ConfidenceScore(
                agent_name=agent_name,
                total_score=0.0,
                semantic_match_score=0.0,
                domain_match_score=0.0,
                context_boost=0.0,
                reasoning="Agent is disabled in configuration"
            )
        
        query_lower = characteristics.get("query", "").lower()
        
        # 1. Semantic pattern matching
        semantic_score = 0.0
        pattern_matches = []
        
        for pattern in capability.semantic_patterns:
            pattern_score = 0.0
            
            # Keyword matching with weights
            keyword_matches = sum(1 for keyword in pattern.keywords if keyword in query_lower)
            if keyword_matches > 0:
                pattern_score += (keyword_matches / len(pattern.keywords)) * pattern.weight
                pattern_matches.append(f"{keyword_matches} keywords")
            
            # Phrase matching (higher weight)
            phrase_matches = sum(1 for phrase in pattern.phrases if phrase in query_lower)
            if phrase_matches > 0:
                pattern_score += (phrase_matches / len(pattern.phrases)) * pattern.weight * 1.5
                pattern_matches.append(f"{phrase_matches} phrases")
            
            # Apply boost factor
            pattern_score *= pattern.boost_factor
            semantic_score = max(semantic_score, pattern_score)
        
        # 2. Domain matching
        domain_score = 0.0
        domain_matches = []
        
        detected_domains = characteristics.get("technical_domains", [])
        for domain in detected_domains:
            if domain in capability.primary_domains:
                domain_score += 0.3
                domain_matches.append(domain)
        
        # Primary domain exact match bonus
        for primary_domain in capability.primary_domains:
            if any(domain_word in query_lower for domain_word in primary_domain.split()):
                domain_score += 0.2
                domain_matches.append(f"primary:{primary_domain}")
        
        # 3. Context-based boosts
        context_boost = 0.0
        context_reasons = []
        
        # Image context boost for visual agent
        if agent_name == "visual_analysis_agent" and characteristics.get("has_image"):
            context_boost += 0.3
            context_reasons.append("image_present")
        
        # Additional visual boost for visual queries
        if agent_name == "visual_analysis_agent" and characteristics.get("query_type") == QueryType.VISUAL:
            context_boost += 0.4  # Strong boost for visual queries
            context_reasons.append("visual_query_type")
        
        # Document reference boost for retrieval agent - 修复检索功能异常
        if agent_name == "retrieval_agent" and characteristics.get("has_document_reference"):
            context_boost += 0.6  # 增加文档引用的boost
            context_reasons.append("document_reference")
        
        # Additional boost for retrieval agent with @ symbol
        if agent_name == "retrieval_agent" and "@" in query_lower:
            context_boost += 0.4
            context_reasons.append("at_symbol_detected")
        
        # Urgency boost for comprehensive agent
        if agent_name == "comprehensive_agent" and characteristics.get("urgency_level") == "high":
            context_boost += 0.2
            context_reasons.append("high_urgency")
        
        # Complexity boost for comprehensive agent
        if agent_name == "comprehensive_agent" and characteristics.get("complexity_level") == "high":
            context_boost += 0.2
            context_reasons.append("high_complexity")
        
        # Search boost for search agent with real-time indicators
        if agent_name == "search_agent":
            realtime_indicators = ["latest", "current", "recent", "update", "new"]
            if any(indicator in query_lower for indicator in realtime_indicators):
                context_boost += 0.25
                context_reasons.append("realtime_need")
        
        # 4. Calculate total score
        base_confidence = capability.base_confidence
        
        # Weighted combination
        total_score = (
            base_confidence * 0.3 +           # Base capability
            semantic_score * 0.4 +            # Semantic matching (highest weight)
            domain_score * 0.2 +              # Domain matching
            context_boost * 0.1               # Context boost
        )
        
        # Apply semantic boost if strong match
        if semantic_score > 0.8:
            total_score *= 1.1
        
        # Ensure score is within bounds
        total_score = max(0.0, min(1.0, total_score))
        
        # Generate reasoning
        reasoning_parts = []
        if pattern_matches:
            reasoning_parts.append(f"Semantic: {', '.join(pattern_matches)}")
        if domain_matches:
            reasoning_parts.append(f"Domain: {', '.join(domain_matches)}")
        if context_reasons:
            reasoning_parts.append(f"Context: {', '.join(context_reasons)}")
        
        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "No strong matches"
        
        return ConfidenceScore(
            agent_name=agent_name,
            total_score=total_score,
            semantic_match_score=semantic_score,
            domain_match_score=domain_score,
            context_boost=context_boost,
            reasoning=reasoning
        )
    
    def _select_optimal_agent_adaptive(self, characteristics: Dict[str, Any]) -> Tuple[str, str, float]:
        """
        Adaptive agent selection using semantic confidence scoring
        Only considers enabled agents and uses dynamic confidence calculation
        """
        
        # Calculate confidence scores for all enabled agents
        confidence_scores = []
        
        for agent_name in self.agent_capabilities.keys():
            if agent_name in self.enabled_agents:  # Only consider enabled agents
                score = self._calculate_semantic_confidence(agent_name, characteristics)
                confidence_scores.append(score)
                logger.info(f"🎯 {agent_name}: confidence={score.total_score:.3f} (semantic={score.semantic_match_score:.3f}, domain={score.domain_match_score:.3f}, context={score.context_boost:.3f})")
        
        if not confidence_scores:
            # Fallback if no agents are enabled
            return "comprehensive_agent", "No enabled agents available, using fallback", 0.5
        
        # Sort by total score
        confidence_scores.sort(key=lambda x: x.total_score, reverse=True)
        
        # Get top choice
        top_choice = confidence_scores[0]
        
        # Check if we have a clear winner or need comprehensive analysis
        if len(confidence_scores) > 1:
            second_choice = confidence_scores[1]
            confidence_gap = top_choice.total_score - second_choice.total_score
            
            # If the gap is too small, use comprehensive agent for complex analysis
            if confidence_gap < self.min_confidence_gap and top_choice.total_score < 0.8:
                if "comprehensive_agent" in self.enabled_agents:
                    return ("comprehensive_agent", 
                           f"Multiple agents viable (gap: {confidence_gap:.2f}), using comprehensive analysis",
                           0.75)
        
        # Apply confidence threshold
        if top_choice.total_score < self.confidence_threshold:
            # Try to find a suitable fallback
            fallback_agents = ["comprehensive_agent", "knowledge_agent", "search_agent"]
            for fallback in fallback_agents:
                if fallback in self.enabled_agents:
                    return (fallback, 
                           f"Low confidence ({top_choice.total_score:.2f}), using {fallback} as fallback",
                           0.6)
        
        return (top_choice.agent_name, 
               f"Best match: {top_choice.reasoning}",
               top_choice.total_score)
    
    async def _process_query(self, input_data: AgentInput) -> AgentOutput:
        """Process routing query with enhanced semantic analysis"""
        query = input_data.query
        context = input_data.context
        
        try:
            # Enhanced semantic analysis
            characteristics = self._analyze_query_semantics(query, context)
            characteristics["query"] = query  # Add query for confidence calculation
            
            # Adaptive agent selection
            if self.routing_strategy == "semantic_adaptive":
                selected_agent, reasoning, confidence = self._select_optimal_agent_adaptive(characteristics)
            else:
                # Fallback to original logic if needed
                selected_agent, reasoning, confidence = self._select_optimal_agent_legacy(characteristics)
            
            # Generate detailed routing metadata
            routing_metadata = {
                "selected_agent": selected_agent,
                "routing_reasoning": reasoning,
                "confidence_score": confidence,
                "query_characteristics": characteristics,
                "routing_strategy": self.routing_strategy,
                "enabled_agents": list(self.enabled_agents),
                "semantic_analysis": {
                    "query_type": characteristics.get("query_type"),
                    "complexity": characteristics.get("complexity_level"),
                    "urgency": characteristics.get("urgency_level"),
                    "technical_domains": characteristics.get("technical_domains", [])
                }
            }
            
            # Log routing decision with enhanced details
            logger.info(f"🧭 Enhanced routing decision: {selected_agent} (confidence: {confidence:.2f})")
            logger.info(f"📊 Query analysis: {characteristics.get('query_type')} | "
                       f"Complexity: {characteristics.get('complexity_level')} | "
                       f"Domains: {characteristics.get('technical_domains', [])}")
            logger.debug(f"🔍 Routing reasoning: {reasoning}")
            
            return AgentOutput(
                response=selected_agent,
                result=selected_agent,
                confidence=confidence,
                context=routing_metadata,
                coordination_needed=characteristics.get("complexity_level") == "high"
            )
        
        except Exception as e:
            logger.error(f"Enhanced routing failed: {e}")
            
            # Intelligent fallback
            fallback_agent = "comprehensive_agent"
            if fallback_agent not in self.enabled_agents:
                # Find any enabled agent as last resort
                if self.enabled_agents:
                    fallback_agent = list(self.enabled_agents)[0]
                else:
                    fallback_agent = "knowledge_agent"  # Ultimate fallback
            
            return AgentOutput(
                response=fallback_agent,
                result=fallback_agent,
                confidence=0.5,
                context={
                    "error": str(e), 
                    "fallback_used": True,
                    "fallback_agent": fallback_agent,
                    "enabled_agents": list(self.enabled_agents)
                }
            )
    
    def _select_optimal_agent_legacy(self, characteristics: Dict[str, Any]) -> Tuple[str, str, float]:
        """Legacy agent selection for backward compatibility"""
        # This is the original logic as fallback
        
        # 1. Document reference (highest priority)
        if characteristics.get("has_document_reference") and "retrieval_agent" in self.enabled_agents:
            return "retrieval_agent", "Document reference detected, using RAG retrieval", 0.95
        
        # 2. Image processing requirements
        if characteristics.get("has_image") and "visual_analysis_agent" in self.enabled_agents:
            return "visual_analysis_agent", "Image processing required", 0.9
        
        # 3. Search requirements
        query_type = characteristics.get("query_type")
        if query_type == QueryType.SEARCH and "search_agent" in self.enabled_agents:
            return "search_agent", "Search query detected", 0.85
        
        # 4. Specialized agents based on query type
        type_mapping = {
            QueryType.LOGS: "log_analysis_agent",
            QueryType.METRICS: "metrics_analysis_agent",
            QueryType.KNOWLEDGE: "knowledge_agent",
            QueryType.TROUBLESHOOTING: "comprehensive_agent"
        }
        
        if query_type in type_mapping:
            agent_name = type_mapping[query_type]
            if agent_name in self.enabled_agents:
                return agent_name, f"Query type '{query_type.value}' mapped to {agent_name}", 0.8
        
        # 5. Fallback to comprehensive or any available agent
        if "comprehensive_agent" in self.enabled_agents:
            return "comprehensive_agent", "Default comprehensive analysis", 0.7
        elif self.enabled_agents:
            fallback = list(self.enabled_agents)[0]
            return fallback, f"Fallback to available agent: {fallback}", 0.6
        else:
            return "knowledge_agent", "No agents enabled, ultimate fallback", 0.5
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get enhanced routing performance statistics"""
        base_metrics = self.get_performance_metrics()
        
        # Add enhanced routing-specific metrics
        routing_metrics = {
            **base_metrics,
            "routing_strategy": self.routing_strategy,
            "confidence_threshold": self.confidence_threshold,
            "min_confidence_gap": self.min_confidence_gap,
            "enabled_agents_count": len(self.enabled_agents),
            "enabled_agents": list(self.enabled_agents),
            "total_agent_capabilities": len(self.agent_capabilities),
            "semantic_patterns_count": sum(len(cap.semantic_patterns) 
                                         for cap in self.agent_capabilities.values())
        }
        
        return routing_metrics
    
    def get_agent_confidence_breakdown(self, query: str, context: Dict[str, Any] = None) -> Dict[str, ConfidenceScore]:
        """Get detailed confidence breakdown for all agents (useful for debugging)"""
        context = context or {}
        characteristics = self._analyze_query_semantics(query, context)
        characteristics["query"] = query
        
        confidence_breakdown = {}
        for agent_name in self.agent_capabilities.keys():
            confidence_breakdown[agent_name] = self._calculate_semantic_confidence(agent_name, characteristics)
        
        return confidence_breakdown