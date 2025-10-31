from typing import Dict, Any, Optional
from .base import BaseAgent, AgentInput, AgentOutput
from oncall_agent.function_calls.base import function_registry, register_function
import logging
import json
import os

logger = logging.getLogger(__name__)

class EngineeringAgent(BaseAgent):
    """Engineering agent that handles engineering-related issues"""
    
    def __init__(self):
        super().__init__("engineering_agent", "1.0.0")
        self.knowledge_base = {}  # Knowledge base
        self.tools = {}  # Tool set
    
    async def initialize(self) -> None:
        """Initialize agent"""
        await super().initialize()
        
        # Load knowledge base
        self._load_knowledge_base()
        
        # Register tool functions
        self._register_tools()
        
        logger.info("Engineering agent initialization completed")
    
    def _load_knowledge_base(self) -> None:
        """Load knowledge base"""
        # TODO: Load knowledge base from file or database
        self.knowledge_base = {
            "kubernetes": "Kubernetes is an open-source container orchestration platform...",
            "docker": "Docker is an open-source containerization platform...",
            "prometheus": "Prometheus is an open-source monitoring system...",
            # More knowledge...
        }
    
    def _register_tools(self) -> None:
        """Register tool functions"""
        @register_function
        async def search_knowledge_base(query: str) -> Dict[str, Any]:
            """Search knowledge base"""
            results = {}
            for key, value in self.knowledge_base.items():
                if query.lower() in key.lower() or query.lower() in value.lower():
                    results[key] = value
            return results
        
        @register_function
        async def get_system_info() -> Dict[str, Any]:
            """Get system information"""
            return {
                "cpu_usage": os.cpu_count(),
                "memory_usage": os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES'),
                "disk_usage": os.statvfs('/').f_frsize * os.statvfs('/').f_blocks
            }
        
        @register_function
        async def check_service_status(service: str) -> Dict[str, Any]:
            """Check service status"""
            # TODO: Implement service status checking
            return {
                "service": service,
                "status": "running",
                "uptime": "1d 2h 3m"
            }
    
    async def _process(self, input_data: AgentInput) -> AgentOutput:
        """Process input"""
        # Analyze question type
        question_type = self._analyze_question_type(input_data.query)
        
        # Select processing strategy based on question type
        if question_type == "knowledge":
            return await self._handle_knowledge_question(input_data)
        elif question_type == "system":
            return await self._handle_system_question(input_data)
        elif question_type == "service":
            return await self._handle_service_question(input_data)
        else:
            return AgentOutput(
                result="Sorry, I cannot understand your question type. Please provide more detailed information.",
                confidence=0.3,
                metadata={"question_type": question_type}
            )
    
    def _analyze_question_type(self, query: str) -> str:
        """Analyze question type"""
        query = query.lower()
        
        # Knowledge-related question keywords
        knowledge_keywords = ["what is", "what", "how", "why"]
        if any(keyword in query for keyword in knowledge_keywords):
            return "knowledge"
        
        # System-related question keywords
        system_keywords = ["system", "resource", "performance", "load"]
        if any(keyword in query for keyword in system_keywords):
            return "system"
        
        # Service-related question keywords
        service_keywords = ["service", "status", "running", "start", "stop"]
        if any(keyword in query for keyword in service_keywords):
            return "service"
        
        return "unknown"
    
    async def _handle_knowledge_question(self, input_data: AgentInput) -> AgentOutput:
        """Handle knowledge-related questions"""
        # Search knowledge base
        search_results = await function_registry.call({
            "name": "search_knowledge_base",
            "arguments": {"query": input_data.query}
        })
        
        if search_results:
            # Found relevant knowledge
            result = "\n".join([f"{key}: {value}" for key, value in search_results.items()])
            return AgentOutput(
                result=result,
                confidence=0.8,
                metadata={"source": "knowledge_base"}
            )
        else:
            # No relevant knowledge found
            return AgentOutput(
                result="Sorry, I cannot answer this question at the moment.",
                confidence=0.2,
                metadata={"source": "knowledge_base"}
            )
    
    async def _handle_system_question(self, input_data: AgentInput) -> AgentOutput:
        """Handle system-related questions"""
        # Get system information
        system_info = await function_registry.call({
            "name": "get_system_info",
            "arguments": {}
        })
        
        # Analyze system status
        status = self._analyze_system_status(system_info)
        
        return AgentOutput(
            result=f"System status: {status}\nDetailed information: {json.dumps(system_info, indent=2)}",
            confidence=0.9,
            metadata={"source": "system_info"}
        )
    
    async def _handle_service_question(self, input_data: AgentInput) -> AgentOutput:
        """Handle service-related questions"""
        # Extract service name
        service_name = self._extract_service_name(input_data.query)
        
        if not service_name:
            return AgentOutput(
                result="Please specify the service name to query.",
                confidence=0.3,
                metadata={"error": "missing_service_name"}
            )
        
        # Check service status
        service_status = await function_registry.call({
            "name": "check_service_status",
            "arguments": {"service": service_name}
        })
        
        return AgentOutput(
            result=f"Service status: {json.dumps(service_status, indent=2)}",
            confidence=0.8,
            metadata={"source": "service_status"}
        )
    
    def _analyze_system_status(self, system_info: Dict[str, Any]) -> str:
        """Analyze system status"""
        # TODO: Implement more complex system status analysis
        return "Normal"
    
    def _extract_service_name(self, query: str) -> Optional[str]:
        """Extract service name"""
        # TODO: Implement more complex service name extraction
        service_keywords = ["service", "process", "program"]
        for keyword in service_keywords:
            if keyword in query:
                # Simple service name extraction
                parts = query.split(keyword)
                if len(parts) > 1:
                    return parts[1].strip()
        return None

# 创建并注册工程代理
engineering_agent = EngineeringAgent()
register_agent(engineering_agent) 