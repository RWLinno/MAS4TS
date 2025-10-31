from typing import Dict, Any, Optional, List
from .base import BaseAgent, AgentInput, AgentOutput
from oncall_agent.function_calls.base import function_registry, register_function
import logging
import json
import subprocess
import psutil
import platform
import socket

logger = logging.getLogger(__name__)

class OperationsAgent(BaseAgent):
    """Operations agent that handles operations-related issues"""
    
    def __init__(self):
        super().__init__("operations_agent", "1.0.0")
        self.commands = {}  # Command set
        self.metrics = {}  # Metrics set
    
    async def initialize(self) -> None:
        """Initialize agent"""
        await super().initialize()
        
        # Load command set
        self._load_commands()
        
        # Register tool functions
        self._register_tools()
        
        logger.info("Operations agent initialization completed")
    
    def _load_commands(self) -> None:
        """Load command set"""
        self.commands = {
            "system": {
                "status": "systemctl status {service}",
                "start": "systemctl start {service}",
                "stop": "systemctl stop {service}",
                "restart": "systemctl restart {service}"
            },
            "process": {
                "list": "ps aux",
                "kill": "kill {pid}",
                "find": "pgrep {name}"
            },
            "network": {
                "status": "netstat -tuln",
                "connections": "ss -tuln",
                "ping": "ping -c 4 {host}"
            }
        }
    
    def _register_tools(self) -> None:
        """Register tool functions"""
        @register_function
        async def execute_command(command: str, timeout: int = 30) -> Dict[str, Any]:
            """Execute command"""
            try:
                result = subprocess.run(
                    command,
                    shell=True,
                    timeout=timeout,
                    capture_output=True,
                    text=True
                )
                return {
                    "success": result.returncode == 0,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode
                }
            except subprocess.TimeoutExpired:
                return {
                    "success": False,
                    "error": "Command timed out",
                    "timeout": timeout
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e)
                }
        
        @register_function
        async def get_process_info(pid: int) -> Dict[str, Any]:
            """Get process information"""
            try:
                process = psutil.Process(pid)
                return {
                    "pid": pid,
                    "name": process.name(),
                    "status": process.status(),
                    "cpu_percent": process.cpu_percent(),
                    "memory_percent": process.memory_percent(),
                    "create_time": process.create_time(),
                    "cmdline": process.cmdline()
                }
            except psutil.NoSuchProcess:
                return {
                    "error": f"Process {pid} not found"
                }
            except Exception as e:
                return {
                    "error": str(e)
                }
        
        @register_function
        async def get_system_metrics() -> Dict[str, Any]:
            """Get system metrics"""
            return {
                "cpu": {
                    "percent": psutil.cpu_percent(interval=1),
                    "count": psutil.cpu_count(),
                    "freq": psutil.cpu_freq()._asdict()
                },
                "memory": psutil.virtual_memory()._asdict(),
                "disk": psutil.disk_usage('/')._asdict(),
                "network": {
                    "connections": len(psutil.net_connections()),
                    "io": psutil.net_io_counters()._asdict()
                }
            }
    
    async def _process(self, input_data: AgentInput) -> AgentOutput:
        """Process input"""
        # Analyze question type
        question_type = self._analyze_question_type(input_data.query)
        
        # Select processing strategy based on question type
        if question_type == "command":
            return await self._handle_command_question(input_data)
        elif question_type == "process":
            return await self._handle_process_question(input_data)
        elif question_type == "system":
            return await self._handle_system_question(input_data)
        else:
            return AgentOutput(
                result="Sorry, I cannot understand your question type. Please provide more detailed information.",
                confidence=0.3,
                metadata={"question_type": question_type}
            )
    
    def _analyze_question_type(self, query: str) -> str:
        """Analyze question type"""
        query = query.lower()
        
        # Command-related question keywords
        command_keywords = ["execute", "run", "command", "operation"]
        if any(keyword in query for keyword in command_keywords):
            return "command"
        
        # Process-related question keywords
        process_keywords = ["process", "program", "PID", "kill"]
        if any(keyword in query for keyword in process_keywords):
            return "process"
        
        # System-related question keywords
        system_keywords = ["system", "resource", "status", "metrics"]
        if any(keyword in query for keyword in system_keywords):
            return "system"
        
        return "unknown"
    
    async def _handle_command_question(self, input_data: AgentInput) -> AgentOutput:
        """Handle command-related questions"""
        # Extract command information
        command_info = self._extract_command_info(input_data.query)
        
        if not command_info:
            return AgentOutput(
                result="Please specify the command to execute.",
                confidence=0.3,
                metadata={"error": "missing_command"}
            )
        
        # Execute command
        result = await function_registry.call({
            "name": "execute_command",
            "arguments": {
                "command": command_info["command"],
                "timeout": command_info.get("timeout", 30)
            }
        })
        
        if result["success"]:
            return AgentOutput(
                result=f"Command executed successfully:\n{result['stdout']}",
                confidence=0.9,
                metadata={"command": command_info["command"]}
            )
        else:
            return AgentOutput(
                result=f"Command execution failed: {result.get('error', 'Unknown error')}",
                confidence=0.5,
                metadata={"command": command_info["command"], "error": result}
            )
    
    async def _handle_process_question(self, input_data: AgentInput) -> AgentOutput:
        """Handle process-related questions"""
        # Extract process information
        process_info = self._extract_process_info(input_data.query)
        
        if not process_info:
            return AgentOutput(
                result="Please specify the process to query.",
                confidence=0.3,
                metadata={"error": "missing_process"}
            )
        
        # Get process information
        result = await function_registry.call({
            "name": "get_process_info",
            "arguments": {"pid": process_info["pid"]}
        })
        
        if "error" in result:
            return AgentOutput(
                result=f"Failed to get process information: {result['error']}",
                confidence=0.5,
                metadata={"pid": process_info["pid"], "error": result}
            )
        else:
            return AgentOutput(
                result=f"Process information:\n{json.dumps(result, indent=2)}",
                confidence=0.8,
                metadata={"pid": process_info["pid"]}
            )
    
    async def _handle_system_question(self, input_data: AgentInput) -> AgentOutput:
        """Handle system-related questions"""
        # Get system metrics
        metrics = await function_registry.call({
            "name": "get_system_metrics",
            "arguments": {}
        })
        
        # Analyze system status
        status = self._analyze_system_status(metrics)
        
        return AgentOutput(
            result=f"System status: {status}\nSystem metrics:\n{json.dumps(metrics, indent=2)}",
            confidence=0.9,
            metadata={"source": "system_metrics"}
        )
    
    def _extract_command_info(self, query: str) -> Optional[Dict[str, Any]]:
        """Extract command information"""
        # TODO: Implement more complex command extraction
        command_keywords = ["execute", "run", "command"]
        for keyword in command_keywords:
            if keyword in query:
                # Simple command extraction
                parts = query.split(keyword)
                if len(parts) > 1:
                    return {
                        "command": parts[1].strip(),
                        "timeout": 30
                    }
        return None
    
    def _extract_process_info(self, query: str) -> Optional[Dict[str, Any]]:
        """Extract process information"""
        # TODO: Implement more complex process information extraction
        process_keywords = ["process", "PID"]
        for keyword in process_keywords:
            if keyword in query:
                # Simple PID extraction
                parts = query.split(keyword)
                if len(parts) > 1:
                    try:
                        pid = int(parts[1].strip())
                        return {"pid": pid}
                    except ValueError:
                        pass
        return None
    
    def _analyze_system_status(self, metrics: Dict[str, Any]) -> str:
        """Analyze system status"""
        # Check CPU usage
        cpu_usage = metrics["cpu"]["percent"]
        if cpu_usage > 90:
            return "CPU usage too high"
        
        # Check memory usage
        memory_usage = metrics["memory"]["percent"]
        if memory_usage > 90:
            return "Memory usage too high"
        
        # Check disk usage
        disk_usage = metrics["disk"]["percent"]
        if disk_usage > 90:
            return "Disk usage too high"
        
        return "System running normally"

# Create and register operations agent
operations_agent = OperationsAgent()
register_agent(operations_agent) 