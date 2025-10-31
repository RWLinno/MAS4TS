#!/usr/bin/env python3
"""
MCP Toolkit
Enhanced Model Context Protocol tools for OnCall scenarios
Inspired by Eigent's tool integration architecture
"""

import asyncio
import logging
import json
import subprocess
import os
from typing import Dict, Any, List, Optional, Protocol
from abc import ABC, abstractmethod
from pathlib import Path

logger = logging.getLogger(__name__)

class MCPTool(Protocol):
    """Protocol for MCP tools"""
    name: str
    description: str
    
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tool with given parameters"""
        ...
    
    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema for LLM function calling"""
        ...

class DatabaseHealthTool:
    """Database health check and monitoring tool"""
    
    name = "database_health"
    description = "Check database connection status and basic health metrics"
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
    
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute database health check"""
        db_type = params.get("db_type", "")
        host = params.get("host", "localhost")
        port = params.get("port")
        
        if db_type == "redis":
            return await self._check_redis(host, port or 6379)
        elif db_type == "mysql":
            return await self._check_mysql(host, port or 3306)
        elif db_type == "mongodb":
            return await self._check_mongodb(host, port or 27017)
        else:
            return {"error": f"Unsupported database type: {db_type}"}
    
    async def _check_redis(self, host: str, port: int) -> Dict[str, Any]:
        """Check Redis health"""
        try:
            cmd = f"redis-cli -h {host} -p {port} ping"
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=5
            )
            
            if result.returncode == 0 and "PONG" in result.stdout:
                # Get additional Redis info
                info_cmd = f"redis-cli -h {host} -p {port} info server"
                info_result = subprocess.run(
                    info_cmd, shell=True, capture_output=True, text=True, timeout=5
                )
                
                return {
                    "success": True,
                    "database": "redis",
                    "host": host,
                    "port": port,
                    "status": "healthy",
                    "response_time": "< 5s",
                    "additional_info": info_result.stdout if info_result.returncode == 0 else None
                }
            else:
                return {
                    "success": False,
                    "database": "redis",
                    "host": host,
                    "port": port,
                    "status": "unhealthy",
                    "error": result.stderr or "Connection failed"
                }
        
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "database": "redis",
                "error": "Connection timeout"
            }
        except Exception as e:
            return {"error": f"Redis check failed: {str(e)}"}
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "db_type": {
                        "type": "string",
                        "enum": ["redis", "mysql", "mongodb"],
                        "description": "Database type to check"
                    },
                    "host": {
                        "type": "string",
                        "default": "localhost",
                        "description": "Database host"
                    },
                    "port": {
                        "type": "integer",
                        "description": "Database port"
                    }
                },
                "required": ["db_type"]
            }
        }

class LogAnalysisTool:
    """Log file analysis and monitoring tool"""
    
    name = "log_analysis"
    description = "Analyze log files for errors, patterns, and system health"
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.max_lines = self.config.get("max_lines", 1000)
    
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute log analysis"""
        action = params.get("action", "analyze")
        file_path = params.get("file_path", "")
        
        if not file_path or not os.path.exists(file_path):
            return {"error": f"Log file not found: {file_path}"}
        
        try:
            if action == "tail":
                return await self._tail_log(file_path, params.get("lines", 50))
            elif action == "grep":
                return await self._grep_log(file_path, params.get("pattern", "ERROR"))
            elif action == "analyze":
                return await self._analyze_log(file_path)
            elif action == "summary":
                return await self._summarize_log(file_path)
            else:
                return {"error": f"Unknown log action: {action}"}
        
        except Exception as e:
            return {"error": f"Log analysis failed: {str(e)}"}
    
    async def _analyze_log(self, file_path: str) -> Dict[str, Any]:
        """Analyze log file for patterns and issues"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()[-self.max_lines:]
            
            # Count different log levels
            stats = {
                "total_lines": len(lines),
                "error_count": 0,
                "warning_count": 0,
                "info_count": 0
            }
            
            error_samples = []
            warning_samples = []
            
            for line_num, line in enumerate(lines, 1):
                line_lower = line.lower()
                
                if any(pattern in line_lower for pattern in ['error', 'fatal', 'critical', 'exception']):
                    stats["error_count"] += 1
                    if len(error_samples) < 5:
                        error_samples.append(f"Line {line_num}: {line.strip()}")
                
                elif any(pattern in line_lower for pattern in ['warn', 'warning']):
                    stats["warning_count"] += 1
                    if len(warning_samples) < 3:
                        warning_samples.append(f"Line {line_num}: {line.strip()}")
                
                elif any(pattern in line_lower for pattern in ['info', 'debug']):
                    stats["info_count"] += 1
            
            # Determine health status
            health_status = "healthy"
            if stats["error_count"] > 0:
                health_status = "unhealthy"
            elif stats["warning_count"] > 10:
                health_status = "warning"
            
            return {
                "success": True,
                "file_path": file_path,
                "statistics": stats,
                "health_status": health_status,
                "error_samples": error_samples,
                "warning_samples": warning_samples
            }
        
        except Exception as e:
            return {"error": f"Log analysis failed: {str(e)}"}
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["tail", "grep", "analyze", "summary"],
                        "description": "Analysis action to perform"
                    },
                    "file_path": {
                        "type": "string",
                        "description": "Path to log file"
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Pattern to search for (used with grep action)"
                    },
                    "lines": {
                        "type": "integer",
                        "description": "Number of lines to process",
                        "default": 50
                    }
                },
                "required": ["action", "file_path"]
            }
        }

class SystemMonitorTool:
    """System resource monitoring tool"""
    
    name = "system_monitor"
    description = "Monitor system resources, processes, and services"
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
    
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute system monitoring"""
        action = params.get("action", "status")
        
        try:
            if action == "status":
                return await self._get_system_status()
            elif action == "processes":
                return await self._get_process_info(params.get("pattern", ""))
            elif action == "disk":
                return await self._get_disk_usage()
            elif action == "network":
                return await self._get_network_status()
            else:
                return {"error": f"Unknown monitor action: {action}"}
        
        except Exception as e:
            return {"error": f"System monitoring failed: {str(e)}"}
    
    async def _get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        try:
            # Get basic system info
            uptime_cmd = "uptime"
            uptime_result = subprocess.run(uptime_cmd, shell=True, capture_output=True, text=True)
            
            # Get memory info (cross-platform)
            if os.name == 'posix':  # Unix-like systems
                mem_cmd = "free -h" if subprocess.run("which free", shell=True, capture_output=True).returncode == 0 else "vm_stat"
            else:  # Windows
                mem_cmd = "systeminfo | findstr Memory"
            
            mem_result = subprocess.run(mem_cmd, shell=True, capture_output=True, text=True)
            
            return {
                "success": True,
                "uptime": uptime_result.stdout.strip(),
                "memory_info": mem_result.stdout.strip(),
                "timestamp": asyncio.get_event_loop().time()
            }
        
        except Exception as e:
            return {"error": f"System status check failed: {str(e)}"}
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["status", "processes", "disk", "network"],
                        "description": "Monitoring action to perform"
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Pattern to filter results (for processes action)"
                    }
                },
                "required": ["action"]
            }
        }

class MCPToolkit:
    """
    Manager for all MCP tools
    Provides unified interface for tool discovery and execution
    Alias for MCPToolManager for backward compatibility
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.manager = MCPToolManager(config)
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of available tools with schemas"""
        return self.manager.get_available_tools()
    
    async def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute specified tool"""
        return await self.manager.execute_tool(tool_name, params)
    
    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get information about specific tool"""
        return self.manager.get_tool_info(tool_name)

class MCPToolManager:
    """
    Manager for all MCP tools
    Provides unified interface for tool discovery and execution
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.tools: Dict[str, MCPTool] = {}
        
        self._initialize_tools()
    
    def _initialize_tools(self):
        """Initialize all available tools"""
        tool_classes = [
            DatabaseHealthTool,
            LogAnalysisTool,
            SystemMonitorTool
        ]
        
        for tool_class in tool_classes:
            try:
                tool_config = self.config.get(tool_class.name, {})
                tool = tool_class(tool_config)
                self.tools[tool.name] = tool
                logger.info(f"✓ Tool {tool.name} initialized")
            
            except Exception as e:
                logger.error(f"✗ Failed to initialize tool {tool_class.name}: {e}")
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of available tools with schemas"""
        return [tool.get_schema() for tool in self.tools.values()]
    
    async def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute specified tool"""
        if tool_name not in self.tools:
            return {"error": f"Tool {tool_name} not found"}
        
        try:
            result = await self.tools[tool_name].execute(params)
            logger.info(f"✓ Tool {tool_name} executed successfully")
            return result
        
        except Exception as e:
            logger.error(f"✗ Tool {tool_name} execution failed: {e}")
            return {"error": f"Tool execution failed: {str(e)}"}
    
    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get information about specific tool"""
        if tool_name in self.tools:
            return self.tools[tool_name].get_schema()
        return None
