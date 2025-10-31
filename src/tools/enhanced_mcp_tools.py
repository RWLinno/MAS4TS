#!/usr/bin/env python3
"""
Enhanced MCP Tools
参考Eigent项目的工具架构，为OnCall场景提供增强的工具集成
"""

import asyncio
import logging
import json
from typing import Any, Dict, List, Optional, Union
from abc import ABC, abstractmethod
import aiohttp
import subprocess
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class EnhancedMCPTool(ABC):
    """增强的MCP工具基类，参考Eigent的工具设计模式"""
    
    def __init__(self, name: str, description: str, config: Dict[str, Any] = None):
        self.name = name
        self.description = description
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)
    
    @abstractmethod
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """执行工具操作"""
        pass
    
    def get_schema(self) -> Dict[str, Any]:
        """获取工具的JSON Schema"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self._get_parameters_schema()
        }
    
    @abstractmethod
    def _get_parameters_schema(self) -> Dict[str, Any]:
        """获取参数Schema"""
        pass

class WebSearchTool(EnhancedMCPTool):
    """网络搜索工具，集成多个搜索引擎"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            name="web_search",
            description="搜索网络获取最新技术信息和解决方案",
            config=config
        )
        
        self.search_engines = self.config.get("search_engines", {
            "duckduckgo": True,
            "github": True
        })
    
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        query = params.get("query", "")
        max_results = params.get("max_results", 5)
        search_type = params.get("search_type", "general")  # general, technical, github
        
        if not query:
            return {"error": "搜索查询不能为空"}
        
        try:
            # 导入SearchAgent进行搜索
            from oncall_agent.agents.search_agent import SearchAgent
            
            search_agent = SearchAgent(self.config)
            search_results = await search_agent.search_web(query)
            
            return {
                "success": True,
                "query": query,
                "results": search_results[:max_results],
                "total_found": len(search_results)
            }
        
        except Exception as e:
            logger.error(f"网络搜索失败: {e}")
            return {"error": f"搜索失败: {str(e)}"}
    
    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索查询"},
                "max_results": {"type": "integer", "description": "最大结果数", "default": 5},
                "search_type": {"type": "string", "enum": ["general", "technical", "github"], "default": "general"}
            },
            "required": ["query"]
        }

class SystemMonitorTool(EnhancedMCPTool):
    """系统监控工具，参考Eigent的系统集成方式"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            name="system_monitor",
            description="监控系统资源和服务状态",
            config=config
        )
    
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        action = params.get("action", "status")
        target = params.get("target", "system")
        
        try:
            if action == "status":
                return await self._get_system_status()
            elif action == "process":
                return await self._check_process(target)
            elif action == "resource":
                return await self._get_resource_usage()
            elif action == "service":
                return await self._check_service(target)
            else:
                return {"error": f"未知的监控操作: {action}"}
        
        except Exception as e:
            logger.error(f"系统监控失败: {e}")
            return {"error": f"监控失败: {str(e)}"}
    
    async def _get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        try:
            # CPU使用率
            cpu_cmd = "top -l 1 | grep 'CPU usage' | awk '{print $3}' | sed 's/%//' || echo '0'"
            cpu_result = subprocess.run(cpu_cmd, shell=True, capture_output=True, text=True)
            cpu_usage = float(cpu_result.stdout.strip() or "0")
            
            # 内存使用率
            mem_cmd = "vm_stat | grep 'Pages free' | awk '{print $3}' | sed 's/\\.//' || echo '0'"
            mem_result = subprocess.run(mem_cmd, shell=True, capture_output=True, text=True)
            
            # 磁盘使用率
            disk_cmd = "df -h / | tail -1 | awk '{print $5}' | sed 's/%//' || echo '0'"
            disk_result = subprocess.run(disk_cmd, shell=True, capture_output=True, text=True)
            disk_usage = int(disk_result.stdout.strip() or "0")
            
            return {
                "success": True,
                "system_status": {
                    "cpu_usage_percent": cpu_usage,
                    "disk_usage_percent": disk_usage,
                    "timestamp": asyncio.get_event_loop().time()
                }
            }
        
        except Exception as e:
            return {"error": f"获取系统状态失败: {str(e)}"}
    
    async def _check_process(self, process_name: str) -> Dict[str, Any]:
        """检查进程状态"""
        try:
            cmd = f"pgrep -f {process_name}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                pids = result.stdout.strip().split('\n')
                return {
                    "success": True,
                    "process_running": True,
                    "pids": pids,
                    "count": len(pids)
                }
            else:
                return {
                    "success": True,
                    "process_running": False,
                    "pids": [],
                    "count": 0
                }
        
        except Exception as e:
            return {"error": f"检查进程失败: {str(e)}"}
    
    async def _check_service(self, service_name: str) -> Dict[str, Any]:
        """检查服务状态（支持systemd和brew services）"""
        try:
            # 尝试systemd
            systemd_cmd = f"systemctl is-active {service_name}"
            systemd_result = subprocess.run(systemd_cmd, shell=True, capture_output=True, text=True)
            
            if systemd_result.returncode == 0:
                return {
                    "success": True,
                    "service": service_name,
                    "status": systemd_result.stdout.strip(),
                    "manager": "systemd"
                }
            
            # 尝试brew services（macOS）
            brew_cmd = f"brew services list | grep {service_name}"
            brew_result = subprocess.run(brew_cmd, shell=True, capture_output=True, text=True)
            
            if brew_result.returncode == 0:
                return {
                    "success": True,
                    "service": service_name,
                    "status": "found in brew services",
                    "manager": "brew",
                    "details": brew_result.stdout.strip()
                }
            
            return {
                "success": True,
                "service": service_name,
                "status": "not found",
                "manager": "none"
            }
        
        except Exception as e:
            return {"error": f"检查服务失败: {str(e)}"}
    
    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["status", "process", "resource", "service"]},
                "target": {"type": "string", "description": "目标进程或服务名称"}
            },
            "required": ["action"]
        }

class LogAnalysisTool(EnhancedMCPTool):
    """日志分析工具，参考Eigent的文件处理能力"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            name="log_analysis",
            description="分析系统日志文件，提取错误和异常信息",
            config=config
        )
        
        self.max_lines = self.config.get("max_lines", 1000)
        self.log_patterns = {
            "error": [r"ERROR", r"FATAL", r"CRITICAL", r"Exception", r"Error"],
            "warning": [r"WARN", r"WARNING"],
            "info": [r"INFO", r"DEBUG"]
        }
    
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        action = params.get("action", "analyze")
        file_path = params.get("file_path", "")
        pattern = params.get("pattern", "")
        lines = params.get("lines", self.max_lines)
        
        try:
            if action == "tail":
                return await self._tail_log(file_path, lines)
            elif action == "grep":
                return await self._grep_log(file_path, pattern)
            elif action == "analyze":
                return await self._analyze_log(file_path, lines)
            elif action == "summary":
                return await self._summarize_log(file_path, lines)
            else:
                return {"error": f"未知的日志操作: {action}"}
        
        except Exception as e:
            logger.error(f"日志分析失败: {e}")
            return {"error": f"日志分析失败: {str(e)}"}
    
    async def _tail_log(self, file_path: str, lines: int) -> Dict[str, Any]:
        """获取日志文件的最后几行"""
        if not os.path.exists(file_path):
            return {"error": f"日志文件不存在: {file_path}"}
        
        try:
            cmd = f"tail -n {lines} '{file_path}'"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            return {
                "success": True,
                "file_path": file_path,
                "lines_requested": lines,
                "content": result.stdout,
                "lines_returned": len(result.stdout.split('\n'))
            }
        
        except Exception as e:
            return {"error": f"读取日志失败: {str(e)}"}
    
    async def _grep_log(self, file_path: str, pattern: str) -> Dict[str, Any]:
        """在日志中搜索特定模式"""
        if not os.path.exists(file_path):
            return {"error": f"日志文件不存在: {file_path}"}
        
        try:
            cmd = f"grep -n '{pattern}' '{file_path}' | head -50"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            matches = result.stdout.strip().split('\n') if result.stdout.strip() else []
            
            return {
                "success": True,
                "file_path": file_path,
                "pattern": pattern,
                "matches": matches,
                "match_count": len(matches)
            }
        
        except Exception as e:
            return {"error": f"搜索日志失败: {str(e)}"}
    
    async def _analyze_log(self, file_path: str, lines: int) -> Dict[str, Any]:
        """分析日志文件，统计错误和警告"""
        if not os.path.exists(file_path):
            return {"error": f"日志文件不存在: {file_path}"}
        
        try:
            # 读取日志文件
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                log_lines = f.readlines()[-lines:]
            
            # 统计不同级别的日志
            stats = {"error": 0, "warning": 0, "info": 0, "total": len(log_lines)}
            error_samples = []
            warning_samples = []
            
            for line_num, line in enumerate(log_lines, 1):
                line_lower = line.lower()
                
                # 检查错误
                for pattern in self.log_patterns["error"]:
                    if pattern.lower() in line_lower:
                        stats["error"] += 1
                        if len(error_samples) < 5:
                            error_samples.append(f"Line {line_num}: {line.strip()}")
                        break
                
                # 检查警告
                for pattern in self.log_patterns["warning"]:
                    if pattern.lower() in line_lower:
                        stats["warning"] += 1
                        if len(warning_samples) < 3:
                            warning_samples.append(f"Line {line_num}: {line.strip()}")
                        break
                
                # 检查信息
                for pattern in self.log_patterns["info"]:
                    if pattern.lower() in line_lower:
                        stats["info"] += 1
                        break
            
            return {
                "success": True,
                "file_path": file_path,
                "analysis": {
                    "statistics": stats,
                    "error_samples": error_samples,
                    "warning_samples": warning_samples,
                    "health_status": "unhealthy" if stats["error"] > 0 else "healthy"
                }
            }
        
        except Exception as e:
            return {"error": f"分析日志失败: {str(e)}"}
    
    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["tail", "grep", "analyze", "summary"]},
                "file_path": {"type": "string", "description": "日志文件路径"},
                "pattern": {"type": "string", "description": "搜索模式（用于grep）"},
                "lines": {"type": "integer", "description": "处理的行数", "default": 1000}
            },
            "required": ["action", "file_path"]
        }

class DatabaseHealthTool(EnhancedMCPTool):
    """数据库健康检查工具"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            name="database_health",
            description="检查数据库连接和健康状态",
            config=config
        )
    
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        db_type = params.get("db_type", "")
        action = params.get("action", "health_check")
        
        try:
            if db_type == "redis":
                return await self._check_redis_health(params)
            elif db_type == "mysql":
                return await self._check_mysql_health(params)
            elif db_type == "mongodb":
                return await self._check_mongodb_health(params)
            else:
                return {"error": f"不支持的数据库类型: {db_type}"}
        
        except Exception as e:
            return {"error": f"数据库检查失败: {str(e)}"}
    
    async def _check_redis_health(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """检查Redis健康状态"""
        host = params.get("host", "localhost")
        port = params.get("port", 6379)
        
        try:
            # 使用redis-cli进行健康检查
            cmd = f"redis-cli -h {host} -p {port} ping"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0 and "PONG" in result.stdout:
                # 获取更多Redis信息
                info_cmd = f"redis-cli -h {host} -p {port} info server"
                info_result = subprocess.run(info_cmd, shell=True, capture_output=True, text=True, timeout=5)
                
                return {
                    "success": True,
                    "database": "redis",
                    "status": "healthy",
                    "connection": "successful",
                    "info": info_result.stdout if info_result.returncode == 0 else "info not available"
                }
            else:
                return {
                    "success": False,
                    "database": "redis", 
                    "status": "unhealthy",
                    "error": result.stderr or "连接失败"
                }
        
        except subprocess.TimeoutExpired:
            return {"success": False, "database": "redis", "error": "连接超时"}
        except Exception as e:
            return {"error": f"Redis检查失败: {str(e)}"}
    
    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "db_type": {"type": "string", "enum": ["redis", "mysql", "mongodb"]},
                "action": {"type": "string", "enum": ["health_check", "status", "info"]},
                "host": {"type": "string", "default": "localhost"},
                "port": {"type": "integer"}
            },
            "required": ["db_type"]
        }

class NetworkDiagnosticTool(EnhancedMCPTool):
    """网络诊断工具，OnCall场景常用"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            name="network_diagnostic",
            description="网络连接诊断和延迟测试",
            config=config
        )
    
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        action = params.get("action", "ping")
        target = params.get("target", "")
        
        try:
            if action == "ping":
                return await self._ping_test(target)
            elif action == "traceroute":
                return await self._traceroute_test(target)
            elif action == "port_check":
                return await self._port_check(target, params.get("port", 80))
            else:
                return {"error": f"未知的网络诊断操作: {action}"}
        
        except Exception as e:
            return {"error": f"网络诊断失败: {str(e)}"}
    
    async def _ping_test(self, target: str) -> Dict[str, Any]:
        """Ping测试"""
        if not target:
            return {"error": "目标地址不能为空"}
        
        try:
            cmd = f"ping -c 4 {target}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=15)
            
            if result.returncode == 0:
                # 解析ping结果
                lines = result.stdout.split('\n')
                stats_line = [line for line in lines if "packet loss" in line]
                
                return {
                    "success": True,
                    "target": target,
                    "status": "reachable",
                    "output": result.stdout,
                    "packet_loss": stats_line[0] if stats_line else "unknown"
                }
            else:
                return {
                    "success": False,
                    "target": target,
                    "status": "unreachable",
                    "error": result.stderr
                }
        
        except subprocess.TimeoutExpired:
            return {"success": False, "target": target, "error": "ping超时"}
        except Exception as e:
            return {"error": f"Ping测试失败: {str(e)}"}
    
    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["ping", "traceroute", "port_check"]},
                "target": {"type": "string", "description": "目标主机或IP"},
                "port": {"type": "integer", "description": "端口号（用于port_check）"}
            },
            "required": ["action", "target"]
        }

class EnhancedMCPToolManager:
    """增强的MCP工具管理器，参考Eigent的工具管理架构"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.tools = {}
        
        # 初始化工具
        self._initialize_tools()
    
    def _initialize_tools(self):
        """初始化所有工具"""
        tool_classes = [
            WebSearchTool,
            SystemMonitorTool, 
            LogAnalysisTool,
            DatabaseHealthTool,
            NetworkDiagnosticTool
        ]
        
        for tool_class in tool_classes:
            try:
                tool_config = self.config.get(tool_class.__name__.lower(), {})
                tool = tool_class(tool_config)
                
                if tool.enabled:
                    self.tools[tool.name] = tool
                    logger.info(f"✓ 工具 {tool.name} 已启用")
                else:
                    logger.info(f"- 工具 {tool.name} 未启用")
            
            except Exception as e:
                logger.error(f"初始化工具 {tool_class.__name__} 失败: {e}")
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """获取可用工具列表"""
        return [tool.get_schema() for tool in self.tools.values()]
    
    async def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """执行指定工具"""
        if tool_name not in self.tools:
            return {"error": f"工具 {tool_name} 不存在或未启用"}
        
        tool = self.tools[tool_name]
        return await tool.execute(params)
    
    def get_tool_description(self, tool_name: str) -> str:
        """获取工具描述"""
        if tool_name in self.tools:
            return self.tools[tool_name].description
        return f"未知工具: {tool_name}"
