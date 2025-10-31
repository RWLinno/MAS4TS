# OnCallAgent/src/oncall_agent/mcp/server.py

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional, Union
import time
from pydantic import BaseModel, Field
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from oncall_agent.knowledge.knowledge_manager import knowledge_manager
from oncall_agent.agents.base import BaseAgent, AgentInput
from oncall_agent.agents.core_agents import RouteAgent
from oncall_agent.agents.engineering_agent import EngineeringAgent
from oncall_agent.agents.operations_agent import OperationsAgent
from oncall_agent.adapters.privacy_adapter import PrivacyAdapter
from oncall_agent.mcp.tools.db_tool import DatabaseTools
from oncall_agent.mcp.tools.log_tool import LogTools

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("oncall_agent.mcp")

# MCP消息模型
class MCPRequest(BaseModel):
    id: str
    type: str = "request"
    tool: str
    params: Dict[str, Any] = Field(default_factory=dict)

class MCPResponse(BaseModel):
    id: str
    type: str = "response"
    tool: str
    status: str = "success"
    result: Optional[Any] = None
    error: Optional[str] = None

class MCPError(BaseModel):
    id: str
    type: str = "error"
    tool: str
    error: str
    details: Optional[Dict[str, Any]] = None

# MCP工具基类
class MCPTool:
    name: str
    description: str
    
    async def execute(self, params: Dict[str, Any]) -> Any:
        raise NotImplementedError("Tool must implement execute method")
    
    def get_schema(self) -> Dict[str, Any]:
        """返回工具的JSON Schema描述"""
        raise NotImplementedError("Tool must implement get_schema method")

# 文件工具
class FileTools(MCPTool):
    name = "file"
    description = "读取和写入文件"
    
    async def execute(self, params: Dict[str, Any]) -> Any:
        action = params.get("action")
        path = params.get("path")
        
        if not path:
            raise ValueError("Path parameter is required")
        
        if action == "read":
            return await self._read_file(path, params.get("encoding", "utf-8"))
        elif action == "write":
            return await self._write_file(path, params.get("content", ""), params.get("encoding", "utf-8"))
        elif action == "list":
            return await self._list_directory(path)
        else:
            raise ValueError(f"Unknown action: {action}")
    
    async def _read_file(self, path: str, encoding: str) -> str:
        """读取文件内容"""
        try:
            with open(path, "r", encoding=encoding) as f:
                return f.read()
        except Exception as e:
            raise ValueError(f"Failed to read file: {str(e)}")
    
    async def _write_file(self, path: str, content: str, encoding: str) -> Dict[str, Any]:
        """写入文件内容"""
        try:
            with open(path, "w", encoding=encoding) as f:
                f.write(content)
            return {"success": True, "path": path}
        except Exception as e:
            raise ValueError(f"Failed to write file: {str(e)}")
    
    async def _list_directory(self, path: str) -> List[Dict[str, Any]]:
        """列出目录内容"""
        try:
            items = []
            for item in os.listdir(path):
                item_path = os.path.join(path, item)
                item_type = "file" if os.path.isfile(item_path) else "directory"
                items.append({
                    "name": item,
                    "type": item_type,
                    "size": os.path.getsize(item_path) if item_type == "file" else None
                })
            return items
        except Exception as e:
            raise ValueError(f"Failed to list directory: {str(e)}")
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "required": ["action", "path"],
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["read", "write", "list"],
                        "description": "要执行的文件操作"
                    },
                    "path": {
                        "type": "string",
                        "description": "文件或目录的路径"
                    },
                    "content": {
                        "type": "string",
                        "description": "要写入的文件内容（仅用于写入操作）"
                    },
                    "encoding": {
                        "type": "string",
                        "default": "utf-8",
                        "description": "文件编码"
                    }
                }
            }
        }

# 知识库工具
class KnowledgeTools(MCPTool):
    name = "knowledge"
    description = "搜索和管理知识库"
    
    async def execute(self, params: Dict[str, Any]) -> Any:
        action = params.get("action")
        
        if action == "search":
            return await self._search_knowledge(
                query=params.get("query"),
                kb_name=params.get("knowledge_base"),
                limit=params.get("limit", 5)
            )
        elif action == "list_kbs":
            return await self._list_knowledge_bases()
        else:
            raise ValueError(f"Unknown action: {action}")
    
    async def _search_knowledge(self, query: str, kb_name: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
        """搜索知识库"""
        if not query:
            raise ValueError("Query parameter is required")
        
        results = knowledge_manager.search(
            query=query,
            knowledge_base_name=kb_name,
            top_k=limit
        )
        
        return results
    
    async def _list_knowledge_bases(self) -> List[Dict[str, Any]]:
        """列出所有知识库"""
        return knowledge_manager.list_knowledge_bases()
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "required": ["action"],
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["search", "list_kbs"],
                        "description": "要执行的知识库操作"
                    },
                    "query": {
                        "type": "string",
                        "description": "搜索查询（仅用于搜索操作）"
                    },
                    "knowledge_base": {
                        "type": "string",
                        "description": "知识库名称，不指定则使用默认知识库"
                    },
                    "limit": {
                        "type": "integer",
                        "default": 5,
                        "description": "返回结果的最大数量"
                    }
                }
            }
        }

# 代理工具
class AgentTools(MCPTool):
    name = "agent"
    description = "调用专业代理处理问题"
    
    def __init__(self):
        self.route_agent = RouteAgent()
        self.engineering_agent = EngineeringAgent()
        self.operations_agent = OperationsAgent()
        self.privacy_adapter = PrivacyAdapter()
    
    async def execute(self, params: Dict[str, Any]) -> Any:
        action = params.get("action")
        
        if action == "process":
            return await self._process_query(
                query=params.get("query"),
                agent_type=params.get("agent_type"),
                context=params.get("context", {})
            )
        elif action == "route":
            return await self._route_query(
                query=params.get("query")
            )
        else:
            raise ValueError(f"Unknown action: {action}")
    
    async def _process_query(self, query: str, agent_type: Optional[str] = None, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """处理查询"""
        if not query:
            raise ValueError("Query parameter is required")
        
        # 应用隐私保护
        safe_query = self.privacy_adapter.process_input(query)
        
        # 选择合适的代理
        agent: BaseAgent
        if not agent_type or agent_type == "auto":
            # 由路由代理确定合适的代理类型
            route_output = await self.route_agent.run(AgentInput(query=safe_query, context=context or {}))
            agent_type = route_output.result if isinstance(route_output.result, str) else "engineering"
        
        if agent_type == "engineering":
            agent = self.engineering_agent
        elif agent_type == "operations":
            agent = self.operations_agent
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        # 处理查询
        result_output = await agent.run(AgentInput(query=safe_query, context=context or {}))
        result = {
            "result": result_output.result,
            "confidence": result_output.confidence,
            "context": result_output.context,
        }
        
        # 应用隐私保护
        if "response" in result:
            result["response"] = self.privacy_adapter.process_output(result["response"])
        
        return result
    
    async def _route_query(self, query: str) -> Dict[str, Any]:
        """路由查询到合适的代理"""
        if not query:
            raise ValueError("Query parameter is required")
        
        # 应用隐私保护
        safe_query = self.privacy_adapter.process_input(query)
        
        # 路由分析
        route_output = await self.route_agent.run(AgentInput(query=safe_query, context={}))
        return {
            "selected_agent": route_output.result,
            "context": route_output.context,
            "confidence": route_output.confidence,
        }
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "required": ["action", "query"],
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["process", "route"],
                        "description": "要执行的代理操作"
                    },
                    "query": {
                        "type": "string",
                        "description": "用户查询或问题"
                    },
                    "agent_type": {
                        "type": "string",
                        "enum": ["auto", "engineering", "operations"],
                        "default": "auto",
                        "description": "要使用的代理类型，auto表示自动选择"
                    },
                    "context": {
                        "type": "object",
                        "description": "查询上下文，包括会话历史等信息"
                    }
                }
            }
        }

# MCP服务器类
class MCPServer:
    def __init__(self):
        self.app = FastAPI(title="OnCallAgent MCP Server")
        self.active_connections: List[WebSocket] = []
        self.tools: Dict[str, MCPTool] = {}
        
        # 注册工具
        self._register_tool(FileTools())
        self._register_tool(KnowledgeTools())
        self._register_tool(DatabaseTools())
        self._register_tool(LogTools())
        self._register_tool(AgentTools())
        
        # 配置CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # 注册WebSocket路由
        @self.app.websocket("/mcp")
        async def websocket_endpoint(websocket: WebSocket):
            await self.handle_websocket(websocket)
    
    def _register_tool(self, tool: MCPTool):
        """注册工具"""
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")
    
    async def handle_websocket(self, websocket: WebSocket):
        """处理WebSocket连接"""
        await websocket.accept()
        self.active_connections.append(websocket)
        
        try:
            # 发送工具列表
            await self._send_tool_list(websocket)
            
            # 处理消息
            while True:
                data = await websocket.receive_text()
                await self._process_message(websocket, data)
        except WebSocketDisconnect:
            logger.info("Client disconnected")
        except Exception as e:
            logger.error(f"Error handling WebSocket: {str(e)}")
        finally:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
    
    async def _send_tool_list(self, websocket: WebSocket):
        """发送工具列表"""
        tools_schema = [tool.get_schema() for tool in self.tools.values()]
        await websocket.send_json({
            "type": "tool_list",
            "tools": tools_schema
        })
    
    async def _process_message(self, websocket: WebSocket, data: str):
        """处理接收到的消息"""
        try:
            request_data = json.loads(data)
            request = MCPRequest(**request_data)
            
            if request.tool not in self.tools:
                await self._send_error(websocket, request.id, request.tool, f"Unknown tool: {request.tool}")
                return
            # 执行工具
            tool = self.tools[request.tool]
            result = await tool.execute(request.params)
            
            # 发送成功响应
            response = MCPResponse(
                id=request.id,
                tool=request.tool,
                status="success",
                result=result
            )
            await websocket.send_json(response.dict())
            
        except ValueError as e:
            # 发送参数错误
            await self._send_error(websocket, request.id, request.tool, str(e))
        except json.JSONDecodeError:
            # 发送格式错误
            await websocket.send_json({
                "type": "error",
                "error": "Invalid JSON format"
            })
        except Exception as e:
            # 发送服务器错误
            logger.exception("Error processing request")
            await self._send_error(
                websocket, 
                request_data.get("id", "unknown"), 
                request_data.get("tool", "unknown"), 
                f"Server error: {str(e)}"
            )
    
    async def _send_error(self, websocket: WebSocket, request_id: str, tool: str, error_message: str, details: Optional[Dict[str, Any]] = None):
        """发送错误响应"""
        error = MCPError(
            id=request_id,
            tool=tool,
            error=error_message,
            details=details
        )
        await websocket.send_json(error.dict())

    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """运行MCP服务器"""
        uvicorn.run(self.app, host=host, port=port)

# 创建MCP客户端类
class MCPClient:
    """MCP客户端，用于与MCP服务器通信"""
    
    def __init__(self, server_url: str = "ws://localhost:8000/mcp"):
        self.server_url = server_url
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.websocket = None
    
    async def connect(self):
        """连接到MCP服务器"""
        import websockets
        self.websocket = await websockets.connect(self.server_url)
        
        # 接收工具列表
        tools_msg = await self.websocket.recv()
        tools_data = json.loads(tools_msg)
        
        if tools_data.get("type") == "tool_list":
            for tool in tools_data.get("tools", []):
                self.tools[tool["name"]] = tool
    
    async def call_tool(self, tool_name: str, **params) -> Any:
        """调用工具"""
        if not self.websocket:
            raise RuntimeError("Not connected to server")
        
        if tool_name not in self.tools:
            raise ValueError(f"Unknown tool: {tool_name}")
        
        request_id = f"req_{int(time.time() * 1000)}"
        request = MCPRequest(
            id=request_id,
            tool=tool_name,
            params=params
        )
        
        await self.websocket.send(json.dumps(request.dict()))
        response_msg = await self.websocket.recv()
        response_data = json.loads(response_msg)
        
        if response_data.get("type") == "error":
            raise RuntimeError(f"Tool error: {response_data.get('error')}")
        
        if response_data.get("status") == "success":
            return response_data.get("result")
        else:
            raise RuntimeError(f"Tool failed: {response_data.get('error')}")
    
    async def close(self):
        """关闭连接"""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None

# 创建MCP配置类
class MCPConfig(BaseModel):
    """MCP服务器配置"""
    host: str = "0.0.0.0"
    port: int = 8000
    ssl_cert: Optional[str] = None
    ssl_key: Optional[str] = None
    log_level: str = "INFO"
    cors_origins: List[str] = ["*"]
    max_connections: int = 100
    
    class Config:
        env_prefix = "MCP_"

# 创建MCP模块入口
def create_mcp_server(config: Optional[MCPConfig] = None) -> MCPServer:
    """创建MCP服务器实例"""
    if not config:
        config = MCPConfig()
    
    # 配置日志级别
    log_level = getattr(logging, config.log_level.upper(), logging.INFO)
    logging.basicConfig(level=log_level)
    
    return MCPServer()

def run_mcp_server(config: Optional[MCPConfig] = None):
    """运行MCP服务器"""
    server = create_mcp_server(config)
    server.run()

if __name__ == "__main__":
    # 从环境变量加载配置
    import os
    
    config = MCPConfig(
        host=os.getenv("MCP_HOST", "0.0.0.0"),
        port=int(os.getenv("MCP_PORT", "8000")),
        log_level=os.getenv("MCP_LOG_LEVEL", "INFO")
    )
    
    run_mcp_server(config)