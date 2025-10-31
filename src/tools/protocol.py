from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union
from pydantic import BaseModel, Field, create_model
import inspect
import json
import re

class MCPRole(str, Enum):
    """MCP角色定义"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"

class MCPFunctionParameter(BaseModel):
    """函数参数定义"""
    name: str
    type: str
    description: Optional[str] = None
    required: bool = False
    default: Optional[Any] = None

class MCPFunction(BaseModel):
    """MCP函数定义"""
    name: str
    description: str
    parameters: List[MCPFunctionParameter]
    
    def to_openai_schema(self) -> Dict[str, Any]:
        """转换为OpenAI函数调用格式"""
        properties = {}
        required = []
        
        for param in self.parameters:
            prop_info = {"type": param.type}
            if param.description:
                prop_info["description"] = param.description
            properties[param.name] = prop_info
            
            if param.required:
                required.append(param.name)
        
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }

class MCPMessage(BaseModel):
    """MCP消息定义"""
    role: MCPRole
    content: Optional[str] = None
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None

class FunctionRegistry:
    """函数注册表"""
    _functions: Dict[str, Callable] = {}
    _function_schemas: Dict[str, MCPFunction] = {}
    
    @classmethod
    def register(cls, func: Callable) -> Callable:
        """注册函数"""
        func_name = func.__name__
        cls._functions[func_name] = func
        
        # 解析函数签名和文档，生成函数schema
        signature = inspect.signature(func)
        docstring = inspect.getdoc(func) or ""
        param_docs = cls._parse_docstring_params(docstring)
        
        # 构建参数列表
        parameters = []
        for name, param in signature.parameters.items():
            if name == "self":
                continue
                
            param_type = "string"
            if param.annotation != inspect.Parameter.empty:
                param_type = cls._get_type_name(param.annotation)
            
            required = param.default == inspect.Parameter.empty
            default = None if required else param.default
            
            description = param_docs.get(name, "")
            
            parameters.append(MCPFunctionParameter(
                name=name,
                type=param_type,
                description=description,
                required=required,
                default=default
            ))
        
        # 提取函数描述
        description = cls._extract_function_description(docstring)
        
        # 创建函数schema
        schema = MCPFunction(
            name=func_name,
            description=description,
            parameters=parameters
        )
        
        cls._function_schemas[func_name] = schema
        return func
    
    @classmethod
    def get_function(cls, name: str) -> Optional[Callable]:
        """获取注册的函数"""
        return cls._functions.get(name)
    
    @classmethod
    def get_function_schema(cls, name: str) -> Optional[MCPFunction]:
        """获取函数schema"""
        return cls._function_schemas.get(name)
    
    @classmethod
    def get_all_functions(cls) -> Dict[str, Callable]:
        """获取所有注册的函数"""
        return cls._functions.copy()
    
    @classmethod
    def get_all_schemas(cls) -> List[Dict[str, Any]]:
        """获取所有函数的OpenAI格式schema"""
        return [schema.to_openai_schema() for schema in cls._function_schemas.values()]
    
    @staticmethod
    def _get_type_name(annotation: Type) -> str:
        """获取类型名称"""
        if annotation == str:
            return "string"
        elif annotation == int:
            return "integer"
        elif annotation == float:
            return "number"
        elif annotation == bool:
            return "boolean"
        elif hasattr(annotation, "__origin__") and annotation.__origin__ == list:
            return "array"
        elif hasattr(annotation, "__origin__") and annotation.__origin__ == dict:
            return "object"
        return "string"
    
    @staticmethod
    def _parse_docstring_params(docstring: str) -> Dict[str, str]:
        """解析docstring中的参数描述"""
        param_pattern = r":param\s+(\w+):\s+(.*?)(?=:param|:return|$)"
        matches = re.finditer(param_pattern, docstring, re.DOTALL)
        
        param_docs = {}
        for match in matches:
            param_name = match.group(1).strip()
            param_desc = match.group(2).strip()
            param_docs[param_name] = param_desc
            
        return param_docs
    
    @staticmethod
    def _extract_function_description(docstring: str) -> str:
        """提取函数描述"""
        lines = docstring.strip().split("\n")
        description_lines = []
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith(":"):
                break
            description_lines.append(line)
            
        return " ".join(description_lines)

class MCPContext:
    """MCP上下文管理器"""
    
    def __init__(self):
        self.messages: List[MCPMessage] = []
        self.function_registry = FunctionRegistry
    
    def add_system_message(self, content: str):
        """添加系统消息"""
        self.messages.append(MCPMessage(role=MCPRole.SYSTEM, content=content))
        
    def add_user_message(self, content: str):
        """添加用户消息"""
        self.messages.append(MCPMessage(role=MCPRole.USER, content=content))
        
    def add_assistant_message(self, content: str, function_call: Optional[Dict[str, Any]] = None):
        """添加助手消息"""
        self.messages.append(MCPMessage(
            role=MCPRole.ASSISTANT, 
            content=content,
            function_call=function_call
        ))
        
    def add_function_result(self, name: str, content: str):
        """添加函数执行结果"""
        self.messages.append(MCPMessage(
            role=MCPRole.FUNCTION,
            name=name,
            content=content
        ))
    
    def call_function(self, name: str, arguments: str) -> Optional[str]:
        """执行函数调用"""
        func = self.function_registry.get_function(name)
        if not func:
            return f"Error: Function '{name}' not found"
        
        try:
            # 解析参数
            args_dict = json.loads(arguments)
            
            # 执行函数
            result = func(**args_dict)
            
            # 记录函数结果
            result_str = str(result)
            self.add_function_result(name, result_str)
            
            return result_str
        except Exception as e:
            error_msg = f"Error executing function '{name}': {str(e)}"
            self.add_function_result(name, error_msg)
            return error_msg
    
    def to_openai_messages(self) -> List[Dict[str, Any]]:
        """转换为OpenAI消息格式"""
        openai_messages = []
        
        for msg in self.messages:
            openai_msg = {"role": msg.role.value}
            
            if msg.content is not None:
                openai_msg["content"] = msg.content
                
            if msg.name:
                openai_msg["name"] = msg.name
                
            if msg.function_call:
                openai_msg["function_call"] = msg.function_call
                
            openai_messages.append(openai_msg)
            
        return openai_messages

def function(func: Callable) -> Callable:
    """函数注册装饰器"""
    return FunctionRegistry.register(func) 