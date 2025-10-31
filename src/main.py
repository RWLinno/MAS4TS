import asyncio
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
import base64
from io import BytesIO
from PIL import Image
import json
import traceback

from .agents.base import AgentInput, AgentConfig, AgentRegistry
from .agents.core_agents import (
    RouteAgent, 
    KnowledgeAgent, 
    VisualAnalysisAgent,
    MetricsAnalysisAgent,
    LogAnalysisAgent,
    ComprehensiveAgent,
    RetrieverAgent
)
from .agents.search_agent import SearchAgent

class OnCallAgentResponse(BaseModel):
    answer: str
    confidence: float
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class OnCallAgent:    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agents = {}
        self.route_agent = None
        
        self._initialize_agents()
        
        print(f"✓ 多智能体系统初始化完成，启用的智能体: {list(self.agents.keys())}")

    def _initialize_agents(self):
        """初始化所有智能体"""
        agents_config = self.config.get("agents", {})
        
        agent_map = {
            "route_agent": RouteAgent,
            "visual_analysis_agent": VisualAnalysisAgent,
            "metrics_analysis_agent": MetricsAnalysisAgent,
            "log_analysis_agent": LogAnalysisAgent,
            "knowledge_agent": KnowledgeAgent,
            "comprehensive_agent": ComprehensiveAgent,
            "retriever_agent": RetrieverAgent,
            "search_agent": SearchAgent  # 新增：网络搜索智能体
        }
        
        global_config = {
            **self.config,
            "model": self.config.get("model", "Qwen/Qwen2.5-VL-7B-Instruct"),
            "device": self.config.get("device", "cpu"),
            "offline_mode": self.config.get("type", "offline") == "offline",
            "device_config": {"gpu_ids": [0]}
        }
        
        route_config = AgentConfig(
            name="route_agent",
            description="路由智能体",
            version="1.0.0",
            extra_params={
                **agents_config.get("route_agent", {}),
                "global_config": global_config
            }
        )
        self.route_agent = RouteAgent(route_config)
        print("✓ 路由智能体已初始化")
        
        for agent_name, agent_class in agent_map.items():
            if agent_name == "route_agent":
                continue 
                
            agent_config_dict = agents_config.get(agent_name, {})
            
            enabled = agent_config_dict.get("enabled", True)
            
            if enabled:
                try:
                    enhanced_config = AgentConfig(
                        name=agent_name,
                        description=f"{agent_name} agent",
                        version="1.0.0",
                        extra_params={
                            **agent_config_dict,
                            "global_config": global_config
                        }
                    )
                    
                    self.agents[agent_name] = agent_class(enhanced_config)
                    print(f"✓ 智能体 {agent_name} 已启用")
                except Exception as e:
                    print(f"✗ 智能体 {agent_name} 初始化失败: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"- 智能体 {agent_name} 未启用")

    async def _prepare_context(self, query: str, image: Any, context: Dict[str, Any], model: str) -> Dict[str, Any]:
        """准备输入上下文"""
        if image:
            try:
                if isinstance(image, str):
                    if image.startswith("data:image"):
                        img_data = image.split(",")[1]
                    else:
                        img_data = image
                    
                    image_bytes = base64.b64decode(img_data)
                    image_obj = Image.open(BytesIO(image_bytes))
                    context["image"] = image_obj
                    print(f"✓ 成功处理图像: {image_obj.size}")
                else:
                    context["image"] = image
            except Exception as e:
                print(f"✗ 图像处理失败: {str(e)}")
        
        # 添加全局配置到上下文
        context.update({
            "global_config": self.config,
            "model": model,
            "device": self.config.get("device", "cpu"),
            "offline_mode": self.config.get("type", "offline") == "offline"
        })
        
        return context

    async def process_query(self, config: Dict[str, Any]) -> OnCallAgentResponse:
        """处理查询请求"""
        query = config.get("query", "")
        image = config.get("image")
        context = config.get("context", {})
        model = config.get("model", "Qwen/Qwen2.5-VL-7B-Instruct")
        
        print(f"🤖 处理查询: {query[:50]}{'...' if len(query) > 50 else ''}")
        
        try:
            # 准备上下文
            context = await self._prepare_context(query, image, context, model)
            input_data = AgentInput(query=query, context=context)
            
            # 步骤1: 路由决策
            if not self.route_agent:
                raise RuntimeError("路由智能体未初始化")
            
            route_output = await self.route_agent.run(input_data)
            selected_agent_name = route_output.result
            route_info = route_output.context.get("route_info", {})
            
            print(f"✓ 路由决策: {selected_agent_name}")
            print(f"✓ 路由理由: {route_info.get('reasoning', 'N/A')}")
            
            # 步骤2: 执行选中的智能体
            if selected_agent_name in self.agents:
                target_agent = self.agents[selected_agent_name]
                print(f"✓ 执行智能体: {selected_agent_name}")
                
                result_output = await target_agent.run(input_data)
                
                return OnCallAgentResponse(
                    answer=result_output.result,
                    confidence=result_output.confidence,
                    metadata={
                        "selected_agent": selected_agent_name,
                        "route_info": route_info,
                        "agent_context": result_output.context
                    }
                )
            else:
                # 如果选中的智能体不可用，回退到综合智能体
                fallback_agent = "comprehensive_agent"
                if fallback_agent in self.agents:
                    print(f"⚠️ 智能体 {selected_agent_name} 不可用，回退到 {fallback_agent}")
                    target_agent = self.agents[fallback_agent]
                    result_output = await target_agent.run(input_data)
                    
                    return OnCallAgentResponse(
                        answer=result_output.result,
                        confidence=result_output.confidence * 0.8,  # 降低置信度
                        metadata={
                            "selected_agent": fallback_agent,
                            "original_selection": selected_agent_name,
                            "fallback_reason": f"智能体 {selected_agent_name} 不可用",
                            "agent_context": result_output.context
                        }
                    )
                else:
                    return OnCallAgentResponse(
                        answer="抱歉，所有智能体都不可用，请检查系统配置。",
                        confidence=0.0,
                        metadata={"error": f"智能体 {selected_agent_name} 和回退智能体都不可用"}
                    )
            
        except Exception as e:
            print(f"✗ 处理请求时发生错误: {str(e)}")
            traceback.print_exc()
            return OnCallAgentResponse(
                answer="抱歉，处理您的请求时发生了错误。请稍后再试。",
                confidence=0.0,
                metadata={"error": str(e)}
            )

# 全局变量，用于保持兼容性
oncall_agent = None

async def process_request(config: Dict[str, Any]) -> OnCallAgentResponse:
    """处理请求的全局函数"""
    global oncall_agent
    if oncall_agent is None:
        oncall_agent = OnCallAgent(config)
    
    return await oncall_agent.process_query(config)

def sync_process_request(
    query: str,
    image: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    project_id: Optional[str] = None,
    model: Optional[str] = None
) -> Dict[str, Any]:
    """
    同步处理请求的便捷函数（保持向后兼容）
    """
    config = {
        "query": query,
        "image": image,
        "context": context or {},
        "model": model or "Qwen/Qwen2.5-VL-7B-Instruct",
        "type": "offline",
        "device": "cpu",
        "agents": {
            "route_agent": {"enabled": True},
            "visual_analysis_agent": {"enabled": True},
            "metrics_analysis_agent": {"enabled": True},
            "log_analysis_agent": {"enabled": True},
            "knowledge_agent": {"enabled": True},
            "comprehensive_agent": {"enabled": True},
            "retriever_agent": {"enabled": True}
        }
    }
    
    response = asyncio.run(process_request(config))
    return response.dict()

def sync_process_request_with_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """根据配置同步处理请求"""
    response = asyncio.run(process_request(config))
    return response.dict() 