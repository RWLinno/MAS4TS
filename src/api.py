#!/usr/bin/env python3
"""
OnCallAgent FastAPI服务
提供REST API接口用于智能运维问答
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import base64

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入OnCallAgent核心模块
try:
    from main import process_request
    print("✅ Successfully imported process_request from main")
except ImportError as e:
    print(f"❌ Failed to import process_request: {e}")
    sys.exit(1)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("OnCallAgent.API")

# 创建FastAPI应用
app = FastAPI(
    title="OnCallAgent API",
    description="智能运维问答系统API接口",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 请求模型
class QueryRequest(BaseModel):
    """查询请求模型"""
    query: str = Field(..., description="用户查询内容")
    image: Optional[str] = Field(None, description="图片base64编码（可选）")
    context: Dict[str, Any] = Field(default_factory=dict, description="上下文信息")
    model: str = Field("Qwen/Qwen2.5-VL-7B-Instruct", description="使用的模型")
    type: str = Field("offline", description="模型类型")
    device: str = Field("cpu", description="运行设备")
    agents: Dict[str, Dict[str, bool]] = Field(
        default_factory=lambda: {
            "route_agent": {"enabled": True},
            "visual_analysis_agent": {"enabled": True},
            "metrics_analysis_agent": {"enabled": True},
            "log_analysis_agent": {"enabled": True},
            "knowledge_agent": {"enabled": True},
            "comprehensive_agent": {"enabled": True},
            "retrieval_agent": {"enabled": True},
            "search_agent": {"enabled": True}
        },
        description="启用的智能体配置"
    )

class QueryResponse(BaseModel):
    """查询响应模型"""
    success: bool = Field(..., description="是否成功")
    answer: str = Field(..., description="回答内容")
    confidence: float = Field(..., description="置信度")
    agent_used: str = Field(..., description="使用的智能体")
    processing_time: float = Field(..., description="处理时间")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    error: Optional[str] = Field(None, description="错误信息")

# API路由
@app.get("/")
async def root():
    """根路径，返回API信息"""
    return {
        "message": "OnCallAgent API服务",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    """健康检查接口"""
    try:
        # 检查系统状态
        import torch
        gpu_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if gpu_available else 0
        
        return {
            "status": "healthy",
            "timestamp": asyncio.get_event_loop().time(),
            "system": {
                "gpu_available": gpu_available,
                "gpu_count": gpu_count,
                "python_version": sys.version
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """处理用户查询"""
    try:
        logger.info(f"🤖 Processing query: {request.query[:100]}...")
        
        # 构建配置
        config = {
            "query": request.query,
            "image": request.image,
            "context": request.context,
            "model": request.model,
            "type": request.type,
            "device": request.device,
            "agents": request.agents
        }
        
        # 调用处理函数
        result = await process_request(config)
        
        # 构建响应
        response = QueryResponse(
            success=True,
            answer=result.get("answer", "No response generated"),
            confidence=result.get("confidence", 0.0),
            agent_used=result.get("agent_used", "unknown"),
            processing_time=result.get("processing_time", 0.0),
            metadata=result.get("metadata", {})
        )
        
        logger.info(f"✅ Query processed successfully, agent: {response.agent_used}")
        return response
        
    except Exception as e:
        logger.error(f"❌ Query processing failed: {e}")
        error_response = QueryResponse(
            success=False,
            answer=f"Processing failed: {str(e)}",
            confidence=0.0,
            agent_used="error",
            processing_time=0.0,
            error=str(e)
        )
        return error_response

@app.post("/query/multimodal")
async def process_multimodal_query(
    query: str = Form(...),
    image: Optional[UploadFile] = File(None),
    context: str = Form("{}"),
    model: str = Form("Qwen/Qwen2.5-VL-7B-Instruct"),
    device: str = Form("cpu")
):
    """处理多模态查询（支持文件上传）"""
    try:
        logger.info(f"🖼️ Processing multimodal query: {query[:100]}...")
        
        # 处理图片
        image_data = None
        if image:
            image_bytes = await image.read()
            image_data = base64.b64encode(image_bytes).decode("utf-8")
            logger.info(f"📷 Image uploaded: {image.filename}, size: {len(image_bytes)} bytes")
        
        # 解析上下文
        try:
            context_dict = json.loads(context)
        except json.JSONDecodeError:
            context_dict = {}
        
        # 构建请求
        request = QueryRequest(
            query=query,
            image=image_data,
            context=context_dict,
            model=model,
            device=device
        )
        
        # 调用查询处理
        return await process_query(request)
        
    except Exception as e:
        logger.error(f"❌ Multimodal query processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents")
async def list_agents():
    """列出可用的智能体"""
    agents = {
        "route_agent": {
            "name": "路由智能体",
            "description": "负责查询路由和智能体选择",
            "type": "routing"
        },
        "visual_analysis_agent": {
            "name": "视觉分析智能体", 
            "description": "处理图像和视觉内容分析",
            "type": "multimodal"
        },
        "metrics_analysis_agent": {
            "name": "指标分析智能体",
            "description": "分析系统指标和性能数据",
            "type": "analysis"
        },
        "log_analysis_agent": {
            "name": "日志分析智能体",
            "description": "分析系统日志和错误信息",
            "type": "analysis"
        },
        "knowledge_agent": {
            "name": "知识问答智能体",
            "description": "基于知识库回答技术问题",
            "type": "knowledge"
        },
        "comprehensive_agent": {
            "name": "综合分析智能体",
            "description": "综合多种信息源进行分析",
            "type": "comprehensive"
        },
        "retrieval_agent": {
            "name": "检索增强智能体",
            "description": "基于文档检索的问答",
            "type": "retrieval"
        },
        "search_agent": {
            "name": "搜索智能体",
            "description": "网络搜索和信息获取",
            "type": "search"
        }
    }
    
    return {
        "agents": agents,
        "total_count": len(agents)
    }

@app.get("/config")
async def get_config():
    """获取系统配置信息"""
    try:
        config_path = project_root / "config.json"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return {
                "config": config,
                "config_path": str(config_path)
            }
        else:
            return {
                "error": "Configuration file not found",
                "config_path": str(config_path)
            }
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load config: {str(e)}")

# 启动函数
def start_server(host: str = "0.0.0.0", port: int = 8080, reload: bool = False):
    """启动API服务器"""
    print("🚀 OnCallAgent API Server Starting...")
    print(f"📍 Server URL: http://{host}:{port}")
    print(f"📖 API Docs: http://{host}:{port}/docs")
    print(f"🔍 ReDoc: http://{host}:{port}/redoc")
    print("=" * 50)
    
    uvicorn.run(
        "src.api:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="OnCallAgent API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8080, help="Port number")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    start_server(args.host, args.port, args.reload)