
"""
Knowledge Base Agent
Implements RAG-based knowledge retrieval and Q&A functionality
"""

import os
from typing import Dict, List, Optional, Any
import logging
import asyncio
from pathlib import Path

from ..base import Agent, AgentInput, AgentOutput, AgentConfig, AgentRegistry
from ..utils.model_manager import UnifiedModelManager, ModelRequest
from ..retrieval_simplified.simple_rag import RAGService, Document

logger = logging.getLogger(__name__)
@AgentRegistry.register()
class KnowledgeAgent(Agent):
    """知识智能体 - 基于通用知识回答问题"""
    
    def __init__(self, config: Optional[AgentConfig] = None):
        super().__init__(config)
        self.model_manager = None
    
    def _get_model_manager(self, context: dict) -> UnifiedModelManager:
        if self.model_manager is None:
            model_name = context.get("model", "Qwen/Qwen2-7B-Instruct")
            device_config = context.get("device_config", {"gpu_ids": [0]})
            offline_mode = context.get("offline_mode", False)
            
            self.model_manager = UnifiedModelManager(
                model_name=model_name,
                device_config=device_config,
                offline_mode=offline_mode
            )
        return self.model_manager
    
    async def execute(self, input_data: AgentInput) -> AgentOutput:
        query = input_data.query
        context = input_data.context or {}
        
        try:
            model_manager = self._get_model_manager(context)
            
            messages = [
                {
                    "role": "system",
                    "content": "你是一个专业的技术助手。请根据你的知识详细回答用户的问题，提供准确、实用的建议。"
                },
                {
                    "role": "user",
                    "content": f"请详细回答以下问题：{query}"
                }
            ]
            
            request = ModelRequest(
                messages=messages,
                max_tokens=512,
                temperature=0.7
            )
            
            response = await model_manager.generate(request)
            
            if response.success:
                return AgentOutput(
                    result=response.content,
                    context={"source": "direct_model", "rag_enabled": False},
                    confidence=0.8
                )
            else:
                return AgentOutput(
                    result=f"模型生成失败: {response.error}",
                    confidence=0.0
                )
                
        except Exception as e:
            logger.exception(f"知识查询失败: {e}")
            return AgentOutput(
                result=f"知识查询失败: {str(e)}",
                confidence=0.0
            )

@AgentRegistry.register()
class VisualAnalysisAgent(Agent):
    """视觉分析智能体 - 处理包含图像的查询"""    
    def __init__(self, config: Optional[AgentConfig] = None):
        super().__init__(config)
        print("VisualAnalysisAgent init")
        self.model_manager = None

    def _get_model_manager(self, context: dict) -> UnifiedModelManager:
        """获取模型管理器"""
        if self.model_manager is None:
            model_name = context.get("model", "Qwen/Qwen2.5-VL-7B-Instruct")
            self.model_manager = UnifiedModelManager.from_env(model_name, context)
        return self.model_manager
    
    async def execute(self, input_data: AgentInput) -> AgentOutput:
        query = input_data.query
        context = input_data.context or {}
        
        image_data = context.get("image")
        
        if not image_data:
            image_data = self._auto_find_image(context)
            if image_data:
                print(f"✓ 自动发现图片: {image_data}")
                context["image"] = image_data
            else:
                return AgentOutput(
                    result="查询提到了图像，但未找到图像数据。请提供图片路径或将图片放在data/imgs/目录下。",
                    confidence=0.1
                )
        
        try:
            model_manager = self._get_model_manager(context)
            
            # 对于Qwen2.5-VL，使用简化的消息格式
            if isinstance(image_data, str) and (image_data.startswith('/') or image_data.startswith('./')):
                # 本地文件路径，直接使用文件路径
                image_content = image_data
            else:
                # 其他格式，转换为合适的格式
                image_content = self._prepare_image_content(image_data)
            
            # 使用简化的文本消息格式
            messages = [
                {
                    "role": "system", 
                    "content": "You are a helpful assistant that can analyze images and answer questions about them."
                },
                {
                    "role": "user",
                    "content": f"Please analyze the image and answer: {query}"
                }
            ]
            
            request = ModelRequest(
                messages=messages,
                max_tokens=512,
                temperature=0.7,
                image=image_content
            )

            response = await model_manager.generate(request)
            if response.success:
                return AgentOutput(
                    result=response.content,
                    context={
                        "visual_analysis": response.content,
                        "image_format": "simplified",
                        "image_path": image_data if isinstance(image_data, str) else "processed_image"
                    },
                    confidence=0.9
                )
            else:
                return AgentOutput(
                    result=f"视觉分析失败: {response.error}",
                    confidence=0.0
                )
                
        except Exception as e:
            logger.exception(f"视觉分析执行失败: {e}")
            return AgentOutput(
                result=f"视觉分析失败: {str(e)}",
                confidence=0.0
            )
    
    def _auto_find_image(self, context: Dict[str, Any]) -> Optional[str]:
        """自动查找目录中的图片"""
        import os
        from pathlib import Path
        
        # 支持的图片格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
        
        # 搜索路径优先级
        search_paths = []
        
        # 1. 从context中获取可能的目录路径
        global_config = context.get("global_config", {})

        # 2. 构建搜索路径列表
        current_dir = os.getcwd()
        search_paths.extend([
            os.path.join(current_dir, "data", "imgs"),     # images目录
            os.path.join(current_dir, "data", "pics"),     # pics目录
            current_dir,                             # 当前目录
        ])
        
        print(f"🔍 搜索图片路径: {search_paths}")
        
        for search_path in search_paths:
            if not os.path.exists(search_path):
                continue
                
            try:
                # 遍历目录中的文件
                for item in os.listdir(search_path):
                    file_path = os.path.join(search_path, item)
                    
                    # 检查是否为图片文件
                    if os.path.isfile(file_path):
                        file_ext = Path(file_path).suffix.lower()
                        if file_ext in image_extensions:
                            print(f"✓ 找到图片: {file_path}")
                            return file_path
                            
            except Exception as e:
                print(f"⚠️ 搜索路径 {search_path} 时出错: {e}")
                continue
        
        print("❌ 未找到任何图片文件")
        return None
    
    def _prepare_image_content(self, image_data):
        import os
        from urllib.parse import urlparse
        
        if isinstance(image_data, str):
            # 检查是否为 URL
            if image_data.startswith(('http://', 'https://')):
                return image_data
            # 检查是否为本地文件路径
            elif os.path.isfile(image_data):
                return image_data
            # 假设是 base64 字符串
            else:
                return f"data:image/jpeg;base64,{image_data}"
        else:
            # PIL Image 对象，转换为 base64
            from io import BytesIO
            import base64
            
            buffered = BytesIO()
            image_data.save(buffered, format="JPEG")
            img_bytes = buffered.getvalue()
            base64_str = base64.b64encode(img_bytes).decode('utf-8')
            return f"data:image/jpeg;base64,{base64_str}"
