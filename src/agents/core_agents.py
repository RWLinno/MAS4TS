from typing import Any, Dict, List, Optional, Tuple, Union
import asyncio
import logging
import re

from .base import Agent, AgentInput, AgentOutput, AgentConfig, AgentRegistry
from ..utils.model_manager import UnifiedModelManager, ModelRequest

logger = logging.getLogger(__name__)

@AgentRegistry.register()
class RouteAgent(Agent):
    """路由智能体 - 负责分析查询并选择最适合的智能体"""
    
    def __init__(self, config: Optional[Union[Dict[str, Any], AgentConfig]] = None):
        if isinstance(config, dict):
            agent_config = AgentConfig(
                name="route_agent",
                description="路由智能体，负责分析查询并路由到合适的智能体",
                version="1.0.0",
                extra_params=config
            )
        else:
            agent_config = config     
        super().__init__(agent_config)
        self.raw_config = config if isinstance(config, dict) else (config.extra_params if config else {})

    def _detect_modalities(self, query: str, context: Dict[str, Any]) -> Dict[str, bool]:
        """检测输入数据的模态特征"""
        
        # 检测查询中是否提到图像相关内容
        image_keywords = [
            "image", "图像", "图片", "picture", "photo", "照片", "screenshot", "截图", 
            "pic", "图", "视觉", "看", "显示", "展示", "img", "jpeg", "jpg", "png",
            "analyze the image", "explain the picture", "what's in the", "describe the",
            "in the picture", "in the image", "图片中", "图像中", "截图中"
        ]
        
        query_mentions_image = any(keyword in query.lower() for keyword in image_keywords)
        
        # 检测@文档引用语法
        doc_reference_pattern = r'@([a-zA-Z0-9_-]+\.md)'
        has_doc_reference = bool(re.search(doc_reference_pattern, query))
        
        # 网络搜索关键词检测（参考Eigent的搜索触发机制）
        search_keywords = [
            "search", "find", "lookup", "latest", "current", "recent", "update",
            "搜索", "查找", "最新", "当前", "最近", "更新",
            "what is", "how to", "best practice", "documentation", "tutorial",
            "什么是", "如何", "最佳实践", "文档", "教程", "指南"
        ]
        
        # 增强技术问题和故障检测
        tech_problem_keywords = [
            # 技术组件
            "redis", "kafka", "mysql", "api", "接口", "数据库", "缓存", "消息队列",
            "服务", "service", "platform", "平台", "系统", "system",
            
            # 问题类型
            "问题", "故障", "error", "错误", "异常", "exception", "bug", "issue",
            "超时", "timeout", "延迟", "慢", "latency", "slow", "卡", "响应",
            "连接", "connection", "断开", "disconnect", "失败", "fail",
            
            # 解决相关
            "解决", "修复", "fix", "排查", "troubleshoot", "诊断", "怎么办", "如何",
            "方案", "solution", "恢复", "recovery", "处理", "handle"
        ]
        
        # 业务领域关键词
        business_keywords = [
            "推荐", "recommend", "数据", "data", "接口", "interface", 
            "监控", "monitor", "告警", "alert", "日志", "log"
        ]
        
        has_tech_problem = any(keyword in query.lower() for keyword in tech_problem_keywords)
        has_business_context = any(keyword in query.lower() for keyword in business_keywords)
        needs_search = any(keyword in query.lower() for keyword in search_keywords)
        
        modalities = {
            "has_text": bool(query.strip()),
            "has_image": ("image" in context and context["image"] is not None) or query_mentions_image,
            "mentions_image": query_mentions_image,  # 新增：查询中提到图像
            "has_actual_image": "image" in context and context["image"] is not None,  # 实际有图像数据
            "has_doc_reference": has_doc_reference,  # 新增：@文档引用
            "has_tech_problem": has_tech_problem,  # 新增：技术问题
            "has_business_context": has_business_context,  # 新增：业务上下文
            "needs_search": needs_search,  # 新增：需要网络搜索
            "has_metrics": any(word in query.lower() for word in [
                "metrics", "指标", "监控", "图表", "数据", "性能", "延迟", "响应时间", "cpu", "memory", "disk"
            ]),
            "has_logs": any(word in query.lower() for word in [
                "log", "logs", "日志", "报错", "异常", "错误", "警告", "error", "warning", "exception"
            ]),
            "needs_knowledge": any(word in query.lower() for word in [
                "what is", "什么是", "介绍", "概念", "定义", "原理", "历史", "发展"
            ]) and not has_tech_problem,  # 只有纯知识查询才算knowledge
            "needs_retrieval": has_doc_reference or has_tech_problem or has_business_context or any(word in query.lower() for word in [
                "查找", "搜索", "检索", "资料", "文档", "知识", "信息", "详细", "具体", "问题", "故障", "排查"
            ])
        }
        return modalities

    def _select_best_agent(self, modalities: Dict[str, bool]) -> Tuple[str, str, float]:
        """根据模态特征选择最佳智能体，参考Eigent的智能路由策略"""
        
        # 1. 如果需要网络搜索，优先使用SearchAgent（参考Eigent）
        if modalities["needs_search"]:
            return "search_agent", "检测到搜索需求，使用网络搜索智能体获取最新信息", 0.9
        
        # 2. 如果有@文档引用，使用RetrieverAgent
        if modalities["has_doc_reference"]:
            return "retriever_agent", "检测到@文档引用，使用检索增强生成", 0.95
        
        # 3. 如果有图像或查询中提到图像，优先视觉分析
        if modalities["has_image"]:
            if modalities["has_actual_image"]:
                return "visual_analysis_agent", "请求包含图像，需要视觉分析", 0.9
            elif modalities["mentions_image"]:
                return "visual_analysis_agent", "查询提到图像，需要视觉分析（将尝试自动查找图片）", 0.85
        
        # 4. 如果需要检索，使用RetrieverAgent
        if modalities["needs_retrieval"]:
            return "retriever_agent", "需要检索相关文档进行回答", 0.85
        
        # 4. 根据内容类型选择专业智能体
        if modalities["has_metrics"]:
            return "metrics_analysis_agent", "请求涉及指标或监控数据分析", 0.8
        elif modalities["has_logs"]:
            return "log_analysis_agent", "请求涉及日志分析", 0.8
        elif modalities["needs_knowledge"]:
            return "knowledge_agent", "请求需要查询知识库", 0.7
        
        # 5. 复杂查询使用综合智能体
        return "comprehensive_agent", "复杂查询需要综合分析", 0.6

    async def execute(self, input_data: AgentInput) -> AgentOutput:
        query = input_data.query
        context = input_data.context or {}
        
        # 检测模态特征
        modalities = self._detect_modalities(query, context)
        
        # 选择最佳智能体
        selected_agent, reasoning, confidence = self._select_best_agent(modalities)
        
        route_result = {
            "next_agent": selected_agent,
            "reasoning": reasoning,
            "modalities": modalities,
            "confidence": confidence
        }
        
        return AgentOutput(
            result=selected_agent,
            context={"route_info": route_result},
            confidence=confidence
        )

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

@AgentRegistry.register()
class MetricsAnalysisAgent(Agent):
    """指标分析智能体 - 处理与系统指标相关的查询"""
    
    def __init__(self, config: Optional[AgentConfig] = None):
        super().__init__(config)
        self.model_manager = None
    
    def _get_model_manager(self, context: dict) -> UnifiedModelManager:
        """获取模型管理器"""
        if self.model_manager is None:
            model_name = context.get("model", "Qwen/Qwen2.5-VL-7B-Instruct")
            self.model_manager = UnifiedModelManager.from_env(model_name, context)
        return self.model_manager
    
    async def execute(self, input_data: AgentInput) -> AgentOutput:
        """执行指标分析"""
        query = input_data.query
        context = input_data.context or {}
        
        try:
            model_manager = self._get_model_manager(context)
            
            messages = [
                {
                    "role": "system",
                    "content": (
                        "你是一个专门分析监控指标的助手。你可以搜索指标数据，生成图表，并提供分析。"
                        "请根据用户的查询，提供详细的指标分析。如果需要具体的指标数据，请说明需要查询什么样的指标。"
                    )
                },
                {
                    "role": "user",
                    "content": f"请分析以下指标相关的查询：{query}"
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
                    context={"metrics_analysis": response.content},
                    confidence=0.9
                )
            else:
                return AgentOutput(
                    result=f"指标分析失败: {response.error}",
                    confidence=0.0
                )
                
        except Exception as e:
            logger.exception(f"指标分析执行失败: {e}")
            return AgentOutput(
                result=f"指标分析失败: {str(e)}",
                confidence=0.0
            )

@AgentRegistry.register()
class LogAnalysisAgent(Agent):
    """日志分析智能体 - 处理与日志相关的查询"""
    
    def __init__(self, config: Optional[AgentConfig] = None):
        super().__init__(config)
        self.model_manager = None
    
    def _get_model_manager(self, context: dict) -> UnifiedModelManager:
        """获取模型管理器"""
        if self.model_manager is None:
            model_name = context.get("model", "Qwen/Qwen2.5-VL-7B-Instruct")
            self.model_manager = UnifiedModelManager.from_env(model_name, context)
        return self.model_manager
    
    async def execute(self, input_data: AgentInput) -> AgentOutput:
        """执行日志分析"""
        query = input_data.query
        context = input_data.context or {}
        
        try:
            # 获取模型管理器
            model_manager = self._get_model_manager(context)
            
            # 构建系统消息和用户消息
            messages = [
                {
                    "role": "system",
                    "content": (
                        "你是一个专门分析系统日志的助手。你可以搜索日志数据，识别异常模式，并提供分析。"
                        "请根据用户的查询，提供详细的日志分析建议。如果需要具体的日志数据，请说明需要查看什么样的日志。"
                    )
                },
                {
                    "role": "user",
                    "content": f"请分析以下日志相关的查询：{query}"
                }
            ]
            
            # 生成请求
            request = ModelRequest(
                messages=messages,
                max_tokens=512,
                temperature=0.7
            )
            
            # 调用模型管理器
            response = await model_manager.generate(request)
            
            if response.success:
                return AgentOutput(
                    result=response.content,
                    context={"log_analysis": response.content},
                    confidence=0.9
                )
            else:
                return AgentOutput(
                    result=f"日志分析失败: {response.error}",
                    confidence=0.0
                )
                
        except Exception as e:
            logger.exception(f"日志分析执行失败: {e}")
            return AgentOutput(
                result=f"日志分析失败: {str(e)}",
                confidence=0.0
            )

@AgentRegistry.register()
class ComprehensiveAgent(Agent):
    """综合智能体 - 处理需要多种能力协作的复杂查询"""
    
    def __init__(self, config: Optional[AgentConfig] = None):
        super().__init__(config)
        self.model_manager = None
        
        # 依赖的所有智能体
        self.add_dependency("KnowledgeAgent")
        self.add_dependency("LogAnalysisAgent")
        self.add_dependency("MetricsAnalysisAgent")
    
    def _get_model_manager(self, context: dict) -> UnifiedModelManager:
        """获取模型管理器"""
        if self.model_manager is None:
            model_name = context.get("model", "Qwen/Qwen2.5-VL-7B-Instruct")
            self.model_manager = UnifiedModelManager.from_env(model_name, context)
        return self.model_manager

    async def execute(self, input_data: AgentInput) -> AgentOutput:
        """执行综合分析"""
        query = input_data.query
        context = input_data.context or {}
        
        try:
            # 首先尝试直接回答
            model_manager = self._get_model_manager(context)
            
            # 构建综合分析的系统提示
            messages = [
                {
                    "role": "system",
                    "content": (
                        "你是一个综合分析助手，擅长处理复杂的技术问题。"
                        "请综合考虑多个方面（系统监控、日志分析、知识库等）来回答用户的问题。"
                        "如果问题涉及多个技术领域，请提供全面的分析和建议。"
                    )
                },
                {
                    "role": "user",
                    "content": f"请综合分析以下问题：{query}"
                }
            ]
            
            # 检查是否包含图像
            image_data = context.get("image")
            request = ModelRequest(
                messages=messages,
                max_tokens=1024,
                temperature=0.7,
                image=image_data
            )
            
            response = await model_manager.generate(request)
            
            if response.success:
                return AgentOutput(
                    result=response.content,
                    context={
                        "comprehensive_analysis": response.content,
                        "analysis_type": "integrated"
                    },
                    confidence=0.8
                )
            else:
                return AgentOutput(
                    result=f"综合分析失败: {response.error}",
                    confidence=0.0
                )
                
        except Exception as e:
            logger.exception(f"综合分析执行失败: {e}")
            return AgentOutput(
                result=f"综合分析失败: {str(e)}",
                confidence=0.0
            )

@AgentRegistry.register()
class RetrieverAgent(Agent):
    """检索增强生成智能体 - 基于SimpleRAG技术检索相关文档并生成回答"""
    
    def __init__(self, config: Optional[AgentConfig] = None):
        super().__init__(config)
        self.model_manager = None
        self.rag = None
        
        # 初始化SimpleRAG
        self._init_simple_rag()
    
    def _init_simple_rag(self):
        """初始化RAG系统（优先使用数据库集成版本）"""
        try:
            # Try database-integrated RAG first
            from oncall_agent.database.connection_manager import DatabaseManager
            from ..retrieval.database_rag import DatabaseRAGService
            
            # Initialize database manager (will only connect to enabled databases)
            db_config = getattr(self.config, 'extra_params', {})
            if db_config and 'database' in db_config:
                self.db_manager = DatabaseManager(db_config)
                self.rag = DatabaseRAGService(self.db_manager, db_config)
                logger.info("✅ Database-integrated RAG系统初始化成功")
                return
            else:
                logger.info("Database not configured, falling back to SimpleRAG")
        except Exception as e:
            logger.warning(f"Database RAG初始化失败，回退到SimpleRAG: {e}")
        
        # Fallback to SimpleRAG
        try:
            from ..retrieval.simple_rag import SimpleRAG
            self.rag = SimpleRAG()
            logger.info("✅ SimpleRAG系统初始化成功")
        except Exception as e:
            logger.error(f"SimpleRAG初始化失败: {e}")
            self.rag = None
    
    def _get_model_manager(self, context: dict) -> UnifiedModelManager:
        """获取模型管理器"""
        if self.model_manager is None:
            model_name = context.get("model", "Qwen/Qwen2.5-VL-7B-Instruct")
            self.model_manager = UnifiedModelManager.from_env(model_name, context)
        return self.model_manager
    
    def _build_enhanced_prompt(self, query: str, search_results) -> str:
        """构建增强的提示"""
        
        # 检测是否有@文档引用
        referenced_file = None
        if self.rag:
            referenced_file = self.rag.detect_document_reference(query)
        
        if referenced_file and search_results:
            # @文档引用的情况
            doc = search_results[0].document
            prompt = f"""你需要严格基于提供的文档内容回答用户问题，不要添加文档中没有的信息。

【指定文档】{referenced_file}
【文档标题】{doc.title}
【文档分类】{doc.category}

【文档原文内容】
{search_results[0].context}

【用户问题】
{query}

【回答要求】
1. 严格按照上述文档内容回答，不要编造或推测信息
2. 如果文档中有具体的排查思路、故障原因、恢复方案，请按照文档结构组织回答
3. 保持文档中的专业术语和具体步骤
4. 如果文档内容不足以完全回答问题，请明确说明哪些信息文档中没有提及
5. 用清晰的结构组织回答，使用项目符号或编号"""
        
        else:
            # 常规检索的情况
            doc_contexts = []
            for i, result in enumerate(search_results, 1):
                doc_contexts.append(f"""
【文档{i}】{result.document.metadata['filename']} (相关度: {result.score:.2f})
标题: {result.document.title}
分类: {result.document.category}

原文内容:
{result.context}
""")
            
            contexts_text = "\n".join(doc_contexts)
            
            prompt = f"""你需要严格基于以下检索到的文档内容回答用户问题，不要添加文档中没有的信息。

{contexts_text}

【用户问题】
{query}

【回答要求】
1. 严格按照上述文档内容回答，不要编造、推测或补充文档中没有的信息
2. 优先使用相关度最高的文档内容
3. 如果多个文档都有相关信息，请综合使用并标明来源
4. 保持文档中的原始表述和专业术语
5. 如果文档内容不足以完全回答问题，请明确说明
6. 用清晰的结构组织回答"""
        
        return prompt
    
    def _format_final_answer(self, answer: str, search_results) -> str:
        """格式化最终回答，包含文档来源"""
        
        # 提取文档来源信息
        sources = []
        for result in search_results:
            sources.append(f"📄 {result.document.metadata['filename']} (相关度: {result.score:.2f})")
        
        # 构建最终回答
        formatted_answer = f"""{answer}

---
📚 **参考文档来源：**
{chr(10).join(sources)}

💡 **提示：** 此回答基于检索到的相关文档生成，具有较高的准确性和可信度。"""
        
        return formatted_answer
    
    async def _fallback_to_knowledge_agent(self, input_data: AgentInput) -> AgentOutput:
        """回退到知识智能体"""
        try:
            # 创建KnowledgeAgent实例
            knowledge_config = AgentConfig(
                name="knowledge_agent_fallback",
                description="回退知识智能体",
                version="1.0.0",
                extra_params=getattr(self.config, 'extra_params', {})
            )
            
            knowledge_agent = KnowledgeAgent(knowledge_config)
            result = await knowledge_agent.execute(input_data)
            
            # 修改结果，标明是回退模式
            fallback_result = f"""【基于通用知识的回答】

{result.result}

⚠️ **注意：** 由于未找到相关文档或检索系统不可用，此回答基于通用知识生成。如需更准确的信息，请尝试使用@文档名.md的格式引用特定文档。"""
            
            return AgentOutput(
                result=fallback_result,
                context={
                    **result.context,
                    "retrieval_enabled": False,
                    "fallback_mode": True
                },
                confidence=result.confidence * 0.8  # 降低置信度
            )
            
        except Exception as e:
            logger.error(f"回退到知识智能体也失败: {e}")
            return AgentOutput(
                result="抱歉，检索系统和知识系统都暂时不可用。请稍后再试。",
                confidence=0.0
            )

    async def execute(self, input_data: AgentInput) -> AgentOutput:
        """执行检索增强生成"""
        query = input_data.query
        context = input_data.context or {}
        
        try:
            # 检查RAG系统是否可用
            if not self.rag:
                return await self._fallback_to_knowledge_agent(input_data)
            
            # 步骤1: 检索相关文档
            search_results = self.rag.search_documents(query, top_k=3)
            
            if not search_results:
                logger.warning("未检索到相关文档，回退到知识智能体")
                return await self._fallback_to_knowledge_agent(input_data)
            
            # 步骤2: 获取模型管理器
            model_manager = self._get_model_manager(context)
            
            # 步骤3: 构建增强的提示
            enhanced_prompt = self._build_enhanced_prompt(query, search_results)
            
            # 步骤4: 生成回答
            request = ModelRequest(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是一个专业的技术文档助手。你的任务是严格基于提供的文档内容回答用户问题。"
                            "重要原则：只使用文档中明确提到的信息，不要添加、推测或编造任何内容。"
                            "如果文档信息不足，请明确说明，而不是补充其他知识。"
                        )
                    },
                    {
                        "role": "user", 
                        "content": enhanced_prompt
                    }
                ],
                max_tokens=1024,
                temperature=0.1  # 降低温度，减少创造性回答
            )
            
            response = await model_manager.generate(request)
            
            if response.success:
                # 构建回答，包含文档来源信息
                final_answer = self._format_final_answer(response.content, search_results)
                
                return AgentOutput(
                    result=final_answer,
                    context={
                        "retrieval_enabled": True,
                        "documents_found": len(search_results),
                        "document_sources": [r.document.metadata['filename'] for r in search_results],
                        "confidence_boost": True,
                        "search_query": query,
                        "search_scores": [r.score for r in search_results]
                    },
                    confidence=0.95  # 高置信度，因为基于检索到的文档
                )
            else:
                return await self._fallback_to_knowledge_agent(input_data)
                
        except Exception as e:
            logger.exception(f"RetrieverAgent执行失败: {e}")
            return await self._fallback_to_knowledge_agent(input_data)
