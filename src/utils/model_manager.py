import os
from typing import Dict, List, Optional, Any, Union
import logging
from pathlib import Path
from pydantic import BaseModel
import numpy as np

import torch

from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoConfig
from transformers import AutoProcessor, AutoModelForImageTextToText

logger = logging.getLogger(__name__)

# 支持的模型列表 - 按显存需求排序，用于降级策略
SUPPORTED_MODELS = {
    # 通用大模型
    "Qwen/Qwen2.5-VL-7B-Instruct": {
        "type": "vl",
        "description": "通义千问2.5多模态大模型",
        "local_path": "models/Qwen2.5-VL-7B-Instruct",
        "device": "gpu",
        "min_gpu_memory": "14G",
        "memory_mb": 14336,
        "fallback_models": ["Qwen/Qwen2.5-3B-Instruct", "microsoft/DialoGPT-small"]
    },
    "Qwen/Qwen1.5-72B-Chat": {
        "type": "text",
        "description": "通义千问1.5-72B对话模型",
        "local_path": "models/Qwen1.5-72B-Chat",
        "device": "gpu",
        "min_gpu_memory": "72G",
        "memory_mb": 73728,
        "fallback_models": ["Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-3B-Instruct"]
    },
    "BAAI/bge-large-zh-v1.5": {
        "type": "embedding",
        "description": "中文文本向量化模型",
        "local_path": "models/bge-large-zh-v1.5",
        "device": "gpu",
        "min_gpu_memory": "2G",
        "memory_mb": 2048,
        "fallback_models": ["BAAI/bge-base-zh-v1.5", "BAAI/bge-small-zh-v1.5"]
    },
    "BAAI/bge-base-zh-v1.5": {
        "type": "embedding",
        "description": "轻量级中文文本向量化模型",
        "local_path": "models/bge-base-zh-v1.5",
        "device": "gpu",
        "min_gpu_memory": "1G",
        "memory_mb": 1024,
        "fallback_models": ["BAAI/bge-small-zh-v1.5"]
    },
    # 轻量级降级模型
    "Qwen/Qwen2.5-3B-Instruct": {
        "type": "text",
        "description": "轻量级通义千问模型",
        "local_path": "models/Qwen2.5-3B-Instruct",
        "device": "gpu",
        "min_gpu_memory": "3G",
        "memory_mb": 3072,
        "fallback_models": ["microsoft/DialoGPT-small"]
    },
    "BAAI/bge-small-zh-v1.5": {
        "type": "embedding",
        "description": "小型中文文本向量化模型",
        "local_path": "models/bge-small-zh-v1.5",
        "device": "cpu",
        "min_gpu_memory": "512M",
        "memory_mb": 512,
        "fallback_models": []
    },
    "microsoft/DialoGPT-small": {
        "type": "text",
        "description": "小型对话模型",
        "local_path": "models/DialoGPT-small",
        "device": "cpu",
        "min_gpu_memory": "1G",
        "memory_mb": 1024,
        "fallback_models": []
    }
}

class ModelRequest(BaseModel):
    """模型请求"""
    messages: List[Dict[str, Any]]  # 支持复杂消息格式
    max_tokens: int = 512
    temperature: float = 0.7
    image: Optional[str] = None

class ModelResponse(BaseModel):
    """模型响应"""
    content: str
    success: bool = True
    error: Optional[str] = None

class UnifiedModelManager:
    _instance = None
    _initialized = False
    _loaded_models = {}  # 类级别的模型缓存
    _model_lock = None  # 模型加载锁
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device_config: Optional[Dict[str, Any]] = None,
        offline_mode: bool = False
    ):
        """
        初始化模型管理器
        
        Args:
            model_name: 模型名称
            device_config: 设备配置
            offline_mode: 是否使用离线模式
        """
        if self._initialized:
            return
            
        # 初始化线程锁
        import threading
        if self._model_lock is None:
            self._model_lock = threading.Lock()
            
        self.model_name = model_name
        self.device_config = device_config or {"gpu_ids": [0]}
        self.offline_mode = offline_mode
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.mock_mode = False
        self.model_type = "unknown"
        self.current_device = None
        self.available_gpu_memory = 0
        self.degraded_model = None
        self.current_loaded_model = None  # 当前加载的模型名称
        
        # 验证模型名称
        if model_name not in SUPPORTED_MODELS:
            logger.warning(f"未知模型: {model_name}。支持的模型: {list(SUPPORTED_MODELS.keys())}")
        
        self._initialize()
        self._initialized = True
    
    def _get_model_path(self) -> str:
        """获取模型路径"""
        model_info = SUPPORTED_MODELS.get(self.model_name, {})
        
        # 如果设置了环境变量，优先使用环境变量中的路径
        env_model_dir = os.getenv("ONCALL_MODEL_DIR")
        if env_model_dir:
            base_path = Path(env_model_dir)
        else:
            # 否则使用项目根目录下的models目录
            base_path = Path(__file__).parent.parent.parent.parent / "models"
        
        model_path = base_path / model_info.get("local_path", self.model_name.split("/")[-1])
        
        if not model_path.exists() and self.offline_mode:
            logger.warning(
                f"离线模式下未找到本地模型: {model_path}\n"
                f"请下载模型到指定目录，或设置ONCALL_MODEL_DIR环境变量指定模型目录\n"
                f"系统将尝试使用模拟模式继续运行"
            )
            return None  # 返回None表示没有找到模型
        
        return str(model_path if model_path.exists() else self.model_name)
    
    def _get_available_gpu_memory(self) -> int:
        """获取可用GPU显存（MB）"""
        try:
            if torch.cuda.is_available():
                device_id = self.device_config.get("gpu_ids", [0])[0]
                torch.cuda.set_device(device_id)
                
                # 获取GPU总显存和已使用显存
                total_memory = torch.cuda.get_device_properties(device_id).total_memory
                allocated_memory = torch.cuda.memory_allocated(device_id)
                reserved_memory = torch.cuda.memory_reserved(device_id)
                
                # 计算可用显存（留出1GB作为缓冲）
                available_memory = total_memory - max(allocated_memory, reserved_memory) - (1024 * 1024 * 1024)
                available_mb = max(0, available_memory // (1024 * 1024))
                
                logger.info(f"GPU {device_id} 显存状态: 总量={total_memory//1024//1024//1024:.1f}GB, "
                           f"已分配={allocated_memory//1024//1024:.1f}MB, "
                           f"已预留={reserved_memory//1024//1024:.1f}MB, "
                           f"可用={available_mb:.1f}MB")
                
                return int(available_mb)
            else:
                logger.info("GPU不可用，使用CPU模式")
                return 0
        except Exception as e:
            logger.warning(f"GPU显存检测失败: {e}")
            return 0
    
    def _predict_optimal_model(self, available_memory_mb: int) -> Optional[str]:
        """显存预判逻辑：根据可用显存预测最优模型 - 修复模型加载效率问题"""
        logger.info(f"🔮 显存预判: 可用显存 {available_memory_mb}MB")
        
        # 按显存需求排序所有支持的模型
        sorted_models = sorted(
            SUPPORTED_MODELS.items(),
            key=lambda x: x[1].get('memory_mb', 0)
        )
        
        # 找到最适合的模型（显存需求不超过可用显存的80%，留出安全边际）
        safe_memory = int(available_memory_mb * 0.8)
        optimal_model = None
        
        for model_name, model_info in sorted_models:
            required_memory = model_info.get('memory_mb', 0)
            if required_memory <= safe_memory:
                optimal_model = model_name
                logger.debug(f"✓ 候选模型: {model_name} (需求: {required_memory}MB)")
            else:
                logger.debug(f"✗ 跳过模型: {model_name} (需求: {required_memory}MB > 安全阈值: {safe_memory}MB)")
        
        if optimal_model:
            optimal_info = SUPPORTED_MODELS[optimal_model]
            logger.info(f"🎯 预判最优模型: {optimal_model} (需求: {optimal_info.get('memory_mb', 0)}MB)")
        else:
            logger.warning(f"⚠️ 无合适模型，可用显存: {available_memory_mb}MB")
        
        return optimal_model
    

    def _get_fallback_model(self, required_memory_mb: int) -> Optional[str]:
        """根据可用显存获取合适的降级模型"""
        current_model_info = SUPPORTED_MODELS.get(self.model_name, {})
        fallback_models = current_model_info.get("fallback_models", [])
        
        # 首先检查当前模型是否适合
        current_memory_req = current_model_info.get("memory_mb", 0)
        if required_memory_mb >= current_memory_req:
            return self.model_name
        
        # 检查降级模型
        for fallback_model in fallback_models:
            if fallback_model in SUPPORTED_MODELS:
                fallback_info = SUPPORTED_MODELS[fallback_model]
                fallback_memory_req = fallback_info.get("memory_mb", 0)
                if required_memory_mb >= fallback_memory_req:
                    logger.info(f"显存不足，从 {self.model_name} 降级到 {fallback_model}")
                    logger.info(f"需求显存: {fallback_memory_req}MB, 可用显存: {required_memory_mb}MB")
                    return fallback_model
        
        # 如果所有降级模型都不适合，返回最小的模型或CPU模式
        logger.warning(f"显存严重不足（可用: {required_memory_mb}MB），将使用CPU模式或模拟模式")
        return None

    
    def _initialize(self) -> None:
        """初始化模型和分词器 - 修复模型加载效率问题"""
        try:
            print("初始化模型和分词器")
            
            # 检测GPU可用性和显存
            self.available_gpu_memory = self._get_available_gpu_memory()
            
            # 设置设备
            if torch.cuda.is_available() and self.available_gpu_memory > 0:
                device_ids = self.device_config.get("gpu_ids", [0])
                device_id = device_ids[0]
                self.current_device = f"cuda:{device_id}"
                torch.cuda.set_device(device_id)
                logger.info(f"使用GPU设备: {self.current_device}, 可用显存: {self.available_gpu_memory}MB")
            else:
                self.current_device = "cpu"
                logger.info("使用CPU模式")
            
            # 显存预判逻辑：根据剩余显存直接选择合适模型
            if self.current_device.startswith("cuda"):
                optimal_model = self._predict_optimal_model(self.available_gpu_memory)
                if optimal_model and optimal_model != self.model_name:
                    logger.warning(f"⚙️ 显存预判优化: 原模型 {self.model_name} 需要 {SUPPORTED_MODELS.get(self.model_name, {}).get('memory_mb', 0)}MB 显存")
                    logger.warning(f"⚙️ 当前可用显存: {self.available_gpu_memory}MB，预判最优模型: {optimal_model}")
                    self.degraded_model = self.model_name
                    self.model_name = optimal_model
                elif optimal_model is None:
                    logger.warning(f"⚠️ 显存严重不足！原模型 {self.model_name} 需要 {SUPPORTED_MODELS.get(self.model_name, {}).get('memory_mb', 0)}MB")
                    logger.warning(f"⚠️ 当前可用显存: {self.available_gpu_memory}MB，将使用CPU模式")
                    self.current_device = "cpu"
            
            # 不在初始化时加载模型，采用懒加载策略
            logger.info("✨ 模型管理器初始化完成，采用串行懒加载策略")
            logger.info(f"📊 GPU显存状态: 可用{self.available_gpu_memory}MB")
            
        except Exception as e:
            logger.error(f"模型初始化失败: {e}")
            self.model = None
            self.tokenizer = None
            self.mock_mode = True
    
    def _release_current_model(self) -> None:
        """释放当前加载的模型，清理GPU显存"""
        try:
            if self.model is not None:
                logger.info(f"🧹 释放模型: {self.current_loaded_model}")
                
                # 将模型移到CPU并删除引用
                if hasattr(self.model, 'to'):
                    self.model = self.model.to('cpu')
                del self.model
                self.model = None
                
                # 释放tokenizer和processor
                if self.tokenizer is not None:
                    del self.tokenizer
                    self.tokenizer = None
                    
                if self.processor is not None:
                    del self.processor
                    self.processor = None
                
                # 清理GPU缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                # 更新状态
                self.current_loaded_model = None
                self.model_type = "unknown"
                
                # 重新检测可用显存
                self.available_gpu_memory = self._get_available_gpu_memory()
                logger.info(f"✅ 模型释放完成，可用显存: {self.available_gpu_memory}MB")
                
        except Exception as e:
            logger.warning(f"模型释放过程中出现警告: {e}")
    
    def _ensure_model_loaded(self, required_model: str) -> bool:
        """确保指定模型已加载（串行加载策略）"""
        with self._model_lock:
            try:
                # 如果当前已加载所需模型，直接返回
                if (self.current_loaded_model == required_model and 
                    self.model is not None and not self.mock_mode):
                    logger.debug(f"✅ 模型 {required_model} 已加载")
                    return True
                
                # 如果加载了其他模型，先释放
                if self.current_loaded_model and self.current_loaded_model != required_model:
                    logger.info(f"🔄 切换模型: {self.current_loaded_model} -> {required_model}")
                    self._release_current_model()
                
                # 加载新模型
                logger.info(f"⏳ 串行加载模型: {required_model}")
                
                # 临时切换模型名称
                original_model_name = self.model_name
                self.model_name = required_model
                
                # 重新检测可用显存
                self.available_gpu_memory = self._get_available_gpu_memory()
                
                # 检查是否需要降级
                if self.current_device.startswith("cuda"):
                    fallback_model = self._get_fallback_model(self.available_gpu_memory)
                    if fallback_model and fallback_model != required_model:
                        logger.warning(f"⚙️ 显存不足，从 {required_model} 降级到 {fallback_model}")
                        self.model_name = fallback_model
                        required_model = fallback_model
                    elif fallback_model is None:
                        logger.warning(f"⚠️ 显存不足，切换到CPU模式")
                        self.current_device = "cpu"
                
                # 尝试加载模型
                success = False
                is_vl_model = "VL" in self.model_name or "vision" in self.model_name.lower()
                
                if is_vl_model:
                    success = self._load_vl_model_with_fallback()
                else:
                    success = self._load_standard_model_with_fallback()
                
                if success:
                    self.current_loaded_model = self.model_name
                    logger.info(f"✅ 模型 {self.model_name} 加载成功")
                    return True
                else:
                    logger.warning(f"❌ 模型 {required_model} 加载失败，使用模拟模式")
                    self.mock_mode = True
                    self.current_loaded_model = required_model
                    return False
                    
            except Exception as e:
                logger.error(f"串行模型加载失败: {e}")
                self.mock_mode = True
                return False
            finally:
                # 恢复原始模型名称（如果需要）
                if 'original_model_name' in locals():
                    self.model_name = original_model_name
    
    def _load_vl_model_with_fallback(self) -> bool:
        """带有降级策略的VL模型加载"""
        try:
            return self._load_vl_model(self.current_device)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) or "out of memory" in str(e).lower():
                logger.error(f"⚠️ CUDA显存不足错误: {e}")
                return self._handle_oom_fallback()
            else:
                logger.error(f"VL模型加载失败: {e}")
                return False
        except Exception as e:
            logger.error(f"VL模型加载失败: {e}")
            return False
    
    def _load_standard_model_with_fallback(self) -> bool:
        """带有降级策略的标准模型加载"""
        try:
            return self._load_standard_model()
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) or "out of memory" in str(e).lower():
                logger.error(f"⚠️ CUDA显存不足错误: {e}")
                return self._handle_oom_fallback()
            else:
                logger.error(f"模型加载失败: {e}")
                return False
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return False
    
    def _handle_oom_fallback(self) -> bool:
        """处理显存不足的降级策略"""
        logger.warning(f"🔄 正在尝试显存不足降级策略...")
        
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("🧹 已清理GPU缓存")
        
        # 重新检测可用显存
        self.available_gpu_memory = self._get_available_gpu_memory()
        
        # 尝试降级模型
        current_model_info = SUPPORTED_MODELS.get(self.model_name, {})
        fallback_models = current_model_info.get("fallback_models", [])
        
        for fallback_model in fallback_models:
            if fallback_model in SUPPORTED_MODELS:
                fallback_info = SUPPORTED_MODELS[fallback_model]
                fallback_memory_req = fallback_info.get("memory_mb", 0)
                
                if self.available_gpu_memory >= fallback_memory_req:
                    logger.warning(f"🔄 尝试降级到模型: {fallback_model} (需求显存: {fallback_memory_req}MB)")
                    
                    # 保存原始模型名称
                    if not self.degraded_model:
                        self.degraded_model = self.model_name
                    
                    # 切换到降级模型
                    self.model_name = fallback_model
                    
                    # 尝试加载降级模型
                    try:
                        if "VL" in fallback_model or "vision" in fallback_model.lower():
                            success = self._load_vl_model(self.current_device)
                        else:
                            success = self._load_standard_model()
                        
                        if success:
                            logger.warning(f"✅ 降级成功！当前使用模型: {fallback_model}")
                            return True
                    except Exception as e:
                        logger.warning(f"降级模型 {fallback_model} 加载也失败: {e}")
                        continue
        
        # 如果所有降级模型都失败，尝试CPU模式
        logger.warning(f"💻 所有GPU模型都无法加载，尝试CPU模式")
        self.current_device = "cpu"
        
        try:
            if self._load_standard_model():
                logger.warning(f"✅ CPU模式加载成功！")
                return True
        except Exception as e:
            logger.error(f"CPU模式也失败: {e}")
        
        # 最后降级到模拟模式
        logger.warning(f"🎭 所有加载尝试都失败，使用模拟模式")
        self.mock_mode = True
        return False
    
    def _load_standard_model(self) -> bool:
        """加载标准模型"""
        model_path = self._get_model_path()
        
        # 如果模型路径为None，说明离线模式下找不到本地模型
        if model_path is None:
            logger.warning(f"模型 {self.model_name} 无法在离线模式下加载")
            return False
        
        # 加载分词器
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=self.offline_mode
            )
            logger.info(f"成功加载分词器: {model_path}")
        except Exception as e:
            logger.error(f"分词器加载失败: {e}")
            return False
        
        # 加载模型 - 尝试多种加载方式
        try:
            # 首先尝试使用AutoModelForCausalLM
            load_kwargs = {
                "trust_remote_code": True,
                "local_files_only": self.offline_mode
            }
            
            # 根据设备设置加载参数
            if self.current_device.startswith("cuda"):
                load_kwargs["device_map"] = self.current_device
                load_kwargs["torch_dtype"] = torch.float16  # 使用半精度节省显存
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **load_kwargs
            )
            self.model_type = "causal_lm"
            logger.info(f"成功使用AutoModelForCausalLM加载模型: {self.model_name}")
            
        except Exception as e:
            logger.warning(f"AutoModelForCausalLM加载失败: {e}")
            try:
                # 尝试使用AutoModel
                load_kwargs = {
                    "trust_remote_code": True,
                    "local_files_only": self.offline_mode
                }
                
                if self.current_device.startswith("cuda"):
                    load_kwargs["device_map"] = self.current_device
                    load_kwargs["torch_dtype"] = torch.float16
                
                self.model = AutoModel.from_pretrained(
                    model_path,
                    **load_kwargs
                )
                self.model_type = "base_model"
                logger.info(f"成功使用AutoModel加载模型: {self.model_name}")
                
            except Exception as e2:
                logger.error(f"所有加载方式都失败: {e2}")
                raise e2  # 重新抛出异常以便上层处理
        
        # 如果成功加载且使用CPU模式，将模型移动到CPU
        if self.model and self.current_device == "cpu":
            self.model = self.model.to("cpu")
        
        if self.model:
            self.mock_mode = False
            logger.info(f"模型 {self.model_name} 加载完成，使用设备: {self.current_device}")
            return True
        
        return False
    
    def _load_vl_model(self, device: str) -> bool:
        """专门用于加载VL模型的方法"""
        try:
            # 首先尝试在线加载processor和模型
            if not self.offline_mode:
                logger.info("在线模式加载VL模型")
                
                # 渐进式加载策略：先加载processor，再加载模型
                try:
                    logger.info(f"步骤1: 加载processor for {self.model_name}")
                    # 检查是否为测试环境或资源受限环境
                    import psutil
                    available_memory = psutil.virtual_memory().available // (1024**3)  # GB
                    
                    if available_memory < 8:  # 如果可用内存小于8GB
                        logger.warning(f"⚠️ 可用内存不足 ({available_memory}GB < 8GB)，跳过大模型加载，使用模拟模式")
                        self.mock_mode = True
                        return False
                    
                    self.processor = AutoProcessor.from_pretrained(
                        self.model_name,
                        trust_remote_code=True
                    )
                    logger.info("✅ Processor加载成功")
                except ImportError:
                    # 如果没有psutil，继续正常加载但添加警告
                    logger.warning("psutil未安装，无法检测内存状态，继续尝试加载模型")
                    try:
                        self.processor = AutoProcessor.from_pretrained(
                            self.model_name,
                            trust_remote_code=True
                        )
                        logger.info("✅ Processor加载成功")
                    except Exception as e:
                        logger.error(f"Processor加载失败: {e}")
                        return False
                except Exception as e:
                    logger.error(f"Processor加载失败: {e}")
                    return False
                
                # 尝试多种VL模型加载方式
                load_kwargs = {
                    "trust_remote_code": True,
                    "torch_dtype": torch.float16,  # 使用半精度节省显存
                    "low_cpu_mem_usage": True  # 优化CPU内存使用
                }
                
                # 根据设备设置加载参数
                if device.startswith("cuda"):
                    load_kwargs["device_map"] = device
                
                try:
                    logger.info(f"步骤2: 尝试使用AutoModelForImageTextToText加载 {self.model_name}")
                    # 方式1: AutoModelForImageTextToText
                    self.model = AutoModelForImageTextToText.from_pretrained(
                        self.model_name,
                        **load_kwargs
                    )
                    self.model_type = "vl_model"
                    logger.info("✅ 成功使用AutoModelForImageTextToText加载VL模型")
                    return True
                    
                except Exception as e1:
                    logger.warning(f"AutoModelForImageTextToText加载失败: {e1}")
                    
                    # 如果是显存不足错误，直接抛出
                    if "CUDA out of memory" in str(e1) or "out of memory" in str(e1).lower():
                        raise e1
                    
                    try:
                        logger.info(f"步骤3: 尝试使用AutoModelForCausalLM加载 {self.model_name}")
                        # 方式2: AutoModelForCausalLM
                        self.model = AutoModelForCausalLM.from_pretrained(
                            self.model_name,
                            **load_kwargs
                        )
                        self.model_type = "vl_causal_lm"
                        logger.info("✅ 成功使用AutoModelForCausalLM加载VL模型")
                        return True
                        
                    except Exception as e2:
                        logger.error(f"所有VL模型加载方式都失败: {e2}")
                        
                        # 如果是显存不足错误，抛出以便上层处理
                        if "CUDA out of memory" in str(e2) or "out of memory" in str(e2).lower():
                            raise e2
                        
                        return False
            else:
                # 离线模式下，VL模型比较复杂，直接进入模拟模式
                logger.warning("离线模式下暂不支持VL模型加载，使用模拟模式")
                return False
                
        except Exception as e:
            logger.error(f"VL模型加载过程出错: {e}")
            # 如果是显存不足错误，抛出以便上层处理
            if "CUDA out of memory" in str(e) or "out of memory" in str(e).lower():
                raise e
            return False
    
    async def generate(self, request: ModelRequest) -> ModelResponse:
        """
        生成回答
        
        Args:
            request: 请求参数
        
        Returns:
            生成的回答
        """
        try:
            # 串行加载策略：确保所需模型已加载
            if not self._ensure_model_loaded(self.model_name):
                # 如果模型加载失败，使用模拟模式
                query_content = self._extract_query_content(request.messages)
                return ModelResponse(
                    content=f"模拟模式回答：基于查询 '{query_content[:50]}...'，建议升级环境以使用真实模型。",
                    success=True
                )
            
            # 模拟模式简单处理
            if hasattr(self, 'mock_mode') and self.mock_mode:
                query_content = self._extract_query_content(request.messages)
                return ModelResponse(
                    content=f"模拟模式回答：基于查询 '{query_content[:50]}...'，建议升级环境以使用真实模型。",
                    success=True
                )
            
            if not self.model:
                raise RuntimeError("模型未正确初始化")
            
            # 根据模型类型选择生成方式
            if self.model_type == "vl_model" and hasattr(self, 'processor') and self.processor:
                return await self._generate_with_vl_processor(request)
            elif self.model_type in ["causal_lm", "vl_causal_lm"] and hasattr(self.model, 'generate'):
                return await self._generate_with_causal_lm(request)
            elif hasattr(self, 'tokenizer') and self.tokenizer:
                return await self._generate_with_base_model(request)
            else:
                raise RuntimeError("缺少合适的生成方法")
            
        except Exception as e:
            logger.error(f"生成失败: {e}")
            return ModelResponse(
                content="",
                success=False,
                error=str(e)
            )
    
    def release_model_after_use(self) -> None:
        """任务完成后释放模型，优化显存使用"""
        try:
            if self.current_loaded_model:
                logger.info(f"🎆 任务完成，自动释放模型: {self.current_loaded_model}")
                self._release_current_model()
        except Exception as e:
            logger.warning(f"任务完成后模型释放失败: {e}")
    
    @classmethod
    def get_gpu_memory_status(cls) -> Dict[str, Any]:
        """获取GPU显存状态信息"""
        try:
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                status = {
                    "gpu_available": True,
                    "device_count": device_count,
                    "devices": []
                }
                
                for i in range(device_count):
                    props = torch.cuda.get_device_properties(i)
                    total_memory = props.total_memory
                    allocated_memory = torch.cuda.memory_allocated(i)
                    reserved_memory = torch.cuda.memory_reserved(i)
                    free_memory = total_memory - max(allocated_memory, reserved_memory)
                    
                    device_info = {
                        "device_id": i,
                        "name": props.name,
                        "total_memory_gb": round(total_memory / (1024**3), 2),
                        "allocated_memory_mb": round(allocated_memory / (1024**2), 2),
                        "reserved_memory_mb": round(reserved_memory / (1024**2), 2),
                        "free_memory_mb": round(free_memory / (1024**2), 2),
                        "utilization_percent": round((allocated_memory / total_memory) * 100, 2)
                    }
                    status["devices"].append(device_info)
                
                return status
            else:
                return {
                    "gpu_available": False,
                    "message": "CUDA不可用或未安装"
                }
        except Exception as e:
            return {
                "gpu_available": False,
                "error": str(e)
            }
    
    async def _generate_with_vl_processor(self, request: ModelRequest) -> ModelResponse:
        """使用VL专用processor生成响应"""
        try:
            # 提取文本内容
            text_content = self._extract_query_content(request.messages)
            image_content = None
            
            # 处理图像
            if request.image:
                image_content = await self._load_image(request.image)
                if image_content:
                    logger.info(f"成功加载图像: {image_content.size}")
            
            # 为Qwen2.5-VL构建正确的输入格式
            if image_content:
                # 使用processor的chat模板功能
                try:
                    # 构建标准的对话消息
                    conversation = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image_content},
                                {"type": "text", "text": text_content}
                            ]
                        }
                    ]
                    
                    # 使用apply_chat_template处理对话
                    if hasattr(self.processor.tokenizer, 'apply_chat_template'):
                        text_prompt = self.processor.tokenizer.apply_chat_template(
                            conversation, 
                            tokenize=False, 
                            add_generation_prompt=True
                        )
                    else:
                        # 回退到简单的文本处理
                        text_prompt = f"<|im_start|>user\n{text_content}<|im_end|>\n<|im_start|>assistant\n"
                    
                    # 使用processor处理图像和文本
                    inputs = self.processor(
                        text=text_prompt,
                        images=image_content,
                        return_tensors="pt"
                    ).to(self.model.device)
                    
                    logger.info("VL多模态输入：使用chat模板格式")
                    
                except Exception as e:
                    logger.warning(f"Chat模板处理失败，使用简化格式: {e}")
                    # 回退到简化处理
                    inputs = self.processor(
                        text=text_content,
                        images=image_content,
                        return_tensors="pt"
                    ).to(self.model.device)
                    logger.info("VL多模态输入：使用简化格式")
            else:
                # 纯文本输入
                inputs = self.processor(
                    text=text_content,
                    return_tensors="pt"
                ).to(self.model.device)
                logger.info("VL纯文本输入")
            
            # 生成回答
            with torch.no_grad():
                if hasattr(self.model, 'generate'):
                    # 使用较小的参数避免token不匹配问题
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=min(request.max_tokens, 256),  # 限制生成长度
                        temperature=request.temperature,
                        do_sample=True if request.temperature > 0 else False,
                        pad_token_id=getattr(self.processor.tokenizer, 'eos_token_id', None),
                        use_cache=True  # 启用缓存
                    )
                    
                    # 解码输出
                    if hasattr(self.processor, 'batch_decode'):
                        # 使用batch_decode处理整个输出
                        full_response = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
                        # 提取生成的部分（去掉输入部分）
                        input_text = self.processor.batch_decode(inputs.input_ids, skip_special_tokens=True)[0]
                        if input_text in full_response:
                            response = full_response[len(input_text):].strip()
                        else:
                            response = full_response
                    else:
                        # 传统方式：只解码新生成的token
                        generated_ids = outputs[0][len(inputs.input_ids[0]):]
                        response = self.processor.decode(generated_ids, skip_special_tokens=True)
                    
                    return ModelResponse(
                        content=response.strip(),
                        success=True
                    )
                else:
                    return ModelResponse(
                        content="VL模型加载成功，但缺少generate方法。请检查模型版本。",
                        success=False,
                        error="Missing generate method"
                    )
            
        except Exception as e:
            logger.error(f"VL Processor生成失败: {e}")
            # 如果是token不匹配错误，尝试降级处理
            if "features and image tokens do not match" in str(e):
                logger.info("检测到token不匹配错误，尝试简化处理...")
                return await self._fallback_vl_generation(request)
            
            return ModelResponse(
                content="",
                success=False,
                error=str(e)
            )
    
    async def _fallback_vl_generation(self, request: ModelRequest) -> ModelResponse:
        """VL模型的回退生成方法"""
        try:
            # 提取查询文本
            text_content = self._extract_query_content(request.messages)
            
            # 如果有图像，尝试描述图像
            image_description = ""
            if request.image:
                image_content = await self._load_image(request.image)
                if image_content:
                    image_description = f"（图像尺寸: {image_content.size}，格式: {image_content.format}）"
            
            # 返回描述性回答
            fallback_content = f"""图像分析请求已接收。{image_description}

查询: {text_content}

由于当前模型配置限制，无法直接分析图像内容。建议：
1. 检查transformers版本是否为最新
2. 确认Qwen2.5-VL模型配置正确
3. 或者提供图像的文字描述以便分析

模拟分析：如果图像包含技术问题、错误信息或需要解释的内容，请详细描述图像中的文字和关键信息，我将基于描述提供帮助。"""
            
            return ModelResponse(
                content=fallback_content,
                success=True
            )
            
        except Exception as e:
            logger.error(f"回退生成也失败: {e}")
            return ModelResponse(
                content=f"图像分析功能暂时不可用: {str(e)}",
                success=False,
                error=str(e)
            )
    
    async def _generate_with_causal_lm(self, request: ModelRequest) -> ModelResponse:
        """使用CausalLM模型生成响应"""
        try:
            messages = request.messages
            
            # 使用tokenizer或processor处理输入
            if hasattr(self, 'processor') and self.processor:
                # 对于VL CausalLM，使用processor
                text_content = self._extract_query_content(messages)
                image_content = None
                
                if request.image:
                    image_content = await self._load_image(request.image)
                
                if image_content:
                    inputs = self.processor(
                        text=text_content,
                        images=image_content,
                        return_tensors="pt"
                    ).to(self.model.device)
                else:
                    inputs = self.processor(
                        text=text_content,
                        return_tensors="pt"
                    ).to(self.model.device)
            else:
                # 对于纯文本CausalLM，使用tokenizer
                if hasattr(self.tokenizer, "apply_chat_template"):
                    prompt = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                else:
                    prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
                
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048
                ).to(self.model.device)
            
            # 生成回答
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature,
                    do_sample=True if request.temperature > 0 else False,
                    pad_token_id=getattr(self.tokenizer, 'eos_token_id', None) or getattr(self.processor.tokenizer, 'eos_token_id', None) if hasattr(self, 'processor') else None
                )
            
            # 解码输出
            if hasattr(self, 'processor') and self.processor:
                generated_ids = outputs[0][len(inputs.input_ids[0]):]
                response = self.processor.decode(generated_ids, skip_special_tokens=True)
            else:
                response = self.tokenizer.decode(
                    outputs[0][len(inputs.input_ids[0]):],
                    skip_special_tokens=True
                )
            
            return ModelResponse(
                content=response.strip(),
                success=True
            )
            
        except Exception as e:
            logger.error(f"CausalLM生成失败: {e}")
            return ModelResponse(
                content="",
                success=False,
                error=str(e)
            )
    
    async def _generate_with_base_model(self, request: ModelRequest) -> ModelResponse:
        """使用基础模型生成响应（没有generate方法的模型）"""
        try:
            # 对于没有generate方法的模型，我们只能返回一个说明
            logger.warning("基础模型没有generate方法，无法生成回答")
            
            query_content = self._extract_query_content(request.messages)
            
            return ModelResponse(
                content=f"模型已加载但缺少generate方法。无法对查询 '{query_content[:50]}...' 生成回答。请使用支持生成的模型版本。",
                success=False,
                error="Model does not support generation"
            )
            
        except Exception as e:
            logger.error(f"基础模型处理失败: {e}")
            return ModelResponse(
                content="",
                success=False,
                error=str(e)
            )
    
    async def _load_image(self, image_source):
        """
        加载图像，支持多种输入格式：
        - URL (http/https)
        - 本地文件路径
        - base64 字符串 (纯 base64 或 data:image/xxx;base64,...)
        - PIL Image 对象
        """
        try:
            from PIL import Image
            import base64
            from io import BytesIO
            import os
            
            if image_source is None:
                return None
            
            # 如果已经是 PIL Image 对象
            if hasattr(image_source, 'save'):
                return image_source.convert('RGB')
            
            if isinstance(image_source, str):
                # URL 图像
                if image_source.startswith(('http://', 'https://')):
                    try:
                        import requests
                        response = requests.get(image_source, timeout=10)
                        response.raise_for_status()
                        image = Image.open(BytesIO(response.content)).convert('RGB')
                        logger.info(f"成功从 URL 加载图像: {image_source}")
                        return image
                    except Exception as e:
                        logger.error(f"URL 图像加载失败 {image_source}: {e}")
                        return None
                
                # 本地文件路径
                elif os.path.isfile(image_source):
                    try:
                        image = Image.open(image_source).convert('RGB')
                        logger.info(f"成功从本地文件加载图像: {image_source}")
                        return image
                    except Exception as e:
                        logger.error(f"本地图像文件加载失败 {image_source}: {e}")
                        return None
                
                # Base64 字符串
                else:
                    try:
                        # 处理 data:image/xxx;base64,... 格式
                        if image_source.startswith('data:image'):
                            base64_data = image_source.split(',')[1] if ',' in image_source else image_source
                        else:
                            base64_data = image_source
                        
                        # 解码 base64
                        img_data = base64.b64decode(base64_data)
                        image = Image.open(BytesIO(img_data)).convert('RGB')
                        logger.info("成功从 base64 字符串加载图像")
                        return image
                    except Exception as e:
                        logger.error(f"Base64 图像解码失败: {e}")
                        return None
            
            logger.warning(f"不支持的图像格式: {type(image_source)}")
            return None
            
        except Exception as e:
            logger.error(f"图像加载过程出错: {e}")
            return None
    
    @classmethod
    def from_env(cls, model_name: str, context: Dict[str, Any]) -> "UnifiedModelManager":
        """
        从环境配置创建模型管理器
        
        Args:
            model_name: 模型名称
            context: 上下文配置
        
        Returns:
            模型管理器实例
        """
        # 获取设备配置
        device_config = context.get("device_config", {"gpu_ids": [0]})
        
        # 获取离线模式设置
        offline_mode = context.get("offline_mode", False)
        
        return cls(model_name, device_config, offline_mode)
    
    def __del__(self):
        """清理资源"""
        if self.model:
            try:
                del self.model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"清理模型资源时出错: {e}")
    
    @staticmethod
    def list_supported_models() -> str:
        """
        获取支持的模型列表
        
        Returns:
            格式化的模型列表字符串
        """
        result = "支持的模型列表：\n\n"
        result += "| 模型名称 | 类型 | 描述 | 最小GPU内存 |\n"
        result += "|----------|------|------|-------------|\n"
        
        for name, info in SUPPORTED_MODELS.items():
            result += f"| {name} | {info['type']} | {info['description']} | {info['min_gpu_memory']} |\n"
        
        return result
    
    def _extract_query_content(self, messages):
        """从消息中提取查询内容，支持复杂格式"""
        try:
            if not messages:
                return ""
            
            last_message = messages[-1]
            content = last_message.get('content', '')
            
            # 如果content是字符串，直接返回
            if isinstance(content, str):
                return content
            
            # 如果content是列表（官方格式），提取文本部分
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        text_parts.append(item.get('text', ''))
                return ' '.join(text_parts)
            
            return str(content)
        except Exception:
            return "" 