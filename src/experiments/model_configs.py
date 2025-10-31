#!/usr/bin/env python3
"""
支持的开源模型配置
为不同的开源模型提供适配的提示词格式
"""

from typing import Dict, Any, Callable

class ModelConfig:
    """模型配置类"""
    
    def __init__(self, name: str, prompt_template: Callable[[str], str], 
                 generation_params: Dict[str, Any]):
        self.name = name
        self.prompt_template = prompt_template
        self.generation_params = generation_params

def qwen_prompt_template(content: str) -> str:
    """Qwen系列模型的提示词模板"""
    return f"""<|im_start|>system
你是一个专业的DevOps工程师和技术文档专家。你需要基于给定的OnCall技术文档生成真实的问答对。
<|im_end|>
<|im_start|>user
{content}
<|im_end|>
<|im_start|>assistant"""

def llama_prompt_template(content: str) -> str:
    """Llama系列模型的提示词模板"""
    return f"""<s>[INST] <<SYS>>
你是一个专业的DevOps工程师和技术文档专家。你需要基于给定的OnCall技术文档生成真实的问答对。
<</SYS>>

{content} [/INST]"""

def chatglm_prompt_template(content: str) -> str:
    """ChatGLM系列模型的提示词模板"""
    return f"""[Round 1]

问：{content}

答："""

def baichuan_prompt_template(content: str) -> str:
    """Baichuan系列模型的提示词模板"""
    return f"""<reserved_106>{content}<reserved_107>"""

def internlm_prompt_template(content: str) -> str:
    """InternLM系列模型的提示词模板"""
    return f"""<|User|>:{content}
<|Bot|>:"""

# 支持的模型配置
SUPPORTED_MODELS = {
    # Qwen系列
    "Qwen/Qwen2.5-7B-Instruct": ModelConfig(
        name="Qwen2.5-7B-Instruct",
        prompt_template=qwen_prompt_template,
        generation_params={
            "max_new_tokens": 2048,
            "temperature": 0.7,
            "do_sample": True,
            "top_p": 0.9,
            "repetition_penalty": 1.1
        }
    ),
    "Qwen/Qwen2.5-14B-Instruct": ModelConfig(
        name="Qwen2.5-14B-Instruct", 
        prompt_template=qwen_prompt_template,
        generation_params={
            "max_new_tokens": 2048,
            "temperature": 0.7,
            "do_sample": True,
            "top_p": 0.9,
            "repetition_penalty": 1.1
        }
    ),
    
    # Llama系列
    "meta-llama/Llama-2-7b-chat-hf": ModelConfig(
        name="Llama-2-7b-chat",
        prompt_template=llama_prompt_template,
        generation_params={
            "max_new_tokens": 2048,
            "temperature": 0.7,
            "do_sample": True,
            "top_p": 0.9,
            "repetition_penalty": 1.1
        }
    ),
    "meta-llama/Llama-2-13b-chat-hf": ModelConfig(
        name="Llama-2-13b-chat",
        prompt_template=llama_prompt_template,
        generation_params={
            "max_new_tokens": 2048,
            "temperature": 0.7,
            "do_sample": True,
            "top_p": 0.9,
            "repetition_penalty": 1.1
        }
    ),
    
    # ChatGLM系列
    "THUDM/chatglm3-6b": ModelConfig(
        name="ChatGLM3-6B",
        prompt_template=chatglm_prompt_template,
        generation_params={
            "max_new_tokens": 2048,
            "temperature": 0.7,
            "do_sample": True,
            "top_p": 0.9
        }
    ),
    
    # Baichuan系列
    "baichuan-inc/Baichuan2-7B-Chat": ModelConfig(
        name="Baichuan2-7B-Chat",
        prompt_template=baichuan_prompt_template,
        generation_params={
            "max_new_tokens": 2048,
            "temperature": 0.7,
            "do_sample": True,
            "top_p": 0.9,
            "repetition_penalty": 1.05
        }
    ),
    
    # InternLM系列
    "internlm/internlm2-chat-7b": ModelConfig(
        name="InternLM2-Chat-7B",
        prompt_template=internlm_prompt_template,
        generation_params={
            "max_new_tokens": 2048,
            "temperature": 0.7,
            "do_sample": True,
            "top_p": 0.9,
            "repetition_penalty": 1.1
        }
    )
}

def get_model_config(model_name: str) -> ModelConfig:
    """获取模型配置"""
    if model_name in SUPPORTED_MODELS:
        return SUPPORTED_MODELS[model_name]
    else:
        # 默认使用Qwen格式
        print(f"警告: 未找到模型 {model_name} 的配置，使用默认Qwen格式")
        return SUPPORTED_MODELS["Qwen/Qwen2.5-7B-Instruct"]

def list_supported_models() -> list:
    """列出所有支持的模型"""
    return list(SUPPORTED_MODELS.keys())

def get_model_info():
    """获取模型信息"""
    info = []
    for model_path, config in SUPPORTED_MODELS.items():
        info.append({
            "model_path": model_path,
            "display_name": config.name,
            "params": config.generation_params
        })
    return info
