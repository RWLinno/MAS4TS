#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
配置文件验证工具
用于验证OnCallAgent配置文件的有效性
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import jsonschema
from jsonschema import ValidationError


CONFIG_SCHEMA = {
    "type": "object",
    "required": ["api_keys", "directories", "models", "logging"],
    "properties": {
        "api_keys": {
            "type": "object",
            "properties": {
                "openai": {"type": "string"},
                "azure_openai": {
                    "type": "object",
                    "properties": {
                        "api_key": {"type": "string"},
                        "api_base": {"type": "string"},
                        "api_version": {"type": "string"}
                    },
                    "required": ["api_key", "api_base"]
                },
                "anthropic": {"type": "string"}
            }
        },
        "directories": {
            "type": "object",
            "properties": {
                "data_dir": {"type": "string"},
                "cache_dir": {"type": "string"},
                "logs_dir": {"type": "string"}
            },
            "required": ["data_dir", "cache_dir", "logs_dir"]
        },
        "models": {
            "type": "object",
            "properties": {
                "default_llm": {"type": "string"},
                "embedding_model": {"type": "string"},
                "fallback_llm": {"type": "string"},
                "local_models": {
                    "type": "object",
                    "properties": {
                        "enabled": {"type": "boolean"},
                        "path": {"type": "string"},
                        "model_name": {"type": "string"}
                    },
                    "required": ["enabled"]
                }
            },
            "required": ["default_llm", "embedding_model"]
        },
        "rag": {
            "type": "object",
            "properties": {
                "retrieval": {
                    "type": "object",
                    "properties": {
                        "top_k": {"type": "integer", "minimum": 1},
                        "similarity_threshold": {"type": "number", "minimum": 0, "maximum": 1},
                        "chunk_size": {"type": "integer", "minimum": 100},
                        "chunk_overlap": {"type": "integer", "minimum": 0}
                    }
                },
                "knowledge_base": {
                    "type": "object",
                    "properties": {
                        "default_index": {"type": "string"},
                        "vector_store": {"type": "string"},
                        "update_frequency": {"type": "string"}
                    }
                }
            }
        },
        "privacy": {
            "type": "object",
            "properties": {
                "mask_emails": {"type": "boolean"},
                "mask_phone_numbers": {"type": "boolean"},
                "mask_ips": {"type": "boolean"},
                "mask_urls": {"type": "boolean"},
                "mask_tokens": {"type": "boolean"},
                "mask_credit_cards": {"type": "boolean"},
                "allowed_domains": {"type": "array", "items": {"type": "string"}},
                "hash_pii": {"type": "boolean"}
            }
        },
        "logging": {
            "type": "object",
            "properties": {
                "level": {"type": "string", "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]},
                "format": {"type": "string"},
                "to_file": {"type": "boolean"},
                "max_file_size_mb": {"type": "integer", "minimum": 1},
                "backup_count": {"type": "integer", "minimum": 0}
            },
            "required": ["level"]
        },
        "api_service": {
            "type": "object",
            "properties": {
                "host": {"type": "string"},
                "port": {"type": "integer", "minimum": 1, "maximum": 65535},
                "workers": {"type": "integer", "minimum": 1},
                "cors": {
                    "type": "object",
                    "properties": {
                        "allow_origins": {"type": "array", "items": {"type": "string"}},
                        "allow_methods": {"type": "array", "items": {"type": "string"}},
                        "allow_headers": {"type": "array", "items": {"type": "string"}}
                    }
                },
                "rate_limit": {
                    "type": "object",
                    "properties": {
                        "enabled": {"type": "boolean"},
                        "rate": {"type": "integer", "minimum": 1},
                        "per_seconds": {"type": "integer", "minimum": 1}
                    }
                }
            }
        },
        "web_app": {
            "type": "object",
            "properties": {
                "port": {"type": "integer", "minimum": 1, "maximum": 65535},
                "theme": {
                    "type": "object",
                    "properties": {
                        "primary_color": {"type": "string"},
                        "background_color": {"type": "string"},
                        "text_color": {"type": "string"}
                    }
                },
                "page_title": {"type": "string"},
                "favicon": {"type": "string"}
            }
        },
        "caching": {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean"},
                "ttl_seconds": {"type": "integer", "minimum": 1},
                "max_size_mb": {"type": "integer", "minimum": 1},
                "redis": {
                    "type": "object",
                    "properties": {
                        "enabled": {"type": "boolean"},
                        "host": {"type": "string"},
                        "port": {"type": "integer", "minimum": 1, "maximum": 65535},
                        "password": {"type": "string"},
                        "db": {"type": "integer", "minimum": 0}
                    }
                }
            }
        },
        "advanced": {
            "type": "object",
            "properties": {
                "max_tokens_per_request": {"type": "integer", "minimum": 1},
                "temperature": {"type": "number", "minimum": 0, "maximum": 2},
                "request_timeout": {"type": "integer", "minimum": 1},
                "max_retries": {"type": "integer", "minimum": 0},
                "retry_delay": {"type": "integer", "minimum": 0},
                "session_expiry_hours": {"type": "integer", "minimum": 1},
                "system_prompts": {
                    "type": "object",
                    "additionalProperties": {"type": "string"}
                }
            }
        }
    }
}


def find_config_file() -> Optional[Path]:
    """
    查找配置文件路径
    
    Returns:
        Optional[Path]: 配置文件路径，如果未找到则返回None
    """
    # 优先检查工作目录
    config_path = Path("config.json")
    if config_path.exists():
        return config_path
    
    # 检查项目根目录
    project_root = Path(__file__).parent.parent.parent.parent
    config_path = project_root / "config.json"
    if config_path.exists():
        return config_path
    
    # 检查环境变量中指定的路径
    if "ONCALL_CONFIG_PATH" in os.environ:
        config_path = Path(os.environ["ONCALL_CONFIG_PATH"])
        if config_path.exists():
            return config_path
    
    return None


def load_config() -> Dict[str, Any]:
    """
    加载并解析配置文件
    
    Returns:
        Dict[str, Any]: 配置信息字典
    
    Raises:
        FileNotFoundError: 如果找不到配置文件
        json.JSONDecodeError: 如果配置文件不是有效的JSON
    """
    config_path = find_config_file()
    if not config_path:
        raise FileNotFoundError(
            "找不到配置文件。请确保config.json位于当前目录、项目根目录，"
            "或通过环境变量ONCALL_CONFIG_PATH指定。"
        )
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    return config


def validate_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    验证配置文件是否符合预定义模式
    
    Args:
        config: 配置字典
    
    Returns:
        Tuple[bool, List[str]]: (是否有效, 错误消息列表)
    """
    errors = []
    is_valid = True
    
    try:
        jsonschema.validate(config, CONFIG_SCHEMA)
    except ValidationError as e:
        is_valid = False
        errors.append(f"配置验证错误: {e.message} 位于 {'/'.join(str(p) for p in e.path)}")
    
    # 额外的自定义验证规则
    if is_valid:
        # 检查目录是否存在
        for dir_key, dir_path in config.get("directories", {}).items():
            if not os.path.exists(dir_path) and not dir_path.startswith("$"):
                errors.append(f"目录 '{dir_path}' ({dir_key}) 不存在")
        
        # 检查本地模型配置
        local_models = config.get("models", {}).get("local_models", {})
        if local_models.get("enabled", False):
            model_path = local_models.get("path")
            if not model_path:
                errors.append("启用本地模型但未指定模型路径")
            elif not os.path.exists(model_path) and not model_path.startswith("$"):
                errors.append(f"本地模型路径 '{model_path}' 不存在")
            
            if not local_models.get("model_name"):
                errors.append("启用本地模型但未指定模型名称")
    
    return len(errors) == 0, errors


def suggest_fixes(errors: List[str]) -> List[str]:
    """
    为配置错误提供修复建议
    
    Args:
        errors: 错误消息列表
    
    Returns:
        List[str]: 修复建议列表
    """
    suggestions = []
    
    for error in errors:
        if "目录" in error and "不存在" in error:
            dir_name = error.split("'")[1]
            suggestions.append(f"  - 创建缺失的目录: mkdir -p {dir_name}")
        
        elif "本地模型路径" in error and "不存在" in error:
            path = error.split("'")[1]
            suggestions.append(f"  - 创建模型目录: mkdir -p {path}")
            suggestions.append(f"  - 或禁用本地模型: 将models.local_models.enabled设置为false")
        
        elif "必需属性" in error:
            missing_prop = error.split("'")[1]
            suggestions.append(f"  - 添加缺失的属性: {missing_prop}")
        
        elif "不是 'string' 类型" in error:
            field = error.split("/")[-1]
            suggestions.append(f"  - 确保 {field} 是字符串类型")
    
    if not suggestions:
        suggestions.append("  - 参考文档中的配置示例: docs/configuration.md")
    
    return suggestions


def main():
    """主入口函数"""
    print("OnCallAgent 配置验证工具")
    print("------------------------")
    
    try:
        config = load_config()
        print(f"已找到配置文件，正在验证...")
        
        is_valid, errors = validate_config(config)
        
        if is_valid:
            print("✓ 配置文件有效！所有检查均已通过。")
            return 0
        else:
            print("\n❌ 配置文件验证失败！")
            print("\n发现以下问题:")
            for i, error in enumerate(errors, 1):
                print(f"  {i}. {error}")
            
            print("\n建议修复:")
            for suggestion in suggest_fixes(errors):
                print(suggestion)
            
            return 1
    
    except FileNotFoundError as e:
        print(f"\n❌ 错误: {e}")
        print("\n建议:")
        print("  - 从示例创建配置文件: cp config.example.json config.json")
        print("  - 然后编辑配置文件添加必要的设置")
        return 1
    
    except json.JSONDecodeError as e:
        print(f"\n❌ 配置文件不是有效的JSON格式: {e}")
        print("\n建议:")
        print("  - 检查JSON语法错误，如缺失逗号或引号")
        print("  - 使用JSON验证工具检查语法")
        return 1
    
    except Exception as e:
        print(f"\n❌ 发生意外错误: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 