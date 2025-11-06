"""
EAS (Elastic Algorithm Service) Client
支持调用阿里云EAS上部署的VLM和LLM服务
"""

import requests
import json
import base64
from typing import Dict, Any, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class EASClient:
    """阿里云EAS服务客户端"""
    
    def __init__(self, endpoint: str, token: str):
        self.endpoint = endpoint
        self.token = token
        self.headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
    
    async def call_vlm(self, image_path: str, prompt: str, timeout: int = 30) -> Dict[str, Any]:
        """
        调用VLM服务
        
        Args:
            image_path: 图片路径
            prompt: 文本prompt
            timeout: 超时时间
            
        Returns:
            VLM响应
        """
        try:
            # 读取并编码图片
            with open(image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            
            payload = {
                'image': image_data,
                'prompt': prompt,
                'max_tokens': 512,
                'temperature': 0.7
            }
            
            response = requests.post(
                f"{self.endpoint}/vlm",
                headers=self.headers,
                json=payload,
                timeout=timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info("VLM call successful")
                return result
            else:
                logger.error(f"VLM call failed: {response.status_code} - {response.text}")
                return {'error': response.text}
                
        except Exception as e:
            logger.error(f"Error calling VLM: {e}")
            return {'error': str(e)}
    
    async def call_llm(self, prompt: str, model_name: str = 'qwen-7b', 
                       max_tokens: int = 512, temperature: float = 0.7, 
                       timeout: int = 30) -> Dict[str, Any]:
        """
        调用LLM服务
        
        Args:
            prompt: 文本prompt
            model_name: 模型名称
            max_tokens: 最大生成token数
            temperature: 温度参数
            timeout: 超时时间
            
        Returns:
            LLM响应
        """
        try:
            payload = {
                'prompt': prompt,
                'model': model_name,
                'max_tokens': max_tokens,
                'temperature': temperature
            }
            
            response = requests.post(
                f"{self.endpoint}/llm",
                headers=self.headers,
                json=payload,
                timeout=timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"LLM ({model_name}) call successful")
                return result
            else:
                logger.error(f"LLM call failed: {response.status_code} - {response.text}")
                return {'error': response.text}
                
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            return {'error': str(e)}

