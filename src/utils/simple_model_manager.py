#!/usr/bin/env python3
"""
Simple Model Manager
Simplified interface compatible with oca831 version
"""

import logging
import torch
from typing import Dict, Any, Optional, List, Union
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Centralized model management for OnCallAgent
    Handles loading, caching, and inference for multiple model types
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models_cache: Dict[str, Any] = {}
        self.device = self._determine_device()
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        logger.info(f"✓ Model manager initialized with device: {self.device}")
    
    def _determine_device(self) -> str:
        """Determine optimal device for model inference"""
        device_config = self.config.get("models", {}).get("device", "auto")
        
        if device_config == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"  # Apple Silicon
            else:
                return "cpu"
        
        return device_config
    
    async def load_model(self, model_name: str, model_type: str = "text") -> Optional[Any]:
        """
        Load and cache model
        
        Args:
            model_name: HuggingFace model identifier
            model_type: Type of model (text, vision, embedding)
            
        Returns:
            Loaded model pipeline or None if failed
        """
        cache_key = f"{model_name}:{model_type}"
        
        if cache_key in self.models_cache:
            logger.debug(f"Using cached model: {model_name}")
            return self.models_cache[cache_key]
        
        try:
            logger.info(f"Loading model: {model_name} (type: {model_type})")
            
            # Load model in thread pool to avoid blocking
            model = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._load_model_sync,
                model_name,
                model_type
            )
            
            if model:
                self.models_cache[cache_key] = model
                logger.info(f"✓ Model {model_name} loaded successfully")
            
            return model
        
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return None
    
    def _load_model_sync(self, model_name: str, model_type: str) -> Optional[Any]:
        """Synchronous model loading (runs in thread pool)"""
        try:
            if model_type == "text":
                return pipeline(
                    "text-generation",
                    model=model_name,
                    tokenizer=model_name,
                    device_map="auto" if self.device == "cuda" else None,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    trust_remote_code=True
                )
            
            elif model_type == "vision":
                return pipeline(
                    "image-to-text",
                    model=model_name,
                    device_map="auto" if self.device == "cuda" else None,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    trust_remote_code=True
                )
            
            elif model_type == "embedding":
                from sentence_transformers import SentenceTransformer
                return SentenceTransformer(model_name, device=self.device)
            
            else:
                logger.warning(f"Unknown model type: {model_type}")
                return None
        
        except Exception as e:
            logger.error(f"Synchronous model loading failed: {e}")
            return None
    
    async def generate_response(self, prompt: str, model_name: Optional[str] = None,
                              max_tokens: int = 512, temperature: float = 0.7,
                              **kwargs) -> str:
        """
        Generate text response using specified or default model
        
        Args:
            prompt: Input prompt
            model_name: Specific model to use (optional)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
        """
        model_name = model_name or self.config.get("models", {}).get("default_model", "Qwen/Qwen2.5-7B-Instruct")
        
        # Check if we should use mock mode to avoid downloading models
        offline_mode = self.config.get("offline_mode", False)
        if offline_mode:
            logger.info("Using mock mode for text generation")
            return self._generate_mock_response(prompt)
        
        try:
            # Load model
            model = await self.load_model(model_name, "text")
            if not model:
                logger.warning("Model loading failed, falling back to mock mode")
                return self._generate_mock_response(prompt)
            
            # Generate response in thread pool
            response = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._generate_sync,
                model,
                prompt,
                max_tokens,
                temperature,
                kwargs
            )
            
            return response
        
        except Exception as e:
            logger.error(f"Text generation failed: {e}, falling back to mock mode")
            return self._generate_mock_response(prompt)
    
    def _generate_sync(self, model: Any, prompt: str, max_tokens: int, 
                      temperature: float, kwargs: Dict[str, Any]) -> str:
        """Synchronous text generation (runs in thread pool)"""
        try:
            generation_params = {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "do_sample": temperature > 0,
                "pad_token_id": model.tokenizer.eos_token_id,
                **kwargs
            }
            
            outputs = model(prompt, **generation_params)
            
            # Extract generated text
            if isinstance(outputs, list) and len(outputs) > 0:
                generated_text = outputs[0].get('generated_text', '')
                
                # Remove the input prompt from output
                if generated_text.startswith(prompt):
                    generated_text = generated_text[len(prompt):].strip()
                
                return generated_text
            
            return "No response generated"
        
        except Exception as e:
            logger.error(f"Synchronous generation failed: {e}")
            return f"Generation error: {str(e)}"
    
    async def analyze_image(self, image_path: str, prompt: str = "Describe this image",
                          model_name: Optional[str] = None) -> str:
        """
        Analyze image using vision-language model
        
        Args:
            image_path: Path to image file
            prompt: Analysis prompt
            model_name: Specific VLM to use
            
        Returns:
            Image analysis result
        """
        model_name = model_name or "Qwen/Qwen2.5-VL-7B-Instruct"
        
        try:
            # Load vision model
            model = await self.load_model(model_name, "vision")
            if not model:
                raise ValueError(f"Failed to load vision model: {model_name}")
            
            # Analyze image in thread pool
            result = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._analyze_image_sync,
                model,
                image_path,
                prompt
            )
            
            return result
        
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            return f"Image analysis failed: {str(e)}"
    
    def _analyze_image_sync(self, model: Any, image_path: str, prompt: str) -> str:
        """Synchronous image analysis (runs in thread pool)"""
        try:
            # Note: Actual implementation depends on the specific VLM pipeline
            # This is a placeholder for the image analysis logic
            result = model(image_path, prompt=prompt)
            
            if isinstance(result, list) and len(result) > 0:
                return result[0].get('generated_text', 'No analysis generated')
            
            return "Image analysis completed"
        
        except Exception as e:
            logger.error(f"Synchronous image analysis failed: {e}")
            return f"Analysis error: {str(e)}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        info = {
            "device": self.device,
            "loaded_models": list(self.models_cache.keys()),
            "cache_size": len(self.models_cache),
            "gpu_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
        
        if torch.cuda.is_available():
            try:
                info["gpu_memory"] = {
                    "total": torch.cuda.get_device_properties(0).total_memory / 1024**3,
                    "allocated": torch.cuda.memory_allocated(0) / 1024**3,
                    "cached": torch.cuda.memory_reserved(0) / 1024**3
                }
            except Exception as e:
                logger.warning(f"Failed to get GPU memory info: {e}")
        
        return info
    
    def clear_cache(self, model_name: Optional[str] = None):
        """Clear model cache"""
        if model_name:
            # Clear specific model
            keys_to_remove = [key for key in self.models_cache.keys() if model_name in key]
            for key in keys_to_remove:
                del self.models_cache[key]
            logger.info(f"Cleared cache for model: {model_name}")
        else:
            # Clear all models
            self.models_cache.clear()
            logger.info("Cleared all model cache")
        
        # Force garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _generate_mock_response(self, prompt: str) -> str:
        """Generate a mock response when models are not available"""
        if "redis" in prompt.lower() and "timeout" in prompt.lower():
            return """Based on the search results, here are the key steps to troubleshoot Redis connection timeouts:

## Immediate Actions:
1. **Check Redis server status**: Verify if Redis is running and accessible
2. **Review connection pool settings**: Ensure proper timeout and pool size configuration
3. **Monitor network connectivity**: Check for network issues between client and Redis server

## Common Solutions:
1. **Increase timeout values** in your Redis client configuration
2. **Optimize connection pooling** to handle high concurrency
3. **Check Redis server performance** and memory usage
4. **Review firewall and network settings**

## Key Configuration Parameters:
- `socket_timeout`: Set to 5-10 seconds
- `socket_connect_timeout`: Set to 5 seconds
- `connection_pool_max_connections`: Adjust based on load

## Monitoring:
- Monitor Redis slow log for performance issues
- Check client-side connection metrics
- Review Redis server logs for errors

For detailed troubleshooting steps, consult the Redis documentation and monitoring tools."""
        
        # Generic technical response
        return """Based on the search results, here are the recommended troubleshooting steps:

1. **Identify the Problem**: Review error messages and logs to understand the specific issue
2. **Check System Status**: Verify that all relevant services and components are running properly
3. **Review Configuration**: Ensure all configuration parameters are set correctly
4. **Monitor Performance**: Check system metrics and performance indicators
5. **Apply Solutions**: Implement fixes based on the identified root cause

For more detailed information, please refer to the official documentation or contact technical support."""
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
