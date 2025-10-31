import time
import functools
from typing import Any, Callable, Dict, Tuple, Optional, TypeVar

T = TypeVar('T')

class TTLCache:
    """具有过期时间的缓存实现"""
    
    def __init__(self):
        self.cache: Dict[str, Tuple[Any, float]] = {}
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值，如果过期则返回None"""
        if key not in self.cache:
            return None
        
        value, expiry = self.cache[key]
        if expiry < time.time():
            # 缓存已过期
            del self.cache[key]
            return None
        
        return value
    
    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        """设置缓存值及其过期时间"""
        expiry = time.time() + ttl_seconds
        self.cache[key] = (value, expiry)
    
    def clear(self) -> None:
        """清空缓存"""
        self.cache.clear()
    
    def remove_expired(self) -> int:
        """移除所有过期项目并返回移除数量"""
        now = time.time()
        expired_keys = [k for k, (_, exp) in self.cache.items() if exp < now]
        
        for key in expired_keys:
            del self.cache[key]
        
        return len(expired_keys)

# 创建一个全局缓存实例
_global_cache = TTLCache()

def cache_with_ttl(ttl_seconds: int = 3600):
    """缓存装饰器，缓存函数调用结果并设置过期时间
    
    :param ttl_seconds: 缓存过期时间（秒），默认1小时
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # 创建缓存键
            cache_key = f"{func.__module__}.{func.__name__}:{hash(str(args))}-{hash(str(kwargs))}"
            
            # 尝试从缓存获取结果
            cached_result = _global_cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # 调用原函数
            result = func(*args, **kwargs)
            
            # 缓存结果
            _global_cache.set(cache_key, result, ttl_seconds)
            
            return result
        return wrapper
    return decorator

def clear_cache() -> None:
    """清空全局缓存"""
    _global_cache.clear()

def clean_expired_cache() -> int:
    """清除过期缓存条目并返回清除数量"""
    return _global_cache.remove_expired() 