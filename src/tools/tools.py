"""
函数工具
提供系统各种功能函数
"""

from typing import Dict, List, Optional, Any, Union
import json
import logging
import datetime
import random
from .protocol import FunctionRegistry

logger = logging.getLogger(__name__)

@FunctionRegistry.register
def search_logs(
    query: str,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    service: Optional[str] = None,
    limit: int = 10
) -> Dict[str, Any]:
    """
    搜索系统日志，查找指定时间段和关键词的日志条目
    
    Args:
        query: 搜索关键词
        start_time: 开始时间，ISO格式
        end_time: 结束时间，ISO格式
        service: 服务名称
        limit: 返回结果数量上限
    
    Returns:
        包含日志条目的字典
    """
    # 这里应当调用实际的日志搜索服务
    # 现在返回模拟数据
    
    now = datetime.datetime.now()
    
    # 随机生成一些日志条目
    log_entries = []
    services = ["api-gateway", "user-service", "payment-service", "order-service"]
    log_levels = ["INFO", "WARN", "ERROR", "DEBUG"]
    
    for i in range(min(limit, 20)):
        timestamp = (now - datetime.timedelta(minutes=random.randint(0, 60))).isoformat()
        
        log_service = service or random.choice(services)
        log_level = random.choice(log_levels)
        
        # 确保至少有一个错误日志
        if i == 0 and "error" in query.lower():
            log_level = "ERROR"
            log_message = f"Error processing request: {query}"
        else:
            log_message = f"Process completed with status {random.randint(200, 500)}"
            
            # 有一定概率包含查询关键词
            if random.random() > 0.5:
                log_message = f"{log_message} for {query}"
        
        log_entries.append({
            "timestamp": timestamp,
            "service": log_service,
            "level": log_level,
            "message": log_message,
            "trace_id": f"trace-{random.randint(1000, 9999)}"
        })
    
    return {
        "query": query,
        "start_time": start_time,
        "end_time": end_time,
        "service": service,
        "count": len(log_entries),
        "logs": log_entries
    }

@FunctionRegistry.register
def get_system_metrics(
    metrics: List[str],
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    interval: str = "1m",
    service: Optional[str] = None
) -> Dict[str, Any]:
    """
    获取系统各项指标数据，如CPU、内存、磁盘使用率等
    
    Args:
        metrics: 要获取的指标列表，如['cpu', 'memory', 'disk']
        start_time: 开始时间，ISO格式
        end_time: 结束时间，ISO格式
        interval: 时间间隔，如'1m', '5m', '1h'
        service: 服务名称
    
    Returns:
        包含指标数据的字典
    """
    # 生成模拟数据
    now = datetime.datetime.now()
    
    # 解析时间间隔
    interval_seconds = 60  # 默认1分钟
    if interval.endswith('s'):
        interval_seconds = int(interval[:-1])
    elif interval.endswith('m'):
        interval_seconds = int(interval[:-1]) * 60
    elif interval.endswith('h'):
        interval_seconds = int(interval[:-1]) * 3600
    
    # 解析时间范围
    if end_time:
        end = datetime.datetime.fromisoformat(end_time)
    else:
        end = now
    
    if start_time:
        start = datetime.datetime.fromisoformat(start_time)
    else:
        start = end - datetime.timedelta(hours=1)
    
    # 生成时间点
    timestamps = []
    current = start
    while current <= end:
        timestamps.append(current.isoformat())
        current += datetime.timedelta(seconds=interval_seconds)
    
    # 生成指标数据
    results = {}
    for metric in metrics:
        values = []
        
        if metric == 'cpu':
            # CPU使用率，60-90%
            values = [random.uniform(60, 90) for _ in timestamps]
        elif metric == 'memory':
            # 内存使用率，50-80%
            values = [random.uniform(50, 80) for _ in timestamps]
        elif metric == 'disk':
            # 磁盘使用率，40-70%
            values = [random.uniform(40, 70) for _ in timestamps]
        elif metric == 'network':
            # 网络带宽使用，1-100Mbps
            values = [random.uniform(1, 100) for _ in timestamps]
        elif metric == 'latency':
            # 延迟，10-200ms
            values = [random.uniform(10, 200) for _ in timestamps]
        elif metric == 'error_rate':
            # 错误率，0-5%
            values = [random.uniform(0, 5) for _ in timestamps]
        else:
            # 其他指标，随机生成
            values = [random.uniform(0, 100) for _ in timestamps]
        
        results[metric] = values
    
    return {
        "metrics": metrics,
        "start_time": start.isoformat(),
        "end_time": end.isoformat(),
        "interval": interval,
        "service": service,
        "timestamps": timestamps,
        "values": results
    }

@FunctionRegistry.register
def generate_metric_chart(
    metric: str,
    chart_type: str = "line",
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    service: Optional[str] = None
) -> Dict[str, Any]:
    """
    生成系统指标的图表，如趋势图、柱状图等
    
    Args:
        metric: 要绘制的指标，如'cpu', 'memory'
        chart_type: 图表类型，如'line', 'bar', 'pie'
        start_time: 开始时间，ISO格式
        end_time: 结束时间，ISO格式
        service: 服务名称
    
    Returns:
        包含图表URL或数据的字典
    """
    # 实际应用中，这里会调用图表生成服务
    # 现在返回模拟数据
    
    chart_url = f"https://example.com/charts/{metric}_{chart_type}_{service or 'all'}.png"
    
    return {
        "metric": metric,
        "chart_type": chart_type,
        "start_time": start_time,
        "end_time": end_time,
        "service": service,
        "chart_url": chart_url,
        "description": f"这是{service or '所有服务'}的{metric}指标{chart_type}图表，显示了从{start_time or '一小时前'}到{end_time or '现在'}的数据。"
    }

@FunctionRegistry.register
def search_knowledge_base(
    query: str,
    category: Optional[str] = None,
    limit: int = 5
) -> Dict[str, Any]:
    """
    搜索知识库，查找与查询相关的文档或解决方案
    
    Args:
        query: 搜索关键词
        category: 知识类别，如'troubleshooting', 'guide', 'api'
        limit: 返回结果数量上限
    
    Returns:
        包含知识库条目的字典
    """
    # 生成一些模拟的知识库条目
    categories = category.split(',') if category else ["troubleshooting", "guide", "api", "faq"]
    if not category:
        category = random.choice(categories)
    
    entries = []
    
    for i in range(min(limit, 10)):
        # 生成条目标题，有一定概率包含查询关键词
        if random.random() > 0.3:
            title = f"关于{query}的{category}文档"
        else:
            topics = ["系统配置", "性能优化", "错误处理", "API使用", "最佳实践"]
            title = f"{random.choice(topics)}指南"
        
        # 生成条目摘要
        summary = f"这份文档提供了关于{title}的详细信息，包括常见问题、解决方案和示例代码。"
        
        entries.append({
            "id": f"kb-{random.randint(1000, 9999)}",
            "title": title,
            "category": category,
            "summary": summary,
            "url": f"https://example.com/kb/{title.replace(' ', '-').lower()}",
            "relevance": random.uniform(0.5, 1.0)
        })
    
    # 按相关性排序
    entries.sort(key=lambda x: x["relevance"], reverse=True)
    
    return {
        "query": query,
        "category": category,
        "count": len(entries),
        "entries": entries
    }

@FunctionRegistry.register
def get_recent_incidents(
    severity: Optional[str] = None,
    limit: int = 5,
    service: Optional[str] = None
) -> Dict[str, Any]:
    """
    获取系统最近的事件或警告
    
    Args:
        severity: 严重程度，如'critical', 'warning', 'info'
        limit: 返回结果数量上限
        service: 服务名称
    
    Returns:
        包含事件信息的字典
    """
    now = datetime.datetime.now()
    
    # 生成模拟数据
    severities = ["critical", "warning", "info"]
    if severity:
        severities = [s for s in severities if s == severity]
    
    services = ["api-gateway", "user-service", "payment-service", "order-service"]
    if service:
        services = [s for s in services if s == service]
    
    incidents = []
    
    for i in range(min(limit, 10)):
        # 随机生成一个过去的时间点
        incident_time = now - datetime.timedelta(minutes=random.randint(10, 1440))
        
        # 随机选择服务和严重程度
        incident_service = random.choice(services)
        incident_severity = random.choice(severities)
        
        # 生成事件描述
        if incident_severity == "critical":
            description = f"{incident_service}服务出现严重故障，导致部分功能不可用"
        elif incident_severity == "warning":
            description = f"{incident_service}服务性能下降，响应时间增加"
        else:
            description = f"{incident_service}服务配置更新，可能影响部分功能"
        
        # 生成事件状态
        statuses = ["active", "resolved", "investigating"]
        weights = [0.2, 0.5, 0.3]  # 更多已解决的事件
        status = random.choices(statuses, weights=weights)[0]
        
        incidents.append({
            "id": f"inc-{random.randint(1000, 9999)}",
            "timestamp": incident_time.isoformat(),
            "service": incident_service,
            "severity": incident_severity,
            "description": description,
            "status": status,
            "duration": random.randint(5, 120) if status == "resolved" else None  # 分钟
        })
    
    # 按时间排序，最近的在前
    incidents.sort(key=lambda x: x["timestamp"], reverse=True)
    
    return {
        "severity": severity,
        "service": service,
        "count": len(incidents),
        "incidents": incidents
    }

@FunctionRegistry.register
def check_service_health(
    service: Optional[str] = None,
    check_dependencies: bool = False
) -> Dict[str, Any]:
    """
    检查服务健康状态
    
    Args:
        service: 服务名称，None表示检查所有服务
        check_dependencies: 是否检查依赖服务
    
    Returns:
        包含服务健康状态的字典
    """
    # 生成模拟数据
    services = ["api-gateway", "user-service", "payment-service", "order-service", "notification-service"]
    
    if service:
        services = [s for s in services if s == service]
    
    # 健康状态
    statuses = ["healthy", "degraded", "unhealthy"]
    weights = [0.8, 0.15, 0.05]  # 大多数服务是健康的
    
    results = {}
    
    for srv in services:
        # 随机选择状态
        status = random.choices(statuses, weights=weights)[0]
        
        # 随机生成一些指标
        metrics = {
            "uptime": f"{random.randint(1, 30)}d {random.randint(0, 23)}h {random.randint(0, 59)}m",
            "response_time": f"{random.randint(10, 500)}ms",
            "error_rate": f"{random.uniform(0, 5):.2f}%",
            "cpu_usage": f"{random.uniform(10, 90):.1f}%",
            "memory_usage": f"{random.uniform(20, 80):.1f}%"
        }
        
        # 如果检查依赖
        dependencies = {}
        if check_dependencies:
            # 随机选择1-3个依赖
            num_deps = random.randint(1, 3)
            potential_deps = [d for d in services if d != srv]
            
            if potential_deps:
                deps = random.sample(potential_deps, min(num_deps, len(potential_deps)))
                
                for dep in deps:
                    dep_status = random.choices(statuses, weights=weights)[0]
                    dependencies[dep] = {
                        "status": dep_status,
                        "response_time": f"{random.randint(10, 500)}ms"
                    }
        
        results[srv] = {
            "status": status,
            "last_checked": datetime.datetime.now().isoformat(),
            "metrics": metrics,
            "dependencies": dependencies
        }
        
        # 如果有依赖且状态不健康，则主服务可能也不健康
        if dependencies and any(d["status"] != "healthy" for d in dependencies.values()):
            if random.random() < 0.7:  # 70%概率受依赖影响
                results[srv]["status"] = "degraded" if random.random() < 0.7 else "unhealthy"
    
    return {
        "checked_at": datetime.datetime.now().isoformat(),
        "service": service,
        "check_dependencies": check_dependencies,
        "results": results
    } 