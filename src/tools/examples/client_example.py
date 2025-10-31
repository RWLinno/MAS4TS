# OnCallAgent/src/oncall_agent/mcp/examples/client_example.py

import asyncio
import json
import logging
from oncall_agent.mcp.server import MCPClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp_client_example")

async def main():
    # 连接到MCP服务器
    client = MCPClient(server_url="ws://localhost:8000/mcp")
    await client.connect()
    
    try:
        # 获取文件列表
        logger.info("获取当前目录文件列表...")
        files = await client.call_tool("file", action="list", path=".")
        logger.info(f"文件列表: {json.dumps(files, indent=2)}")
        
        # 搜索知识库
        logger.info("搜索知识库...")
        search_results = await client.call_tool("knowledge", action="search", query="Kubernetes troubleshooting", limit=3)
        logger.info(f"搜索结果: {json.dumps(search_results, indent=2)}")
        
        # 使用代理处理查询
        logger.info("使用代理处理查询...")
        agent_result = await client.call_tool("agent", action="process", query="如何解决Kubernetes中的Pod无法启动问题?")
        logger.info(f"代理回答: {agent_result.get('response')}")
        
        # 使用GitHub工具
        logger.info("搜索GitHub仓库...")
        github_results = await client.call_tool("github", action="search_repositories", query="kubernetes", limit=3)
        logger.info(f"GitHub搜索结果: {json.dumps(github_results, indent=2)}")
        
    finally:
        # 关闭连接
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())