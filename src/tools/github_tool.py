# OnCallAgent/src/oncall_agent/mcp/tools/github_tool.py

import os
import json
import aiohttp
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel

from oncall_agent.mcp.server import MCPTool

class GitHubTool(MCPTool):
    """GitHub MCP工具，用于与GitHub API交互"""
    
    name = "github"
    description = "与GitHub API交互，查询仓库、问题和PR信息"
    
    def __init__(self, token: Optional[str] = None):
        self.token = token or os.environ.get("GITHUB_TOKEN")
        self.api_url = "https://api.github.com"
    
    async def execute(self, params: Dict[str, Any]) -> Any:
        """执行GitHub工具"""
        action = params.get("action")
        
        if not action:
            raise ValueError("Action parameter is required")
        
        if action == "search_repositories":
            return await self._search_repositories(
                query=params.get("query"),
                limit=params.get("limit", 5)
            )
        elif action == "get_repository":
            return await self._get_repository(
                owner=params.get("owner"),
                repo=params.get("repo")
            )
        elif action == "list_issues":
            return await self._list_issues(
                owner=params.get("owner"),
                repo=params.get("repo"),
                state=params.get("state", "open"),
                limit=params.get("limit", 10)
            )
        elif action == "get_issue":
            return await self._get_issue(
                owner=params.get("owner"),
                repo=params.get("repo"),
                issue_number=params.get("issue_number")
            )
        else:
            raise ValueError(f"Unknown action: {action}")
    
    async def _github_request(self, method: str, endpoint: str, params: Dict[str, Any] = None) -> Any:
        """发送GitHub API请求"""
        headers = {
            "Accept": "application/vnd.github.v3+json"
        }
        
        if self.token:
            headers["Authorization"] = f"token {self.token}"
        
        url = f"{self.api_url}{endpoint}"
        
        async with aiohttp.ClientSession() as session:
            async with session.request(method, url, headers=headers, params=params) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ValueError(f"GitHub API error ({response.status}): {error_text}")
                
                return await response.json()
    
    async def _search_repositories(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """搜索GitHub仓库"""
        if not query:
            raise ValueError("Query parameter is required")
        
        params = {
            "q": query,
            "per_page": limit
        }
        
        response = await self._github_request("GET", "/search/repositories", params)
        
        repositories = []
        for repo in response.get("items", []):
            repositories.append({
                "id": repo.get("id"),
                "name": repo.get("name"),
                "full_name": repo.get("full_name"),
                "description": repo.get("description"),
                "html_url": repo.get("html_url"),
                "stars": repo.get("stargazers_count"),
                "forks": repo.get("forks_count"),
                "language": repo.get("language"),
                "owner": {
                    "login": repo.get("owner", {}).get("login"),
                    "avatar_url": repo.get("owner", {}).get("avatar_url")
                }
            })
        
        return repositories
    
    async def _get_repository(self, owner: str, repo: str) -> Dict[str, Any]:
        """获取GitHub仓库信息"""
        if not owner or not repo:
            raise ValueError("Owner and repo parameters are required")
        
        response = await self._github_request("GET", f"/repos/{owner}/{repo}")
        
        return {
            "id": response.get("id"),
            "name": response.get("name"),
            "full_name": response.get("full_name"),
            "description": response.get("description"),
            "html_url": response.get("html_url"),
            "stars": response.get("stargazers_count"),
            "forks": response.get("forks_count"),
            "language": response.get("language"),
            "topics": response.get("topics", []),
            "default_branch": response.get("default_branch"),
            "created_at": response.get("created_at"),
            "updated_at": response.get("updated_at"),
            "owner": {
                "login": response.get("owner", {}).get("login"),
                "avatar_url": response.get("owner", {}).get("avatar_url")
            }
        }
    
    async def _list_issues(self, owner: str, repo: str, state: str = "open", limit: int = 10) -> List[Dict[str, Any]]:
        """列出GitHub仓库的问题"""
        if not owner or not repo:
            raise ValueError("Owner and repo parameters are required")
        
        params = {
            "state": state,
            "per_page": limit
        }
        
        response = await self._github_request("GET", f"/repos/{owner}/{repo}/issues", params)
        
        issues = []
        for issue in response:
            # 过滤掉PR（GitHub API将PR也作为issue返回）
            if "pull_request" not in issue:
                issues.append({
                    "number": issue.get("number"),
                    "title": issue.get("title"),
                    "state": issue.get("state"),
                    "html_url": issue.get("html_url"),
                    "created_at": issue.get("created_at"),
                    "updated_at": issue.get("updated_at"),
                    "user": {
                        "login": issue.get("user", {}).get("login"),
                        "avatar_url": issue.get("user", {}).get("avatar_url")
                    },
                    "labels": [label.get("name") for label in issue.get("labels", [])]
                })
        
        return issues
    
    async def _get_issue(self, owner: str, repo: str, issue_number: int) -> Dict[str, Any]:
        """获取GitHub问题详情"""
        if not owner or not repo or not issue_number:
            raise ValueError("Owner, repo and issue_number parameters are required")
        
        response = await self._github_request("GET", f"/repos/{owner}/{repo}/issues/{issue_number}")
        
        return {
            "number": response.get("number"),
            "title": response.get("title"),
            "body": response.get("body"),
            "state": response.get("state"),
            "html_url": response.get("html_url"),
            "created_at": response.get("created_at"),
            "updated_at": response.get("updated_at"),
            "user": {
                "login": response.get("user", {}).get("login"),
                "avatar_url": response.get("user", {}).get("avatar_url")
            },
            "labels": [label.get("name") for label in response.get("labels", [])],
            "comments": response.get("comments")
        }
    
    def get_schema(self) -> Dict[str, Any]:
        """返回工具的JSON Schema描述"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "required": ["action"],
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["search_repositories", "get_repository", "list_issues", "get_issue"],
                        "description": "要执行的GitHub操作"
                    },
                    "query": {
                        "type": "string",
                        "description": "搜索查询（仅用于搜索操作）"
                    },
                    "owner": {
                        "type": "string",
                        "description": "仓库所有者名称"
                    },
                    "repo": {
                        "type": "string",
                        "description": "仓库名称"
                    },
                    "state": {
                        "type": "string",
                        "enum": ["open", "closed", "all"],
                        "default": "open",
                        "description": "问题状态（仅用于列出问题操作）"
                    },
                    "issue_number": {
                        "type": "integer",
                        "description": "问题编号（仅用于获取问题操作）"
                    },
                    "limit": {
                        "type": "integer",
                        "default": 10,
                        "description": "返回结果的最大数量"
                    }
                }
            }
        }