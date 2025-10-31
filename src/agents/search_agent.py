#!/usr/bin/env python3
"""
Search Agent
Real-time web search and content extraction for OnCall scenarios
Inspired by Eigent's search capabilities
"""

import asyncio
import logging
import aiohttp
import json
import re
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse
from bs4 import BeautifulSoup

from .base import BaseAgent, AgentInput, AgentOutput, AgentConfig, AgentRegistry


from ..utils.simple_model_manager import ModelManager

logger = logging.getLogger(__name__)

@AgentRegistry.register()
class SearchAgent(BaseAgent):
    """
    Web search agent for real-time information retrieval
    Searches technical documentation, GitHub issues, and community resources
    """
    
    def __init__(self, config: AgentConfig, global_config: Dict[str, Any]):
        super().__init__(config, global_config)
        
        # Search configuration
        self.max_results = config.specialized_config.get("max_results", 10)
        self.max_content_length = config.specialized_config.get("max_content_length", 5000)
        self.trusted_domains = config.specialized_config.get("trusted_domains", [
            "stackoverflow.com", "github.com", "kubernetes.io", "redis.io",
            "kafka.apache.org", "mysql.com", "prometheus.io", "grafana.com"
        ])
        
        # Initialize model manager for content summarization
        self.model_manager = ModelManager(global_config)
    
    async def _search_duckduckgo(self, query: str) -> List[Dict[str, Any]]:
        """Search using DuckDuckGo API (free, no API key required)"""
        try:
            params = {
                'q': query,
                'format': 'json',
                'no_html': '1',
                'skip_disambig': '1'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://api.duckduckgo.com/",
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        results = self._parse_duckduckgo_results(data)
                        logger.info(f"DuckDuckGo search returned {len(results)} results")
                        return results
                    else:
                        logger.warning(f"DuckDuckGo API returned status {response.status}")
                        return []
        
        except Exception as e:
            logger.warning(f"DuckDuckGo search failed: {e}")
            return []
    
    def _parse_duckduckgo_results(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse DuckDuckGo API response"""
        results = []
        
        # Parse instant answer
        if data.get('Abstract'):
            results.append({
                'title': data.get('Heading', 'Instant Answer'),
                'url': data.get('AbstractURL', ''),
                'content': data.get('Abstract', ''),
                'source': 'duckduckgo_instant',
                'relevance_score': 0.9
            })
        
        # Parse related topics
        for topic in data.get('RelatedTopics', [])[:self.max_results]:
            if isinstance(topic, dict) and topic.get('FirstURL'):
                results.append({
                    'title': topic.get('Text', '').split(' - ')[0],
                    'url': topic.get('FirstURL', ''),
                    'content': topic.get('Text', ''),
                    'source': 'duckduckgo_related',
                    'relevance_score': 0.7
                })
        
        return results
    
    async def _search_github_issues(self, query: str) -> List[Dict[str, Any]]:
        """Search GitHub issues for technical problems"""
        try:
            # Optimize query for GitHub search
            github_query = f"{query} is:issue state:closed"
            
            async with aiohttp.ClientSession() as session:
                headers = {
                    'Accept': 'application/vnd.github.v3+json',
                    'User-Agent': 'OnCallAgent/1.0'
                }
                
                params = {
                    'q': github_query,
                    'sort': 'relevance',
                    'per_page': 5
                }
                
                async with session.get(
                    "https://api.github.com/search/issues",
                    headers=headers,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_github_results(data)
        
        except Exception as e:
            logger.warning(f"GitHub search failed: {e}")
            return []
    
    def _parse_github_results(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse GitHub API response"""
        results = []
        
        for item in data.get('items', []):
            results.append({
                'title': item.get('title', ''),
                'url': item.get('html_url', ''),
                'content': (item.get('body', '') or '')[:500],
                'source': 'github_issues',
                'repository': item.get('repository_url', '').split('/')[-1],
                'state': item.get('state', ''),
                'labels': [label['name'] for label in item.get('labels', [])],
                'relevance_score': 0.8
            })
        
        return results
    
    async def _extract_webpage_content(self, url: str) -> Optional[str]:
        """Extract content from webpage"""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (compatible; OnCallAgent/1.0)'
                }
                
                async with session.get(
                    url,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=15)
                ) as response:
                    if response.status != 200:
                        return None
                    
                    content = await response.text()
                    soup = BeautifulSoup(content, 'html.parser')
                    
                    # Remove unwanted elements
                    for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                        element.decompose()
                    
                    # Extract main content
                    content_selectors = [
                        'main', 'article', '.content', '.main-content',
                        '.documentation', '.docs', '.markdown-body'
                    ]
                    
                    for selector in content_selectors:
                        element = soup.select_one(selector)
                        if element:
                            text = element.get_text(strip=True)
                            return self._clean_extracted_text(text)
                    
                    # Fallback to body content
                    body = soup.find('body')
                    if body:
                        text = body.get_text(strip=True)
                        return self._clean_extracted_text(text)
        
        except Exception as e:
            logger.warning(f"Content extraction failed for {url}: {e}")
            return None
    
    def _clean_extracted_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Truncate if too long
        if len(text) > self.max_content_length:
            text = text[:self.max_content_length] + "..."
        
        return text.strip()
    
    def _prioritize_results(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Prioritize search results based on OnCall relevance"""
        def calculate_priority_score(result: Dict[str, Any]) -> float:
            score = result.get('relevance_score', 0.5)
            
            # Boost trusted domains
            url = result.get('url', '')
            domain = urlparse(url).netloc
            if any(trusted in domain for trusted in self.trusted_domains):
                score += 0.3
            
            # Boost closed GitHub issues (proven solutions)
            if result.get('source') == 'github_issues' and result.get('state') == 'closed':
                score += 0.2
            
            # Boost official documentation
            if any(keyword in url.lower() for keyword in ['docs', 'documentation', 'manual']):
                score += 0.15
            
            return min(score, 1.0)  # Cap at 1.0
        
        # Calculate priority scores and sort
        scored_results = [(result, calculate_priority_score(result)) for result in results]
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        return [result for result, score in scored_results]
    
    async def _summarize_search_results(self, results: List[Dict[str, Any]], query: str) -> str:
        """Summarize search results using LLM"""
        if not results:
            return "No relevant search results found."
        
        try:
            # Prepare content for summarization
            content_pieces = []
            for i, result in enumerate(results[:5], 1):
                title = result.get('title', 'Unknown')
                url = result.get('url', '')
                content = result.get('content', '')
                
                content_pieces.append(f"{i}. {title}\nURL: {url}\nContent: {content}")
            
            combined_content = "\n\n".join(content_pieces)
            
            # Generate summary using model manager
            prompt = f"""Based on the following search results, provide a concise and actionable answer for the OnCall engineer's question.

Question: {query}

Search Results:
{combined_content}

Please provide:
1. Direct answer to the question
2. Key actionable steps (if applicable)
3. Relevant links for further reading
4. Any urgent considerations for OnCall scenarios

Keep the response focused and practical for immediate use."""
            
            response = await self.model_manager.generate_response(
                prompt=prompt,
                max_tokens=1024,
                temperature=0.3
            )
            
            return response
        
        except Exception as e:
            logger.error(f"Search result summarization failed: {e}")
            # Fallback to basic formatting
            return self._format_basic_summary(results)
    
    def _format_basic_summary(self, results: List[Dict[str, Any]]) -> str:
        """Basic formatting for search results"""
        if not results:
            return "No search results available."
        
        summary = "🔍 **Search Results:**\n\n"
        
        for i, result in enumerate(results[:5], 1):
            title = result.get('title', 'Unknown')
            url = result.get('url', '')
            content = result.get('content', '')[:200]
            
            summary += f"{i}. **{title}**\n"
            if url:
                summary += f"   🔗 {url}\n"
            if content:
                summary += f"   📝 {content}...\n"
            summary += "\n"
        
        return summary
    
    async def _process_query(self, input_data: AgentInput) -> AgentOutput:
        """Process search query and return summarized results"""
        query = input_data.query
        context = input_data.context
        
        try:
            logger.info(f"🔍 Starting web search for: {query}")
            
            # Optimize query for technical search
            optimized_query = self._optimize_search_query(query)
            
            # Perform searches
            search_tasks = [
                self._search_duckduckgo(optimized_query),
                self._search_github_issues(optimized_query)
            ]
            
            search_results_lists = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            # Combine results
            all_results = []
            for results in search_results_lists:
                if isinstance(results, list):
                    all_results.extend(results)
                elif isinstance(results, Exception):
                    logger.warning(f"Search task failed: {results}")
            
            if not all_results:
                return AgentOutput(
                    response="No search results found. Please try different keywords or contact technical support.",
                    confidence=0.3,
                    context={"search_query": optimized_query, "results_count": 0}
                )
            
            # Prioritize and limit results
            prioritized_results = self._prioritize_results(all_results, query)[:self.max_results]
            
            # Summarize results
            summary = await self._summarize_search_results(prioritized_results, query)
            
            return AgentOutput(
                response=summary,
                confidence=0.8,
                context={
                    "search_query": optimized_query,
                    "results_count": len(prioritized_results),
                    "top_sources": [r.get('url', '') for r in prioritized_results[:3]],
                    "domains_searched": list(set(urlparse(r.get('url', '')).netloc for r in prioritized_results))
                }
            )
        
        except Exception as e:
            logger.error(f"Search agent execution failed: {e}")
            return AgentOutput(
                response=f"Search failed: {str(e)}. Please try again or use alternative information sources.",
                confidence=0.0,
                context={"error": str(e)}
            )
    
    def _optimize_search_query(self, query: str) -> str:
        """Optimize query for technical search"""
        # Add technical context
        optimized = query
        
        # Add common technical suffixes
        if any(keyword in query.lower() for keyword in ['error', 'problem', 'issue']):
            optimized += " solution troubleshooting"
        elif any(keyword in query.lower() for keyword in ['how to', 'tutorial']):
            optimized += " guide documentation"
        elif any(keyword in query.lower() for keyword in ['best practice', 'optimization']):
            optimized += " best practices"
        
        return optimized