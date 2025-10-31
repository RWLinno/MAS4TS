#!/usr/bin/env python3
"""
MCP (Model Context Protocol) Agents for OnCallAgent
Provides MCP integration for log analysis and numerical computation tasks
"""

import asyncio
import os
import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

try:
    from mcp_use import MCPAgent, MCPClient
    from langchain_openai import ChatOpenAI
    from dotenv import load_dotenv
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    logging.warning("MCP dependencies not available. Install with: pip install mcp-use langchain-openai python-dotenv")

from .base import BaseAgent

class MCPLogAnalysisAgent(BaseAgent):
    """MCP-powered agent for log analysis and operational insights"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = "MCP Log Analysis Agent"
        self.description = "Analyzes logs using MCP tools for operational insights"
        
        if not MCP_AVAILABLE:
            raise ImportError("MCP dependencies not available")
        
        load_dotenv()
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable required")
        
        self.mcp_client = None
        self.mcp_agent = None
        self._initialize_mcp()
    
    def _initialize_mcp(self):
        """Initialize MCP client and agent"""
        try:
            # Define MCP server configuration for log analysis tools
            server_config = {
                "mcpServers": {
                    "filesystem": {
                        "command": "npx",
                        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
                    },
                    "everything": {
                        "command": "npx",
                        "args": ["-y", "@modelcontextprotocol/server-everything"],
                    }
                }
            }
            
            # Create MCP client
            self.mcp_client = MCPClient(config=server_config)
            
            # Create LLM
            llm = ChatOpenAI(
                model="gpt-4o-mini",  # Use cost-effective model for log analysis
                api_key=self.openai_api_key,
                temperature=0.1
            )
            
            # Create MCP agent
            self.mcp_agent = MCPAgent(llm=llm, client=self.mcp_client)
            
            logging.info("MCP Log Analysis Agent initialized successfully")
            
        except Exception as e:
            logging.error(f"Failed to initialize MCP components: {e}")
            raise
    
    async def analyze_logs(self, log_path: str, analysis_type: str = "error_summary") -> Dict[str, Any]:
        """
        Analyze logs using MCP tools
        
        Args:
            log_path: Path to log file
            analysis_type: Type of analysis ('error_summary', 'performance_metrics', 'security_audit')
        """
        if not self.mcp_agent:
            raise RuntimeError("MCP agent not initialized")
        
        try:
            if analysis_type == "error_summary":
                prompt = f"""
                Please analyze the log file at {log_path} and provide:
                1. Count of different error types
                2. Most frequent error patterns
                3. Timeline of critical errors
                4. Suggested remediation steps
                
                Use the filesystem tools to read and process the log file.
                """
            
            elif analysis_type == "performance_metrics":
                prompt = f"""
                Analyze the log file at {log_path} for performance metrics:
                1. Response time statistics (min, max, avg, p95, p99)
                2. Throughput analysis
                3. Resource utilization patterns
                4. Performance bottlenecks identification
                
                Extract numerical data and provide statistical summary.
                """
            
            elif analysis_type == "security_audit":
                prompt = f"""
                Perform security analysis on log file at {log_path}:
                1. Failed authentication attempts
                2. Suspicious IP patterns
                3. Potential security threats
                4. Access pattern anomalies
                
                Provide security recommendations based on findings.
                """
            
            else:
                prompt = f"Analyze log file at {log_path} and provide general insights."
            
            # Execute analysis using MCP agent
            result = await self.mcp_agent.run(prompt)
            
            return {
                "success": True,
                "analysis_type": analysis_type,
                "log_path": log_path,
                "result": result,
                "timestamp": self._get_timestamp()
            }
            
        except Exception as e:
            logging.error(f"Log analysis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "analysis_type": analysis_type,
                "log_path": log_path
            }
    
    async def calculate_metrics(self, data_source: str, metrics_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate operational metrics using MCP tools
        
        Args:
            data_source: Path to data file or database query
            metrics_config: Configuration for metrics calculation
        """
        try:
            prompt = f"""
            Calculate operational metrics from data source: {data_source}
            
            Metrics configuration: {json.dumps(metrics_config, indent=2)}
            
            Please:
            1. Read and parse the data source
            2. Calculate the requested metrics
            3. Provide statistical analysis
            4. Generate alerts if thresholds are exceeded
            5. Create visualization recommendations
            
            Use appropriate tools to process the data and perform calculations.
            """
            
            result = await self.mcp_agent.run(prompt)
            
            return {
                "success": True,
                "data_source": data_source,
                "metrics": result,
                "config": metrics_config,
                "timestamp": self._get_timestamp()
            }
            
        except Exception as e:
            logging.error(f"Metrics calculation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "data_source": data_source
            }
    
    async def troubleshoot_issue(self, issue_description: str, context_files: List[str] = None) -> Dict[str, Any]:
        """
        Troubleshoot operational issues using MCP tools
        
        Args:
            issue_description: Description of the issue
            context_files: List of relevant files (logs, configs, etc.)
        """
        try:
            context_info = ""
            if context_files:
                context_info = f"\nContext files to analyze: {', '.join(context_files)}"
            
            prompt = f"""
            Troubleshoot the following operational issue:
            
            Issue: {issue_description}
            {context_info}
            
            Please:
            1. Analyze the provided context files
            2. Identify potential root causes
            3. Provide step-by-step troubleshooting guide
            4. Suggest preventive measures
            5. Create monitoring recommendations
            
            Use filesystem and analysis tools to investigate the issue thoroughly.
            """
            
            result = await self.mcp_agent.run(prompt)
            
            return {
                "success": True,
                "issue": issue_description,
                "troubleshooting_result": result,
                "context_files": context_files or [],
                "timestamp": self._get_timestamp()
            }
            
        except Exception as e:
            logging.error(f"Troubleshooting failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "issue": issue_description
            }
    
    async def cleanup(self):
        """Clean up MCP resources"""
        if self.mcp_client:
            try:
                await self.mcp_client.close_all_sessions()
                logging.info("MCP client sessions closed")
            except Exception as e:
                logging.error(f"Error closing MCP sessions: {e}")


class MCPMetricsAgent(BaseAgent):
    """MCP-powered agent specialized for metrics calculation and monitoring"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = "MCP Metrics Agent"
        self.description = "Calculates and monitors operational metrics using MCP tools"
        
        if not MCP_AVAILABLE:
            raise ImportError("MCP dependencies not available")
        
        load_dotenv()
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable required")
        
        self.mcp_client = None
        self.mcp_agent = None
        self._initialize_mcp()
    
    def _initialize_mcp(self):
        """Initialize MCP client for metrics calculation"""
        try:
            # Configure MCP servers for metrics and calculations
            server_config = {
                "mcpServers": {
                    "everything": {
                        "command": "npx",
                        "args": ["-y", "@modelcontextprotocol/server-everything"],
                    },
                    "filesystem": {
                        "command": "npx",
                        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
                    }
                }
            }
            
            self.mcp_client = MCPClient(config=server_config)
            
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                api_key=self.openai_api_key,
                temperature=0
            )
            
            self.mcp_agent = MCPAgent(llm=llm, client=self.mcp_client)
            logging.info("MCP Metrics Agent initialized successfully")
            
        except Exception as e:
            logging.error(f"Failed to initialize MCP Metrics Agent: {e}")
            raise
    
    async def calculate_sla_metrics(self, data_path: str, sla_config: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate SLA metrics from operational data"""
        try:
            prompt = f"""
            Calculate SLA metrics from data file: {data_path}
            
            SLA Configuration:
            {json.dumps(sla_config, indent=2)}
            
            Please calculate:
            1. Uptime percentage
            2. Response time percentiles (P50, P95, P99)
            3. Error rate
            4. Availability metrics
            5. SLA compliance status
            
            Provide detailed numerical analysis and recommendations.
            """
            
            result = await self.mcp_agent.run(prompt)
            
            return {
                "success": True,
                "metrics_type": "sla",
                "data_path": data_path,
                "sla_config": sla_config,
                "results": result,
                "timestamp": self._get_timestamp()
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def analyze_capacity_planning(self, usage_data_path: str, growth_config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze capacity planning metrics"""
        try:
            prompt = f"""
            Perform capacity planning analysis on usage data: {usage_data_path}
            
            Growth Configuration:
            {json.dumps(growth_config, indent=2)}
            
            Please analyze:
            1. Current resource utilization trends
            2. Growth rate calculations
            3. Capacity forecasting (3, 6, 12 months)
            4. Resource scaling recommendations
            5. Cost optimization opportunities
            
            Use mathematical calculations and trend analysis.
            """
            
            result = await self.mcp_agent.run(prompt)
            
            return {
                "success": True,
                "analysis_type": "capacity_planning",
                "data_path": usage_data_path,
                "growth_config": growth_config,
                "results": result,
                "timestamp": self._get_timestamp()
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def cleanup(self):
        """Clean up MCP resources"""
        if self.mcp_client:
            try:
                await self.mcp_client.close_all_sessions()
            except Exception as e:
                logging.error(f"Error closing MCP sessions: {e}")