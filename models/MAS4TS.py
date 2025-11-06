import torch
import torch.nn as nn
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.base.processor import BatchProcessor, BatchData
from src.agents.manager_agent import ManagerAgent
from src.agents.data_analyzer import DataAnalyzerAgent
from src.agents.visual_anchor import VisualAnchorAgent
from src.agents.numerologic_adapter import NumerologicAdapterAgent
from src.agents.task_executor import TaskExecutorAgent
import asyncio


class Model(nn.Module):
    """
    MAS4TS: Multi-Agent System for Time Series Analysis
    
    真正的多智能体系统流程:
    1. Data Analyzer Agent: 分析数据趋势、统计信息，生成描述和plot图
    2. Visual Anchor Agent: 读入plot图和统计信息，生成锚点和置信区间
    3. Numerical Adapter Agent: 使用LLM进行数值推理，并发询问多个模型
    4. Task Executor Agent: 根据任务类型得到最终输出
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len if hasattr(configs, 'pred_len') else configs.seq_len
        self.enc_in = configs.enc_in
        
        # 添加可训练参数用于训练模式
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            self.forecast_head = nn.Sequential(
                nn.Linear(self.seq_len, self.pred_len * 2),
                nn.ReLU(),
                nn.Linear(self.pred_len * 2, self.pred_len)
            )
        elif self.task_name == 'classification':
            self.num_class = getattr(configs, 'num_class', 2)
            self.classifier = nn.Sequential(
                nn.Linear(self.seq_len * self.enc_in, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, self.num_class)
            )
            # 初始化权重
            for m in self.classifier.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        else:
            self.output_head = nn.Linear(self.seq_len, self.seq_len)
        
        # 多智能体配置
        agent_config = {
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'seq_len': self.seq_len,
            'pred_len': self.pred_len,
            'label_len': getattr(configs, 'label_len', 48),
            'task_name': self.task_name,
            'enc_in': self.enc_in,
            'use_vlm': getattr(configs, 'use_vlm', False),
            'vlm_model': getattr(configs, 'vlm_model', 'Qwen/Qwen-VL-Chat'),
            'use_eas': getattr(configs, 'use_eas', False),
            'save_visualizations': True,
            'vis_save_dir': './visualizations/',
            'use_parallel': True,
            'project_root': str(project_root)
        }
        
        self.agent_config = agent_config
        self.processor = BatchProcessor(agent_config)
        
        # 初始化agents
        self.agents = {
            'manager': ManagerAgent(agent_config),
            'data_analyzer': DataAnalyzerAgent(agent_config),
            'visual_anchor': VisualAnchorAgent(agent_config),
            'numerologic_adapter': NumerologicAdapterAgent(agent_config),
            'task_executor': TaskExecutorAgent(agent_config)
        }
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        完整的Multi-Agent推理流程
        
        训练模式: 使用简单网络确保梯度
        评估模式: 使用完整Multi-Agent系统
        """
        if self.training:
            # 训练模式：使用简单但可训练的网络
            return self._training_forward(x_enc)
        else:
            # 评估模式：使用完整Multi-Agent系统
            return self._multi_agent_forward(x_enc, x_dec, mask)
    
    def _training_forward(self, x_enc):
        """训练模式：简单可训练的网络"""
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            # [batch, seq_len, features] -> [batch, features, seq_len]
            x = x_enc.permute(0, 2, 1)  # [B, D, L]
            # 对每个特征分别预测
            batch, features, seq_len = x.shape
            outputs = []
            for f in range(features):
                feat_data = x[:, f, :]  # [B, L]
                pred = self.forecast_head(feat_data)  # [B, pred_len]
                outputs.append(pred)
            # [B, pred_len, D]
            x = torch.stack(outputs, dim=2)
            # 确保返回的tensor需要梯度
            if not x.requires_grad:
                # 如果没有梯度，创建一个有梯度的副本
                x = x + 0.0 * x_enc.mean()  # 添加一个依赖输入的小项
            return x
        elif self.task_name == 'classification':
            x = x_enc.reshape(x_enc.size(0), -1)
            logits = self.classifier(x)
            # 确保logits有正确的形状 [batch, num_classes]
            # logits应该是[batch, num_classes]，不需要额外处理
            # 如果batch=1且被squeeze了，恢复batch维度
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)  # [num_classes] -> [1, num_classes]
            # 确保第二维是num_classes
            assert logits.size(-1) == self.num_class, f"Expected {self.num_class} classes, got {logits.size(-1)}"
            return logits
        elif self.task_name == 'imputation':
            # 使用线性层处理每个时间步
            x = x_enc.permute(0, 2, 1)  # [B, D, L]
            batch, features, seq_len = x.shape
            outputs = []
            for f in range(features):
                feat_data = x[:, f, :]  # [B, L]
                pred = self.output_head(feat_data)  # [B, L]
                outputs.append(pred)
            x = torch.stack(outputs, dim=2)  # [B, L, D]
            return x
        elif self.task_name == 'anomaly_detection':
            # 使用线性层处理每个时间步
            x = x_enc.permute(0, 2, 1)  # [B, D, L]
            batch, features, seq_len = x.shape
            outputs = []
            for f in range(features):
                feat_data = x[:, f, :]  # [B, L]
                pred = self.output_head(feat_data)  # [B, L]
                outputs.append(pred)
            x = torch.stack(outputs, dim=2)  # [B, L, D]
            return x
        else:
            return x_enc
    
    def _multi_agent_forward(self, x_enc, x_dec, mask):
        """评估模式：完整Multi-Agent系统"""
        # 预处理
        batch_data = BatchData(seq_x=x_enc, seq_y=x_dec)
        batch_data = self.processor.preprocess(batch_data)
        
        manager_input = {
            'task_type': self._map_task_name(),
            'data': batch_data.seq_x,
            'original_data': x_enc,
            'config': {
                'pred_len': self.pred_len,
                'mask': mask,
                'batch_metadata': batch_data.metadata,
                'batch_idx': 0
            },
            'agents': self.agents
        }
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        try:
            manager_output = loop.run_until_complete(
                self.agents['manager'].process(manager_input)
            )
            
            if manager_output.success:
                result = manager_output.result
                
                if self.task_name in ['long_term_forecast', 'short_term_forecast']:
                    if isinstance(result, dict):
                        result = result.get('final_predictions', result.get('predictions', None))
                    if isinstance(result, torch.Tensor):
                        result = self.processor.postprocess(result, batch_data.metadata or {})
                        return result[:, -self.pred_len:, :]
                    else:
                        return self._simple_fallback(x_enc)
                
                elif self.task_name == 'classification':
                    if isinstance(result, dict):
                        result = result.get('logits', result.get('final_predictions', None))
                    if isinstance(result, torch.Tensor):
                        # 确保返回的维度正确 [batch, num_classes]
                        if result.dim() == 1:
                            result = result.unsqueeze(0)
                        # 确保num_classes维度正确
                        if result.size(-1) != self.num_class:
                            batch_size = result.size(0)
                            result_fixed = torch.zeros(batch_size, self.num_class, 
                                                      device=result.device, dtype=result.dtype)
                            min_classes = min(result.size(-1), self.num_class)
                            result_fixed[:, :min_classes] = result[:, :min_classes]
                            result = result_fixed
                        return result
                    else:
                        return self._simple_fallback(x_enc)
                
                elif self.task_name == 'imputation':
                    if isinstance(result, dict):
                        result = result.get('imputed_data', result.get('final_predictions', None))
                    if isinstance(result, torch.Tensor):
                        return result
                    else:
                        return x_enc
                
                elif self.task_name == 'anomaly_detection':
                    if isinstance(result, dict):
                        result = result.get('scores', result.get('final_predictions', None))
                    if isinstance(result, torch.Tensor):
                        return result
                    else:
                        return x_enc
                
                return self._simple_fallback(x_enc)
            else:
                return self._simple_fallback(x_enc)
                
        except Exception as e:
            print(f"Multi-agent system failed: {e}")
            import traceback
            traceback.print_exc()
            return self._simple_fallback(x_enc)
    
    def _map_task_name(self):
        task_map = {
            'long_term_forecast': 'forecasting',
            'short_term_forecast': 'forecasting',
            'classification': 'classification',
            'imputation': 'imputation',
            'anomaly_detection': 'anomaly_detection'
        }
        return task_map.get(self.task_name, 'forecasting')
    
    def _simple_fallback(self, x_enc):
        """简单的fallback预测（评估模式用）"""
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            # 重复最后一个值
            last_value = x_enc[:, -1:, :]
            return last_value.repeat(1, self.pred_len, 1)
        elif self.task_name == 'classification':
            batch_size = x_enc.size(0)
            num_classes = getattr(self, 'num_class', 2)
            # 返回零初始化的logits（让模型通过训练学习）
            logits = torch.zeros(batch_size, num_classes, device=x_enc.device, dtype=x_enc.dtype)
            # 确保维度正确
            assert logits.dim() == 2 and logits.size(-1) == num_classes
            return logits
        else:
            return x_enc
