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
from src.agents.numeric_reasoner import NumericReasonerAgent
from src.agents.task_executor import TaskExecutorAgent
import asyncio


class Model(nn.Module):
    """
    MAS4TS: Multi-Agent System for Time Series Analysis 
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len if hasattr(configs, 'pred_len') else configs.seq_len
        self.enc_in = configs.enc_in
        self.num_class = getattr(configs, 'num_class', 2)
        
        agent_config = {
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'seq_len': self.seq_len,
            'pred_len': self.pred_len,
            'label_len': getattr(configs, 'label_len', 48),
            'task_name': self.task_name,
            'enc_in': self.enc_in,
            'num_class': self.num_class,
            'use_vlm': getattr(configs, 'use_vlm', False),
            'vlm_model': getattr(configs, 'vlm_model', 'Qwen/Qwen-VL-Chat'),
            'use_eas': getattr(configs, 'use_eas', False),
            'save_visualizations': True,
            'save_anchors': True,  # 默认开启anchor本地存储
            'vis_save_dir': './visualizations/',
            'use_parallel': True,
            'project_root': str(project_root)
        }

        self.processor = BatchProcessor(agent_config)
        self.manager = ManagerAgent(agent_config)
        self.agents = {
            'manager': self.manager,
            'data_analyzer': DataAnalyzerAgent(agent_config),
            'visual_anchor': VisualAnchorAgent(agent_config),
            'numerologic_adapter': NumericReasonerAgent(agent_config), 
            'task_executor': TaskExecutorAgent(agent_config)
        }
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        batch_data = BatchData(seq_x=x_enc, seq_y=x_dec)
        batch_data = self.processor.preprocess(batch_data)
        
        manager_input = {
            'task_type': self._map_task_name(),
            'data': batch_data.seq_x,
            'original_data': x_enc,
            'config': {
                'pred_len': self.pred_len,
                'num_classes': self.num_class,
                'mask': mask,
                'batch_metadata': batch_data.metadata,
                'batch_idx': getattr(self, '_batch_counter', 0)
            },
            'agents': self.agents
        }
        
        self._batch_counter = getattr(self, '_batch_counter', 0) + 1
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        try:
            manager_output = loop.run_until_complete(
                self.manager.process(manager_input)
            )
            
            if not manager_output.success:
                print(f"Manager执行失败: {manager_output.metadata.get('error', 'Unknown')}")
                return self._simple_fallback(x_enc)
            
            result = manager_output.result
            return self.agents['task_executor'].process_output(
                result, batch_data, x_enc, self.task_name
            )
                
        except Exception as e:
            print(f"Multi-agent系统异常: {e}")
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
        print(f"WARNING: 正在使用fallback方法进行{self.task_name}任务")
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            last_value = x_enc[:, -1:, :]
            return last_value.repeat(1, self.pred_len, 1)
        elif self.task_name == 'classification':
            batch_size = x_enc.size(0)
            return torch.zeros(batch_size, self.num_class, device=x_enc.device, dtype=x_enc.dtype)
        else:
            return x_enc
