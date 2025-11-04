"""
Training script for MAS4TS
支持多任务训练和评估
"""

import torch
import argparse
import json
import sys
from pathlib import Path
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.model import MAS4TS, DEFAULT_CONFIG
from src.utils.logger import setup_logger
from data_provider.data_factory import data_provider
import logging

logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Train MAS4TS for time series tasks')
    
    # 基本配置
    parser.add_argument('--config', type=str, default='src/config.json',
                       help='Path to config file')
    parser.add_argument('--task_name', type=str, required=True,
                       choices=['forecasting', 'classification', 'imputation', 'anomaly_detection'],
                       help='Task name')
    parser.add_argument('--data', type=str, required=True,
                       help='Dataset name (ETTh1, ETTm1, Weather, etc.)')
    
    # 模型配置
    parser.add_argument('--model', type=str, default='DLinear',
                       help='Model name')
    parser.add_argument('--seq_len', type=int, default=96,
                       help='Input sequence length')
    parser.add_argument('--pred_len', type=int, default=96,
                       help='Prediction length')
    parser.add_argument('--label_len', type=int, default=48,
                       help='Label length')
    
    # 训练配置
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--train_epochs', type=int, default=10,
                       help='Training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                       help='Learning rate')
    parser.add_argument('--patience', type=int, default=3,
                       help='Early stopping patience')
    
    # 设备配置
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU id')
    
    # 其他
    parser.add_argument('--seed', type=int, default=2021,
                       help='Random seed')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/',
                       help='Checkpoint directory')
    parser.add_argument('--test_only', action='store_true',
                       help='Test only mode')
    
    return parser.parse_args()


def set_seed(seed):
    """设置随机种子"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path):
    """加载配置文件"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded config from {config_path}")
        return config
    except:
        logger.warning(f"Failed to load config from {config_path}, using default")
        return DEFAULT_CONFIG


def prepare_data(args, config):
    """准备数据加载器"""
    # 从配置文件获取数据集配置
    dataset_config = config.get('datasets', {}).get(args.data, {})
    
    # 构建数据加载器参数
    data_dict = {
        'data': args.data,
        'root_path': dataset_config.get('root_path', f'./dataset/{args.data}/'),
        'data_path': dataset_config.get('data_path', f'{args.data}.csv'),
        'features': dataset_config.get('features', 'M'),
        'target': dataset_config.get('target', 'OT'),
        'freq': dataset_config.get('freq', 'h'),
        'seq_len': args.seq_len,
        'label_len': args.label_len,
        'pred_len': args.pred_len,
        'batch_size': args.batch_size,
        'num_workers': config.get('system', {}).get('num_workers', 10),
        'timeenc': 0
    }
    
    # 创建数据加载器
    train_loader = data_provider(data_dict, 'train')
    val_loader = data_provider(data_dict, 'val')
    test_loader = data_provider(data_dict, 'test')
    
    logger.info(f"Data loaders created for {args.data}")
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader


def train_epoch(model, train_loader, criterion, device):
    """训练一个epoch"""
    total_loss = 0
    num_batches = 0
    
    for batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
        batch_x = batch_x.to(device).float()
        batch_y = batch_y.to(device).float()
        
        # 通过MAS4TS处理
        result = model.process(batch_x, 'forecasting', {'pred_len': batch_y.size(1)})
        
        if result['success']:
            predictions = result['predictions']
            
            # 计算损失
            loss = criterion(predictions, batch_y)
            total_loss += loss.item()
            num_batches += 1
            
            if (batch_idx + 1) % 50 == 0:
                logger.info(f"Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        else:
            logger.warning(f"Batch {batch_idx} processing failed")
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return avg_loss


def validate(model, val_loader, criterion, device):
    """验证"""
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in val_loader:
            batch_x = batch_x.to(device).float()
            batch_y = batch_y.to(device).float()
            
            # 通过MAS4TS处理
            result = model.process(batch_x, 'forecasting', {'pred_len': batch_y.size(1)})
            
            if result['success']:
                predictions = result['predictions']
                
                # 计算损失
                loss = criterion(predictions, batch_y)
                total_loss += loss.item()
                num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    return avg_loss


def test(model, test_loader, device):
    """测试"""
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
            batch_x = batch_x.to(device).float()
            batch_y = batch_y.to(device).float()
            
            # 通过MAS4TS处理
            result = model.process(batch_x, 'forecasting', {'pred_len': batch_y.size(1)})
            
            if result['success']:
                predictions = result['predictions']
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())
    
    # 计算指标
    if all_predictions:
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        mse = np.mean((predictions - targets) ** 2)
        mae = np.mean(np.abs(predictions - targets))
        rmse = np.sqrt(mse)
        
        logger.info("=" * 50)
        logger.info("Test Results:")
        logger.info(f"MSE: {mse:.4f}")
        logger.info(f"MAE: {mae:.4f}")
        logger.info(f"RMSE: {rmse:.4f}")
        logger.info("=" * 50)
        
        return {'mse': mse, 'mae': mae, 'rmse': rmse}
    else:
        return {}


def main():
    """主函数"""
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置日志
    setup_logger('INFO', config.get('logging', {}))
    
    logger.info("=" * 50)
    logger.info("MAS4TS Training Script")
    logger.info("=" * 50)
    logger.info(f"Task: {args.task_name}")
    logger.info(f"Dataset: {args.data}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Device: {device}")
    logger.info("=" * 50)
    
    # 更新配置
    config.update({
        'device': str(device),
        'seq_len': args.seq_len,
        'pred_len': args.pred_len,
        'label_len': args.label_len,
        'default_model': args.model,
        'task_name': args.task_name
    })
    
    # 创建模型
    logger.info("Creating MAS4TS model...")
    model = MAS4TS(config)
    
    # 准备数据
    logger.info("Preparing data loaders...")
    train_loader, val_loader, test_loader = prepare_data(args, config)
    
    if args.test_only:
        # 仅测试模式
        logger.info("Running in test-only mode")
        test(model, test_loader, device)
    else:
        # 训练模式
        criterion = torch.nn.MSELoss()
        best_val_loss = float('inf')
        patience_counter = 0
        
        logger.info("Starting training...")
        for epoch in range(args.train_epochs):
            logger.info(f"\nEpoch [{epoch + 1}/{args.train_epochs}]")
            
            # 训练
            train_loss = train_epoch(model, train_loader, criterion, device)
            logger.info(f"Train Loss: {train_loss:.4f}")
            
            # 验证
            val_loss = validate(model, val_loader, criterion, device)
            logger.info(f"Val Loss: {val_loss:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # 保存最佳模型
                checkpoint_dir = Path(args.checkpoints)
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_path = checkpoint_dir / f"mas4ts_{args.data}_{args.task_name}_best.pt"
                model.save_checkpoint(str(checkpoint_path))
                logger.info(f"Best model saved to {checkpoint_path}")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
        
        # 测试
        logger.info("\nRunning final test...")
        test(model, test_loader, device)
    
    logger.info("Training completed!")


if __name__ == '__main__':
    main()

