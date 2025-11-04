"""
Evaluation script for MAS4TS
支持多任务评估和可视化
"""

import torch
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

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
    parser = argparse.ArgumentParser(description='Evaluate MAS4TS')
    
    parser.add_argument('--config', type=str, default='src/config.json')
    parser.add_argument('--task_name', type=str, required=True)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--model', type=str, default='DLinear')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint to load')
    
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--batch_size', type=int, default=32)
    
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--gpu', type=int, default=0)
    
    parser.add_argument('--save_predictions', action='store_true')
    parser.add_argument('--save_visualizations', action='store_true')
    parser.add_argument('--output_dir', type=str, default='./results/')
    parser.add_argument('--num_vis_samples', type=int, default=4)
    
    return parser.parse_args()


def compute_metrics(predictions, targets, task_name):
    """计算评估指标"""
    metrics = {}
    
    if task_name == 'forecasting':
        mse = np.mean((predictions - targets) ** 2)
        mae = np.mean(np.abs(predictions - targets))
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((predictions - targets) / (targets + 1e-8))) * 100
        
        metrics = {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }
    
    elif task_name == 'classification':
        correct = (predictions == targets).sum()
        total = len(targets)
        accuracy = correct / total
        
        metrics = {
            'Accuracy': accuracy
        }
    
    elif task_name == 'imputation':
        mse = np.mean((predictions - targets) ** 2)
        mae = np.mean(np.abs(predictions - targets))
        
        metrics = {
            'MSE': mse,
            'MAE': mae
        }
    
    elif task_name == 'anomaly_detection':
        # 假设predictions和targets都是binary labels
        tp = ((predictions == 1) & (targets == 1)).sum()
        fp = ((predictions == 1) & (targets == 0)).sum()
        fn = ((predictions == 0) & (targets == 1)).sum()
        tn = ((predictions == 0) & (targets == 0)).sum()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        metrics = {
            'Precision': precision,
            'Recall': recall,
            'F1': f1
        }
    
    return metrics


def visualize_forecasting(inputs, predictions, targets, save_path, num_samples=4):
    """可视化预测结果"""
    num_samples = min(num_samples, inputs.shape[0])
    
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 3 * num_samples))
    if num_samples == 1:
        axes = [axes]
    
    for i in range(num_samples):
        ax = axes[i]
        
        # 绘制输入序列
        input_len = inputs.shape[1]
        pred_len = predictions.shape[1]
        
        ax.plot(range(input_len), inputs[i, :, 0], 'b-', label='Input', linewidth=2)
        ax.plot(range(input_len, input_len + pred_len), targets[i, :, 0], 'g-', label='Ground Truth', linewidth=2)
        ax.plot(range(input_len, input_len + pred_len), predictions[i, :, 0], 'r--', label='Prediction', linewidth=2)
        
        ax.axvline(x=input_len, color='gray', linestyle=':', alpha=0.5)
        ax.legend()
        ax.set_title(f'Sample {i+1}')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Visualization saved to {save_path}")


def evaluate(model, test_loader, device, args):
    """评估模型"""
    all_inputs = []
    all_predictions = []
    all_targets = []
    
    logger.info("Starting evaluation...")
    
    with torch.no_grad():
        for batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            batch_x = batch_x.to(device).float()
            batch_y = batch_y.to(device).float()
            
            # 通过MAS4TS处理
            result = model.process(batch_x, args.task_name, {'pred_len': args.pred_len})
            
            if result['success']:
                predictions = result['predictions']
                
                all_inputs.append(batch_x.cpu().numpy())
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())
                
                if (batch_idx + 1) % 50 == 0:
                    logger.info(f"Processed {batch_idx + 1}/{len(test_loader)} batches")
    
    if not all_predictions:
        logger.error("No predictions generated")
        return None
    
    # 合并结果
    inputs = np.concatenate(all_inputs, axis=0)
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    logger.info(f"Evaluation completed. Total samples: {len(predictions)}")
    
    # 计算指标
    metrics = compute_metrics(predictions, targets, args.task_name)
    
    logger.info("=" * 50)
    logger.info("Evaluation Metrics:")
    for metric_name, value in metrics.items():
        logger.info(f"{metric_name}: {value:.4f}")
    logger.info("=" * 50)
    
    # 保存结果
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.save_predictions:
        pred_path = output_dir / f"predictions_{args.data}_{args.task_name}.npz"
        np.savez(pred_path,
                inputs=inputs,
                predictions=predictions,
                targets=targets,
                metrics=metrics)
        logger.info(f"Predictions saved to {pred_path}")
    
    if args.save_visualizations and args.task_name == 'forecasting':
        vis_path = output_dir / f"visualization_{args.data}_{args.task_name}.png"
        visualize_forecasting(inputs, predictions, targets, vis_path, args.num_vis_samples)
    
    # 保存指标到JSON
    metrics_path = output_dir / f"metrics_{args.data}_{args.task_name}.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")
    
    return metrics


def main():
    """主函数"""
    args = parse_args()
    
    # 设置设备
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    
    # 加载配置
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except:
        config = DEFAULT_CONFIG
    
    # 设置日志
    setup_logger('INFO', config.get('logging', {}))
    
    logger.info("=" * 50)
    logger.info("MAS4TS Evaluation Script")
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
    
    # 加载checkpoint（如果提供）
    if args.checkpoint:
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        model.load_checkpoint(args.checkpoint)
    
    # 准备数据
    logger.info("Preparing data loader...")
    dataset_config = config.get('datasets', {}).get(args.data, {})
    
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
    
    test_loader = data_provider(data_dict, 'test')
    logger.info(f"Test batches: {len(test_loader)}")
    
    # 评估
    metrics = evaluate(model, test_loader, device, args)
    
    if metrics:
        logger.info("Evaluation completed successfully!")
    else:
        logger.error("Evaluation failed!")


if __name__ == '__main__':
    main()

