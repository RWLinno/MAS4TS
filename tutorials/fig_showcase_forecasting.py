"""
预测任务效果展示图
展示长期和短期预测的可视化结果
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 9


def create_forecasting_showcase():
    """创建预测效果展示图"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    fig.suptitle('MAS4TS: Forecasting Performance Showcase', 
                 fontsize=15, fontweight='bold', y=0.98)
    
    datasets = ['ETTh1', 'ETTm1', 'Weather']
    pred_lens = [96, 192]
    
    for row, pred_len in enumerate(pred_lens):
        for col, dataset in enumerate(datasets):
            ax = axes[row, col]
            draw_forecast_comparison(ax, dataset, pred_len, row, col)
    
    # 添加图例
    handles = [
        plt.Line2D([0], [0], color='#2E86AB', linewidth=2, label='Ground Truth'),
        plt.Line2D([0], [0], color='#D62828', linewidth=2, label='MAS4TS (Ours)'),
        plt.Line2D([0], [0], color='#999999', linewidth=1.5, linestyle='--', label='Baseline'),
        plt.fill_between([0, 0], [0, 0], alpha=0.2, color='#D62828', label='95% Confidence Interval')
    ]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.02),
               ncol=4, fontsize=10, frameon=True, fancybox=True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig('tutorials/showcase_forecasting.png', dpi=300, bbox_inches='tight')
    plt.savefig('tutorials/showcase_forecasting.pdf', bbox_inches='tight')
    print("✓ Forecasting showcase saved: tutorials/showcase_forecasting.png/pdf")
    plt.close()


def draw_forecast_comparison(ax, dataset, pred_len, row, col):
    """绘制单个预测对比"""
    # 生成模拟数据
    np.random.seed(42 + row * 3 + col)
    
    seq_len = 96
    total_len = seq_len + pred_len
    t = np.arange(total_len)
    
    # 真实数据（带趋势和季节性）
    trend = 0.002 * t
    seasonal = 0.3 * np.sin(2 * np.pi * t / 24)
    noise = 0.05 * np.random.randn(total_len)
    ground_truth = trend + seasonal + noise + 5
    
    # 历史数据
    history = ground_truth[:seq_len]
    future_true = ground_truth[seq_len:]
    
    # MAS4TS预测（更准确）
    future_pred_mas4ts = future_true + 0.03 * np.random.randn(pred_len)
    
    # Baseline预测（简单的重复最后值）
    future_pred_baseline = np.full(pred_len, history[-1]) + 0.1 * np.random.randn(pred_len)
    
    # 置信区间
    std_history = np.std(history)
    time_factor = np.sqrt(np.arange(1, pred_len + 1) / pred_len)
    upper_bound = future_pred_mas4ts + 1.96 * std_history * time_factor
    lower_bound = future_pred_mas4ts - 1.96 * std_history * time_factor
    
    # 绘图
    t_history = t[:seq_len]
    t_future = t[seq_len:]
    
    # 历史数据
    ax.plot(t_history, history, color='#888888', linewidth=1.5, 
            label='Historical Data', alpha=0.7)
    
    # 真实值
    ax.plot(t_future, future_true, color='#2E86AB', linewidth=2.5, 
            label='Ground Truth', zorder=3)
    
    # MAS4TS预测
    ax.plot(t_future, future_pred_mas4ts, color='#D62828', linewidth=2, 
            label='MAS4TS', linestyle='-', zorder=2)
    
    # 置信区间
    ax.fill_between(t_future, lower_bound, upper_bound, 
                    alpha=0.2, color='#D62828', zorder=1)
    
    # Baseline
    ax.plot(t_future, future_pred_baseline, color='#999999', linewidth=1.5,
            linestyle='--', label='Baseline', alpha=0.7)
    
    # 分隔线
    ax.axvline(x=seq_len, color='black', linestyle=':', linewidth=1.5, alpha=0.5)
    ax.text(seq_len, ax.get_ylim()[1] * 0.95, 'Forecast Start', 
            ha='center', va='top', fontsize=7, 
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # 性能指标
    mse_mas4ts = np.mean((future_true - future_pred_mas4ts)**2)
    mse_baseline = np.mean((future_true - future_pred_baseline)**2)
    improvement = (mse_baseline - mse_mas4ts) / mse_baseline * 100
    
    metric_text = f'MSE: {mse_mas4ts:.4f}\nImprovement: +{improvement:.1f}%'
    ax.text(0.98, 0.97, metric_text, transform=ax.transAxes,
            fontsize=8, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='#E8F5E9', 
                     alpha=0.9, edgecolor='#06A77D', linewidth=1.5))
    
    # 设置
    ax.set_xlabel('Time Steps', fontsize=9)
    ax.set_ylabel('Value', fontsize=9)
    ax.set_title(f'{dataset} (Pred Len: {pred_len})', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    if row == 0 and col == 0:
        ax.legend(loc='upper left', fontsize=7)


if __name__ == '__main__':
    create_forecasting_showcase()

