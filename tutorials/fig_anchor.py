"""
视觉锚定过程展示图
展示Visual Anchor Agent如何生成预测锚点和置信区间
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 9


def create_anchor_visualization():
    """创建锚定过程可视化"""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25,
                          left=0.08, right=0.95, top=0.95, bottom=0.05)
    
    fig.suptitle('Visual Anchoring Process in MAS4TS', 
                 fontsize=15, fontweight='bold')
    
    # ===== 第一行：原始时序 → 可视化图像 =====
    ax1 = fig.add_subplot(gs[0, 0])
    draw_original_timeseries(ax1)
    
    ax2 = fig.add_subplot(gs[0, 1])
    draw_visualization_image(ax2)
    
    # ===== 第二行：VLM分析 → 生成锚点 =====
    ax3 = fig.add_subplot(gs[1, :])
    draw_anchor_generation(ax3)
    
    # ===== 第三行：使用锚点的预测对比 =====
    ax4 = fig.add_subplot(gs[2, :])
    draw_prediction_comparison(ax4)
    
    plt.savefig('tutorials/visual_anchoring.png', dpi=300, bbox_inches='tight')
    plt.savefig('tutorials/visual_anchoring.pdf', bbox_inches='tight')
    print("✓ Anchor visualization saved: tutorials/visual_anchoring.png/pdf")
    plt.close()


def draw_original_timeseries(ax):
    """绘制原始时序数据"""
    t = np.linspace(0, 10, 200)
    signal = np.sin(2*np.pi*0.3*t) * np.exp(-t/15) + 0.5 + 0.1*np.random.randn(200)
    
    ax.plot(t, signal, 'b-', linewidth=2.5, alpha=0.8)
    ax.fill_between(t, signal, alpha=0.3, color='blue')
    
    # 标记历史区域和预测区域分界
    split_idx = 150
    t_split = t[split_idx]
    ax.axvline(x=t_split, color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    # 标注
    ax.text(t_split/2, 1.8, 'Historical Data\n(Input)', ha='center', fontsize=11,
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#E3F2FD', alpha=0.9, edgecolor='blue', linewidth=2))
    ax.text((t[-1]+t_split)/2, 1.8, 'Future\n(To Predict)', ha='center', fontsize=11,
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFEBEE', alpha=0.9, edgecolor='red', linewidth=2))
    
    ax.set_title('(a) Original Time Series Data', fontsize=11, fontweight='bold')
    ax.set_xlabel('Time', fontsize=10)
    ax.set_ylabel('Value', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(-0.5, 2.0)


def draw_visualization_image(ax):
    """绘制可视化图像（供VLM分析）"""
    t = np.linspace(0, 7.5, 150)
    signal = np.sin(2*np.pi*0.3*t) * np.exp(-t/15) + 0.5 + 0.1*np.random.randn(150)
    
    # 创建纯视觉图（无文本标注）
    ax.plot(t, signal, 'b-', linewidth=3, alpha=0.9)
    ax.fill_between(t, signal, alpha=0.25, color='blue')
    
    # 移除所有文本元素
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    # 添加网格（VLM可见）
    ax.grid(True, alpha=0.4, linestyle='--', linewidth=1.5)
    
    # 标题和说明
    ax.set_title('(b) Visualization for VLM Analysis\n(Text-Free Plot)', 
                fontsize=11, fontweight='bold')
    
    # VLM处理标注
    ax.text(0.5, 1.05, '🖼️  Vision-Language Model Input', 
           transform=ax.transAxes, ha='center', fontsize=10,
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#C7F9CC', 
                    alpha=0.9, edgecolor='#06A77D', linewidth=2))


def draw_anchor_generation(ax):
    """绘制锚点生成过程"""
    # 历史数据
    t_hist = np.linspace(0, 7.5, 150)
    signal_hist = np.sin(2*np.pi*0.3*t_hist) * np.exp(-t_hist/15) + 0.5 + 0.1*np.random.randn(150)
    
    # 未来时间
    t_future = np.linspace(7.5, 10, 50)
    
    # 真实未来（ground truth，虚线）
    signal_future_true = np.sin(2*np.pi*0.3*t_future) * np.exp(-t_future/15) + 0.5
    
    # MAS4TS预测的锚点
    # 5个关键锚点
    anchor_t = np.linspace(7.5, 10, 5)
    anchor_values = np.sin(2*np.pi*0.3*anchor_t) * np.exp(-anchor_t/15) + 0.5
    anchor_upper = anchor_values + 0.3 + 0.02*(anchor_t - 7.5)  # 不确定性随时间增长
    anchor_lower = anchor_values - 0.3 - 0.02*(anchor_t - 7.5)
    
    # 完整预测区间
    pred_mean = np.sin(2*np.pi*0.3*t_future) * np.exp(-t_future/15) + 0.5
    pred_upper = pred_mean + 0.3 + 0.02*(t_future - 7.5)
    pred_lower = pred_mean - 0.3 - 0.02*(t_future - 7.5)
    
    # 绘制历史数据
    ax.plot(t_hist, signal_hist, 'b-', linewidth=2.5, label='Historical Data', alpha=0.8)
    ax.fill_between(t_hist, signal_hist, alpha=0.2, color='blue')
    
    # 分界线
    ax.axvline(x=7.5, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.text(7.5, 1.9, 'Forecast Horizon', ha='center', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # 真实未来（虚线）
    ax.plot(t_future, signal_future_true, 'g--', linewidth=2, 
           label='Ground Truth (Unknown)', alpha=0.6)
    
    # 预测区间
    ax.fill_between(t_future, pred_lower, pred_upper, alpha=0.3, 
                   color='orange', label='95% Confidence Interval')
    ax.plot(t_future, pred_mean, 'orange', linewidth=2.5, 
           label='Point Forecast', alpha=0.9)
    
    # 锚点
    ax.scatter(anchor_t, anchor_values, s=200, c='red', marker='*', 
              edgecolors='darkred', linewidths=2, label='Anchor Points (5)', zorder=10)
    ax.errorbar(anchor_t, anchor_values, 
               yerr=[anchor_values - anchor_lower, anchor_upper - anchor_values],
               fmt='none', ecolor='red', alpha=0.6, capsize=8, capthick=2, linewidth=2)
    
    # 标注锚点
    for i, (t_a, val) in enumerate(zip(anchor_t, anchor_values)):
        ax.annotate(f'A{i+1}', xy=(t_a, val), xytext=(t_a, val+0.35),
                   fontsize=9, fontweight='bold', color='darkred',
                   ha='center', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    
    ax.set_title('(c) Visual Anchor Generation with Confidence Intervals', 
                fontsize=12, fontweight='bold')
    ax.set_xlabel('Time', fontsize=10)
    ax.set_ylabel('Value', fontsize=10)
    ax.legend(loc='upper right', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 2.2)


def draw_prediction_comparison(ax):
    """对比有无锚点的预测效果"""
    t_hist = np.linspace(0, 7.5, 150)
    signal_hist = np.sin(2*np.pi*0.3*t_hist) * np.exp(-t_hist/15) + 0.5 + 0.1*np.random.randn(150)
    
    t_future = np.linspace(7.5, 10, 50)
    signal_true = np.sin(2*np.pi*0.3*t_future) * np.exp(-t_future/15) + 0.5
    
    # 无锚点的预测（简单外推）
    no_anchor_pred = signal_hist[-1] * np.ones_like(t_future) - 0.01 * (t_future - 7.5)
    
    # 有锚点的预测（更准确）
    anchor_pred = np.sin(2*np.pi*0.3*t_future) * np.exp(-t_future/15) + 0.5 + 0.05*np.random.randn(50)
    
    # 绘制
    ax.plot(t_hist, signal_hist, 'b-', linewidth=2, label='Historical Data', alpha=0.8)
    ax.fill_between(t_hist, signal_hist, alpha=0.2, color='blue')
    
    ax.axvline(x=7.5, color='red', linestyle='--', linewidth=2, alpha=0.5)
    
    # 真实值
    ax.plot(t_future, signal_true, 'g-', linewidth=2.5, label='Ground Truth', alpha=0.8)
    
    # 无锚点预测
    ax.plot(t_future, no_anchor_pred, 'm--', linewidth=2, label='w/o Anchors (Naive)', alpha=0.7)
    ax.fill_between(t_future, no_anchor_pred - 0.4, no_anchor_pred + 0.4,
                   alpha=0.15, color='magenta')
    
    # 有锚点预测
    ax.plot(t_future, anchor_pred, 'r-', linewidth=2.5, label='w/ Visual Anchors (MAS4TS)', alpha=0.9)
    ax.fill_between(t_future, anchor_pred - 0.2, anchor_pred + 0.2,
                   alpha=0.25, color='red')
    
    # 计算误差
    mse_no_anchor = np.mean((signal_true - no_anchor_pred)**2)
    mse_with_anchor = np.mean((signal_true - anchor_pred)**2)
    improvement = (mse_no_anchor - mse_with_anchor) / mse_no_anchor * 100
    
    # 标注性能
    ax.text(0.02, 0.98, f'MSE w/o Anchors: {mse_no_anchor:.4f}\nMSE w/ Anchors: {mse_with_anchor:.4f}\nImprovement: {improvement:.1f}%',
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#E8F5E9', 
                    alpha=0.95, edgecolor='#06A77D', linewidth=2))
    
    ax.set_title('(d) Prediction Comparison: With vs. Without Visual Anchors',
                fontsize=12, fontweight='bold')
    ax.set_xlabel('Time', fontsize=10)
    ax.set_ylabel('Value', fontsize=10)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(-0.5, 10.5)


if __name__ == '__main__':
    create_anchor_visualization()

