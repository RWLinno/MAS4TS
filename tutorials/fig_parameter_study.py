"""
参数敏感性研究图
展示关键超参数对模型性能的影响
"""

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 9


def create_parameter_study():
    """创建参数敏感性研究图"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Parameter Sensitivity Analysis', fontsize=14, fontweight='bold')
    
    # ===== 参数1: Top-K Features =====
    ax1 = axes[0, 0]
    k_values = [3, 5, 7, 10, 15, 20, 'All']
    k_numeric = [3, 5, 7, 10, 15, 20, 25]
    mse_values = [0.425, 0.385, 0.352, 0.338, 0.341, 0.348, 0.355]
    
    ax1.plot(k_numeric, mse_values, 'o-', linewidth=2.5, markersize=8, 
            color='#2E86AB', markerfacecolor='#89CFF0', markeredgewidth=2)
    ax1.axvline(x=10, color='r', linestyle='--', alpha=0.6, linewidth=2)
    ax1.text(10, 0.43, 'Optimal: K=10', ha='center', fontsize=8, 
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    ax1.set_xlabel('Number of Selected Features (K)', fontsize=10, fontweight='bold')
    ax1.set_ylabel('MSE', fontsize=10, fontweight='bold')
    ax1.set_title('(a) Impact of Top-K Feature Selection', fontsize=10, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xticks(k_numeric)
    ax1.set_xticklabels(k_values, fontsize=8)
    
    # ===== 参数2: VLM Temperature =====
    ax2 = axes[0, 1]
    temp_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    consistency = [0.68, 0.85, 0.82, 0.75, 0.62]
    quality = [0.72, 0.88, 0.84, 0.78, 0.65]
    
    ax2.plot(temp_values, consistency, 's-', linewidth=2.5, markersize=8,
            color='#06A77D', label='Consistency', markeredgewidth=2)
    ax2.plot(temp_values, quality, 'D-', linewidth=2.5, markersize=8,
            color='#F77F00', label='Quality', markeredgewidth=2)
    ax2.axvline(x=0.3, color='r', linestyle='--', alpha=0.6, linewidth=2)
    
    ax2.set_xlabel('VLM Temperature', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Score', fontsize=10, fontweight='bold')
    ax2.set_title('(b) VLM Temperature vs. Output Quality', fontsize=10, fontweight='bold')
    ax2.legend(loc='lower left', fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylim(0.5, 1.0)
    
    # ===== 参数3: LLM Ensemble Size =====
    ax3 = axes[0, 2]
    ensemble_sizes = [1, 2, 3, 4, 5]
    mae_values = [0.458, 0.421, 0.398, 0.402, 0.405]
    inference_time = [1.2, 2.1, 2.8, 3.9, 5.2]
    
    ax3_twin = ax3.twinx()
    
    line1 = ax3.plot(ensemble_sizes, mae_values, 'o-', linewidth=2.5, markersize=8,
                    color='#D62828', label='MAE (↓)', markerfacecolor='#FF6B6B', markeredgewidth=2)
    line2 = ax3_twin.plot(ensemble_sizes, inference_time, 's-', linewidth=2.5, markersize=8,
                         color='#4361EE', label='Inference Time (↑)', 
                         markerfacecolor='#7B9EFF', markeredgewidth=2)
    
    ax3.axvline(x=3, color='g', linestyle='--', alpha=0.6, linewidth=2)
    ax3.text(3, 0.47, 'Best\nTrade-off', ha='center', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    ax3.set_xlabel('Number of LLM Models in Ensemble', fontsize=10, fontweight='bold')
    ax3.set_ylabel('MAE (Lower Better)', fontsize=9, fontweight='bold', color='#D62828')
    ax3_twin.set_ylabel('Inference Time (s)', fontsize=9, fontweight='bold', color='#4361EE')
    ax3.set_title('(c) LLM Ensemble Size vs. Performance', fontsize=10, fontweight='bold')
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='upper left', fontsize=9)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_xticks(ensemble_sizes)
    
    # ===== 参数4: Confidence Level =====
    ax4 = axes[1, 0]
    conf_levels = [0.80, 0.85, 0.90, 0.95, 0.99]
    coverage = [0.78, 0.84, 0.89, 0.94, 0.98]
    sharpness = [0.92, 0.88, 0.82, 0.78, 0.68]
    
    ax4.plot(conf_levels, coverage, 'o-', linewidth=2.5, markersize=8,
            color='#06A77D', label='Coverage', markeredgewidth=2)
    ax4.plot(conf_levels, sharpness, 's-', linewidth=2.5, markersize=8,
            color='#F77F00', label='Sharpness', markeredgewidth=2)
    ax4.plot(conf_levels, conf_levels, 'k--', alpha=0.5, linewidth=1.5, label='Ideal Coverage')
    
    ax4.set_xlabel('Confidence Level', fontsize=10, fontweight='bold')
    ax4.set_ylabel('Score', fontsize=10, fontweight='bold')
    ax4.set_title('(d) Confidence Level vs. Interval Quality', fontsize=10, fontweight='bold')
    ax4.legend(loc='center left', fontsize=9)
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.set_ylim(0.6, 1.05)
    
    # ===== 参数5: Batch Parallel Split =====
    ax5 = axes[1, 1]
    batch_sizes = [8, 16, 32, 64, 128]
    serial_time = [2.1, 4.3, 8.9, 18.2, 37.5]
    parallel_2 = [1.2, 2.5, 5.1, 10.5, 21.3]
    parallel_4 = [0.8, 1.6, 3.2, 6.8, 13.9]
    parallel_8 = [0.6, 1.1, 2.3, 4.9, 10.2]
    
    ax5.plot(batch_sizes, serial_time, 'o-', linewidth=2, markersize=7,
            color='#95A5A6', label='Serial', markeredgewidth=1.5)
    ax5.plot(batch_sizes, parallel_2, 's-', linewidth=2, markersize=7,
            color='#3498DB', label='Parallel (2 splits)', markeredgewidth=1.5)
    ax5.plot(batch_sizes, parallel_4, 'D-', linewidth=2, markersize=7,
            color='#2ECC71', label='Parallel (4 splits)', markeredgewidth=1.5)
    ax5.plot(batch_sizes, parallel_8, '^-', linewidth=2, markersize=7,
            color='#E74C3C', label='Parallel (8 splits)', markeredgewidth=1.5)
    
    ax5.set_xlabel('Batch Size', fontsize=10, fontweight='bold')
    ax5.set_ylabel('Inference Time (s)', fontsize=10, fontweight='bold')
    ax5.set_title('(e) Batch Parallel Efficiency', fontsize=10, fontweight='bold')
    ax5.legend(loc='upper left', fontsize=8)
    ax5.grid(True, alpha=0.3, linestyle='--')
    ax5.set_yscale('log')
    
    # ===== 参数6: Anchor Strategy =====
    ax6 = axes[1, 2]
    strategies = ['Rule-based', 'Statistical', 'VLM-only', 'Hybrid\n(Ours)']
    accuracy = [0.78, 0.84, 0.88, 0.93]
    reliability = [0.85, 0.88, 0.82, 0.91]
    
    x = np.arange(len(strategies))
    width = 0.35
    
    bars1 = ax6.bar(x - width/2, accuracy, width, label='Prediction Accuracy',
                   color='#4361EE', alpha=0.8, edgecolor='black', linewidth=0.8)
    bars2 = ax6.bar(x + width/2, reliability, width, label='Anchor Reliability',
                   color='#9D4EDD', alpha=0.8, edgecolor='black', linewidth=0.8)
    
    # 标注
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    ax6.set_ylabel('Score', fontsize=10, fontweight='bold')
    ax6.set_title('(f) Anchor Strategy Comparison', fontsize=10, fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels(strategies, fontsize=9)
    ax6.legend(loc='lower right', fontsize=9)
    ax6.set_ylim(0, 1.05)
    ax6.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('tutorials/parameter_study.png', dpi=300, bbox_inches='tight')
    plt.savefig('tutorials/parameter_study.pdf', bbox_inches='tight')
    print("✓ Parameter study saved: tutorials/parameter_study.png/pdf")
    plt.close()


if __name__ == '__main__':
    create_parameter_study()

