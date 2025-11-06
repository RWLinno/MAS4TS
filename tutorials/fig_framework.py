"""
框架图
展示MAS4TS多智能体系统如何处理时序数据并应用于不同任务
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import numpy as np

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.unicode_minus'] = False


def create_framework_figure():
    """创建框架图"""
    fig, axes = plt.subplots(2, 1, figsize=(16, 12), 
                             gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.25})
    
    fig.suptitle('MAS4TS: Multi-Agent System Framework for Time Series Analysis', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # 上半部分：主要框架
    draw_main_framework(axes[0])
    
    # 下半部分：应用场景
    draw_applications(axes[1])
    
    plt.savefig('tutorials/framework.png', dpi=300, bbox_inches='tight')
    plt.savefig('tutorials/framework.pdf', bbox_inches='tight')
    print("✓ Framework figure saved: tutorials/framework.png/pdf")
    plt.close()


def draw_main_framework(ax):
    """绘制主要框架流程"""
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 12)
    ax.axis('off')
    ax.set_title('Multi-Agent Collaborative Process', fontsize=13, 
                 fontweight='bold', pad=15, loc='left')
    
    # ===== Stage 0: Input =====
    input_box = FancyBboxPatch((0.5, 9), 2.5, 2, boxstyle="round,pad=0.15",
                               edgecolor='#2C3E50', facecolor='#E8F4F8', linewidth=2.5)
    ax.add_patch(input_box)
    ax.text(1.75, 10.6, 'Input', ha='center', va='center', 
            fontsize=11, fontweight='bold')
    ax.text(1.75, 10, 'Time Series', ha='center', va='center', fontsize=9)
    ax.text(1.75, 9.5, '[B, L, D]', ha='center', va='center', 
            fontsize=8, style='italic', color='#555')
    
    # ===== Stage 1: Data Analyzer Agent =====
    stage1_y = 9.5
    agent1_box = FancyBboxPatch((3.8, stage1_y - 0.5), 3, 2, 
                                boxstyle="round,pad=0.15",
                                edgecolor='#F77F00', facecolor='#FFF3E0', linewidth=2.5)
    ax.add_patch(agent1_box)
    
    ax.text(5.3, stage1_y + 1.2, '① Data Analyzer Agent', ha='center', va='center',
            fontsize=10, fontweight='bold', color='#F77F00')
    ax.text(5.3, stage1_y + 0.7, '• Statistical Analysis', ha='center', va='center', fontsize=8)
    ax.text(5.3, stage1_y + 0.3, '• Feature Extraction', ha='center', va='center', fontsize=8)
    ax.text(5.3, stage1_y - 0.1, '• Covariance Analysis', ha='center', va='center', fontsize=8)
    ax.text(5.3, stage1_y - 0.5, '• Generate Plot', ha='center', va='center', fontsize=8)
    
    # 输出：Plot + Features
    ax.text(5.3, stage1_y - 1, '📊 Plot + Features', ha='center', va='center',
            fontsize=8, style='italic', bbox=dict(boxstyle='round', 
            facecolor='#FFE5CC', alpha=0.6, pad=0.3))
    
    # ===== Stage 2: Visual Anchor Agent (VLM) =====
    stage2_y = 6.5
    agent2_box = FancyBboxPatch((3.8, stage2_y - 0.5), 3, 2,
                                boxstyle="round,pad=0.15",
                                edgecolor='#06A77D', facecolor='#E8F5E9', linewidth=2.5)
    ax.add_patch(agent2_box)
    
    ax.text(5.3, stage2_y + 1.2, '② Visual Anchor Agent', ha='center', va='center',
            fontsize=10, fontweight='bold', color='#06A77D')
    ax.text(5.3, stage2_y + 0.7, '🖼️  Vision-Language Model', ha='center', va='center', fontsize=8)
    ax.text(5.3, stage2_y + 0.2, '• Visual Pattern Recognition', ha='center', va='center', fontsize=8)
    ax.text(5.3, stage2_y - 0.2, '• Generate Prediction Anchors', ha='center', va='center', fontsize=8)
    ax.text(5.3, stage2_y - 0.6, '• Confidence Intervals', ha='center', va='center', fontsize=8)
    
    # 输出：Anchors
    ax.text(5.3, stage2_y - 1.1, '⚓ Visual Anchors', ha='center', va='center',
            fontsize=8, style='italic', bbox=dict(boxstyle='round',
            facecolor='#C7F9CC', alpha=0.6, pad=0.3))
    
    # ===== Stage 3: Numerologic Adapter Agent (LLM) =====
    stage3_y = 3.5
    agent3_box = FancyBboxPatch((3.8, stage3_y - 0.5), 3, 2,
                                boxstyle="round,pad=0.15",
                                edgecolor='#4361EE', facecolor='#E3F2FD', linewidth=2.5)
    ax.add_patch(agent3_box)
    
    ax.text(5.3, stage3_y + 1.2, '③ Numerologic Adapter', ha='center', va='center',
            fontsize=10, fontweight='bold', color='#4361EE')
    ax.text(5.3, stage3_y + 0.7, '🤖 Large Language Model', ha='center', va='center', fontsize=8)
    ax.text(5.3, stage3_y + 0.2, '• Numerical Reasoning', ha='center', va='center', fontsize=8)
    ax.text(5.3, stage3_y - 0.2, '• Multi-Modal Fusion', ha='center', va='center', fontsize=8)
    ax.text(5.3, stage3_y - 0.6, '• LLM Ensemble (3 models)', ha='center', va='center', fontsize=8)
    
    # 输出：Adapted Features
    ax.text(5.3, stage3_y - 1.1, '🔢 Numerical Constraints', ha='center', va='center',
            fontsize=8, style='italic', bbox=dict(boxstyle='round',
            facecolor='#BBDEFB', alpha=0.6, pad=0.3))
    
    # ===== Stage 4: Task Executor Agent =====
    executor_box = FancyBboxPatch((8, 5), 3.5, 4,
                                  boxstyle="round,pad=0.2",
                                  edgecolor='#9D4EDD', facecolor='#F3E5F5', linewidth=2.5)
    ax.add_patch(executor_box)
    
    ax.text(9.75, 8.5, '④ Task Executor Agent', ha='center', va='center',
            fontsize=10, fontweight='bold', color='#9D4EDD')
    ax.text(9.75, 7.8, 'Integrate All Priors', ha='center', va='center', 
            fontsize=9, style='italic')
    ax.text(9.75, 7.2, '• Statistical Features', ha='center', va='center', fontsize=8)
    ax.text(9.75, 6.7, '• Visual Anchors', ha='center', va='center', fontsize=8)
    ax.text(9.75, 6.2, '• Numerical Constraints', ha='center', va='center', fontsize=8)
    ax.text(9.75, 5.6, '→ Final Predictions', ha='center', va='center', 
            fontsize=9, fontweight='bold', color='#9D4EDD')
    
    # ===== Output =====
    output_box = FancyBboxPatch((12.5, 9), 2.5, 2, boxstyle="round,pad=0.15",
                                edgecolor='#2C3E50', facecolor='#E8F4F8', linewidth=2.5)
    ax.add_patch(output_box)
    ax.text(13.75, 10.6, 'Output', ha='center', va='center',
            fontsize=11, fontweight='bold')
    ax.text(13.75, 10, 'Task-Specific', ha='center', va='center', fontsize=9)
    ax.text(13.75, 9.5, 'Results', ha='center', va='center', fontsize=9)
    
    # ===== Manager Agent (中央协调器) =====
    manager_box = FancyBboxPatch((16, 7), 3, 3, boxstyle="round,pad=0.2",
                                 edgecolor='#D62828', facecolor='#FFEBEE', 
                                 linewidth=3, linestyle='--')
    ax.add_patch(manager_box)
    ax.text(17.5, 9.3, 'Manager Agent', ha='center', va='center',
            fontsize=11, fontweight='bold', color='#D62828')
    ax.text(17.5, 8.7, '(Central Coordinator)', ha='center', va='center',
            fontsize=9, style='italic')
    ax.text(17.5, 8.1, '• Task Decomposition', ha='center', va='center', fontsize=8)
    ax.text(17.5, 7.7, '• Agent Scheduling', ha='center', va='center', fontsize=8)
    ax.text(17.5, 7.3, '• Result Integration', ha='center', va='center', fontsize=8)
    
    # ===== 数据流箭头 =====
    # Input → Stage 1
    arrow1 = FancyArrowPatch((3, 10), (3.8, 10), arrowstyle='->', 
                            mutation_scale=25, linewidth=2.5, color='#34495E')
    ax.add_patch(arrow1)
    ax.text(3.4, 10.3, 'Raw TS', ha='center', fontsize=8, color='#555')
    
    # Stage 1 → Stage 2
    arrow2 = FancyArrowPatch((5.3, 9), (5.3, 8), arrowstyle='->', 
                            mutation_scale=25, linewidth=2.5, color='#F77F00')
    ax.add_patch(arrow2)
    ax.text(5.8, 8.5, 'Plot +\nStatistics', ha='left', fontsize=7, color='#F77F00')
    
    # Stage 2 → Stage 3
    arrow3 = FancyArrowPatch((5.3, 6), (5.3, 5.5), arrowstyle='->', 
                            mutation_scale=25, linewidth=2.5, color='#06A77D')
    ax.add_patch(arrow3)
    ax.text(5.8, 5.75, 'Anchors +\nPriors', ha='left', fontsize=7, color='#06A77D')
    
    # Stage 3 → Executor
    arrow4 = FancyArrowPatch((6.8, 4), (8, 7), arrowstyle='->', 
                            mutation_scale=25, linewidth=2.5, color='#4361EE')
    ax.add_patch(arrow4)
    ax.text(7.2, 5.3, 'Numerical\nConstraints', ha='center', fontsize=7, color='#4361EE')
    
    # Executor → Output
    arrow5 = FancyArrowPatch((11.5, 10), (12.5, 10), arrowstyle='->', 
                            mutation_scale=25, linewidth=2.5, color='#9D4EDD')
    ax.add_patch(arrow5)
    ax.text(12, 10.3, 'Final', ha='center', fontsize=8, color='#9D4EDD')
    
    # Manager协调线（虚线）
    for agent_x in [5.3, 5.3, 5.3, 9.75]:
        for agent_y in [10, 7, 4, 7]:
            ax.plot([17.5, agent_x], [8.5, agent_y], 'r--', alpha=0.3, linewidth=1.2)
    
    # ===== 关键创新标注 =====
    innovations = [
        (5.3, 11.5, '💡 Innovation 1:\nVisual-Semantic Anchoring'),
        (9.75, 10.5, '💡 Innovation 2:\nMulti-Agent Collaboration'),
        (13.75, 11.5, '💡 Innovation 3:\nTask-Adaptive Integration')
    ]
    
    for x, y, text in innovations:
        ax.text(x, y, text, ha='center', va='center', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF9C4', 
                         alpha=0.9, edgecolor='#F57F17', linewidth=1.5))


def draw_applications(ax):
    """绘制应用场景"""
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 5)
    ax.axis('off')
    ax.set_title('Downstream Applications', fontsize=13, fontweight='bold', 
                 pad=10, loc='left')
    
    tasks = [
        ('Long-term\nForecasting', '#2E86AB', 'Predict future\n96-720 steps'),
        ('Short-term\nForecasting', '#0077B6', 'Predict future\n1-96 steps'),
        ('Classification', '#7209B7', 'Classify TS\npatterns'),
        ('Imputation', '#06A77D', 'Fill missing\nvalues'),
        ('Anomaly\nDetection', '#D62828', 'Detect\noutliers')
    ]
    
    # 中央系统
    center_x, center_y = 10, 2.5
    center_circle = Circle((center_x, center_y), 0.8, 
                          color='#9D4EDD', alpha=0.3, zorder=1)
    ax.add_patch(center_circle)
    ax.text(center_x, center_y, 'MAS4TS\nSystem', ha='center', va='center',
            fontsize=9, fontweight='bold', color='#9D4EDD', zorder=2)
    
    # 任务分布（圆形分布）
    num_tasks = len(tasks)
    angles = np.linspace(0, 2*np.pi, num_tasks, endpoint=False)
    radius = 5
    
    for i, (task_name, color, desc) in enumerate(tasks):
        angle = angles[i]
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        
        # 任务框
        task_box = FancyBboxPatch((x-0.9, y-0.5), 1.8, 1, 
                                  boxstyle="round,pad=0.12",
                                  edgecolor=color, facecolor=color, 
                                  alpha=0.2, linewidth=2)
        ax.add_patch(task_box)
        
        ax.text(x, y + 0.25, task_name, ha='center', va='center',
                fontsize=9, fontweight='bold', color=color)
        ax.text(x, y - 0.15, desc, ha='center', va='center',
                fontsize=7, color='#555')
        
        # 连接线
        ax.plot([center_x, x], [center_y, y], color=color, 
                linewidth=2, alpha=0.5, linestyle='--')
        
        # 箭头
        arrow = FancyArrowPatch((center_x + 0.8*np.cos(angle), 
                                center_y + 0.8*np.sin(angle)),
                               (x - 0.9*np.cos(angle), y - 0.5*np.sin(angle)),
                               arrowstyle='->', mutation_scale=15, 
                               linewidth=2, color=color, alpha=0.7)
        ax.add_patch(arrow)
    
    # 底部说明
    ax.text(10, 0.3, 'Unified Framework for Multiple Time Series Tasks', 
            ha='center', va='center', fontsize=10, style='italic',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#F0F0F0', 
                     alpha=0.8, edgecolor='#999', linewidth=1))


if __name__ == '__main__':
    create_framework_figure()

