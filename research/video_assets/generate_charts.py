#!/usr/bin/env python3
"""Generate charts and graphics for PROMETHEUS video."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Create output directory
output_dir = Path("/home/user/aimo3/video_assets")
output_dir.mkdir(exist_ok=True)

# Style settings
plt.style.use('dark_background')
COLORS = {
    'primary': '#FF6B35',      # Orange fire
    'secondary': '#4ECDC4',    # Teal
    'accent': '#FFE66D',       # Yellow
    'bg': '#1a1a2e',           # Dark blue
    'text': '#FFFFFF',
    'grid': '#333355'
}

# =============================================================================
# CHART 1: Before/After Results (Hero Chart)
# =============================================================================
def create_hero_results_chart():
    fig, ax = plt.subplots(figsize=(16, 9), facecolor=COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])

    metrics = ['Format\nCompliance', 'Reasoning\nTraces', 'Coherence\nScore']
    before = [5.5, 11.0, 1.0]
    after = [89.0, 98.5, 1.23]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(x - width/2, before, width, label='Before Training',
                   color='#555555', edgecolor='white', linewidth=2)
    bars2 = ax.bar(x + width/2, after, width, label='After PROMETHEUS',
                   color=COLORS['primary'], edgecolor='white', linewidth=2)

    # Add value labels
    for bar, val in zip(bars1, before):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val}%' if val > 2 else f'{val}', ha='center', va='bottom',
                fontsize=20, fontweight='bold', color='#888888')

    for bar, val in zip(bars2, after):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val}%' if val > 2 else f'{val:.2f}', ha='center', va='bottom',
                fontsize=24, fontweight='bold', color=COLORS['primary'])

    ax.set_ylabel('Score', fontsize=18, color='white')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=18, color='white')
    ax.legend(loc='upper left', fontsize=16, framealpha=0.8)
    ax.set_ylim(0, 120)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.tick_params(colors='white')

    plt.title('PROMETHEUS Results: Single TPU Session', fontsize=28,
              fontweight='bold', color='white', pad=20)

    plt.tight_layout()
    plt.savefig(output_dir / 'chart_hero_results.png', dpi=150,
                facecolor=COLORS['bg'], edgecolor='none')
    plt.close()
    print("Created: chart_hero_results.png")

# =============================================================================
# CHART 2: Temperature Annealing
# =============================================================================
def create_temperature_chart():
    fig, ax = plt.subplots(figsize=(16, 9), facecolor=COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])

    steps = np.linspace(0, 3738, 100)
    progress = steps / 3738

    # Cosine decay
    temp_start, temp_end = 1.2, 0.5
    temp = temp_end + 0.5 * (temp_start - temp_end) * (1 + np.cos(np.pi * progress))

    # Create gradient effect
    for i in range(len(steps)-1):
        color_val = (temp[i] - temp_end) / (temp_start - temp_end)
        color = plt.cm.YlOrRd(color_val * 0.8 + 0.2)
        ax.plot(steps[i:i+2], temp[i:i+2], color=color, linewidth=6)

    # Annotations
    ax.annotate('HOT\nExploration', xy=(200, 1.15), fontsize=20,
                color=COLORS['accent'], fontweight='bold', ha='center')
    ax.annotate('COOL\nExploitation', xy=(3500, 0.55), fontsize=20,
                color=COLORS['secondary'], fontweight='bold', ha='center')

    ax.axhline(y=1.2, color=COLORS['accent'], linestyle='--', alpha=0.5, linewidth=2)
    ax.axhline(y=0.5, color=COLORS['secondary'], linestyle='--', alpha=0.5, linewidth=2)

    ax.set_xlabel('Training Steps', fontsize=18, color='white')
    ax.set_ylabel('Temperature', fontsize=18, color='white')
    ax.set_xlim(0, 3738)
    ax.set_ylim(0.3, 1.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.tick_params(colors='white', labelsize=14)

    plt.title('Temperature Annealing: Cosine Decay', fontsize=28,
              fontweight='bold', color='white', pad=20)

    plt.tight_layout()
    plt.savefig(output_dir / 'chart_temperature.png', dpi=150,
                facecolor=COLORS['bg'], edgecolor='none')
    plt.close()
    print("Created: chart_temperature.png")

# =============================================================================
# CHART 3: Reward Function Breakdown
# =============================================================================
def create_reward_chart():
    fig, ax = plt.subplots(figsize=(16, 9), facecolor=COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])

    rewards = ['Format\nExact', 'Format\nApprox', 'Reasoning\nCoherence',
               'Answer\nComplete', 'Question\nRelevance', 'Compression\nBonus']
    points = [3.0, 2.0, 3.0, 2.0, 2.0, 1.5]
    colors = [COLORS['primary'], COLORS['primary'],
              COLORS['secondary'], COLORS['secondary'],
              COLORS['accent'], COLORS['accent']]

    bars = ax.barh(rewards, points, color=colors, edgecolor='white', linewidth=2, height=0.7)

    # Add value labels
    for bar, val in zip(bars, points):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                f'+{val}', ha='left', va='center',
                fontsize=20, fontweight='bold', color='white')

    ax.set_xlabel('Max Points', fontsize=18, color='white')
    ax.set_xlim(0, 4.5)
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.tick_params(colors='white', labelsize=16)

    # Legend
    format_patch = mpatches.Patch(color=COLORS['primary'], label='Format (5 pts)')
    quality_patch = mpatches.Patch(color=COLORS['secondary'], label='Quality (5 pts)')
    meta_patch = mpatches.Patch(color=COLORS['accent'], label='Meta (3.5 pts)')
    ax.legend(handles=[format_patch, quality_patch, meta_patch],
              loc='lower right', fontsize=14, framealpha=0.8)

    plt.title('6 Reward Functions = 13.5 Max Points', fontsize=28,
              fontweight='bold', color='white', pad=20)

    plt.tight_layout()
    plt.savefig(output_dir / 'chart_rewards.png', dpi=150,
                facecolor=COLORS['bg'], edgecolor='none')
    plt.close()
    print("Created: chart_rewards.png")

# =============================================================================
# CHART 4: Dataset Distribution
# =============================================================================
def create_dataset_chart():
    fig, ax = plt.subplots(figsize=(12, 12), facecolor=COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])

    datasets = ['OpenAssistant', 'Dolly', 'Alpaca', 'FLAN']
    sizes = [25, 25, 30, 20]
    colors = ['#FF6B35', '#4ECDC4', '#FFE66D', '#95E1D3']
    explode = (0.02, 0.02, 0.02, 0.02)

    wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=datasets,
                                       colors=colors, autopct='%1.0f%%',
                                       shadow=True, startangle=90,
                                       textprops={'fontsize': 20, 'color': 'white'},
                                       wedgeprops={'edgecolor': 'white', 'linewidth': 2})

    for autotext in autotexts:
        autotext.set_fontsize(18)
        autotext.set_fontweight('bold')

    plt.title('Real User Task Datasets', fontsize=28,
              fontweight='bold', color='white', pad=20)

    plt.tight_layout()
    plt.savefig(output_dir / 'chart_datasets.png', dpi=150,
                facecolor=COLORS['bg'], edgecolor='none')
    plt.close()
    print("Created: chart_datasets.png")

# =============================================================================
# CHART 5: Training Progress Timeline
# =============================================================================
def create_timeline_chart():
    fig, ax = plt.subplots(figsize=(16, 6), facecolor=COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])

    # Timeline points
    events = ['Start', 'Warmup\nComplete', 'Midpoint', 'Cooldown', 'Complete']
    times = [0, 374, 1869, 2800, 3738]

    ax.plot(times, [1]*5, 'o-', color=COLORS['primary'], markersize=25, linewidth=4)

    for i, (event, time) in enumerate(zip(events, times)):
        ax.annotate(event, xy=(time, 1), xytext=(time, 1.15 if i % 2 == 0 else 0.85),
                    fontsize=16, color='white', ha='center', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='white', lw=2))
        ax.annotate(f'{time} steps', xy=(time, 1), xytext=(time, 0.7 if i % 2 == 0 else 1.3),
                    fontsize=12, color='#888888', ha='center')

    ax.set_xlim(-200, 4000)
    ax.set_ylim(0.5, 1.5)
    ax.axis('off')

    plt.title('Training Timeline: ~3 Hours on TPU v5e-8', fontsize=28,
              fontweight='bold', color='white', pad=20)

    plt.tight_layout()
    plt.savefig(output_dir / 'chart_timeline.png', dpi=150,
                facecolor=COLORS['bg'], edgecolor='none')
    plt.close()
    print("Created: chart_timeline.png")

# =============================================================================
# CHART 6: Architecture Diagram
# =============================================================================
def create_architecture_chart():
    fig, ax = plt.subplots(figsize=(16, 10), facecolor=COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])

    # Boxes
    boxes = [
        {'name': 'Gemma 3\n1B-IT', 'pos': (0.1, 0.5), 'color': '#4ECDC4'},
        {'name': 'LoRA\nR=32, α=32', 'pos': (0.3, 0.5), 'color': '#FFE66D'},
        {'name': 'GRPO\nTrainer', 'pos': (0.5, 0.5), 'color': '#FF6B35'},
        {'name': '6 Reward\nFunctions', 'pos': (0.7, 0.5), 'color': '#95E1D3'},
        {'name': 'Trained\nModel', 'pos': (0.9, 0.5), 'color': '#FF6B35'},
    ]

    for box in boxes:
        rect = mpatches.FancyBboxPatch(
            (box['pos'][0] - 0.08, box['pos'][1] - 0.12), 0.16, 0.24,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            facecolor=box['color'], edgecolor='white', linewidth=3
        )
        ax.add_patch(rect)
        ax.text(box['pos'][0], box['pos'][1], box['name'],
                ha='center', va='center', fontsize=16, fontweight='bold',
                color='black' if box['color'] in ['#FFE66D', '#95E1D3'] else 'white')

    # Arrows
    for i in range(len(boxes) - 1):
        ax.annotate('', xy=(boxes[i+1]['pos'][0] - 0.09, boxes[i+1]['pos'][1]),
                    xytext=(boxes[i]['pos'][0] + 0.09, boxes[i]['pos'][1]),
                    arrowprops=dict(arrowstyle='->', color='white', lw=3))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.title('PROMETHEUS Architecture', fontsize=32,
              fontweight='bold', color='white', pad=20)

    plt.tight_layout()
    plt.savefig(output_dir / 'chart_architecture.png', dpi=150,
                facecolor=COLORS['bg'], edgecolor='none')
    plt.close()
    print("Created: chart_architecture.png")

# =============================================================================
# Generate all charts
# =============================================================================
if __name__ == '__main__':
    print("\n=== Generating PROMETHEUS Video Assets ===\n")
    create_hero_results_chart()
    create_temperature_chart()
    create_reward_chart()
    create_dataset_chart()
    create_timeline_chart()
    create_architecture_chart()
    print("\n=== All charts generated! ===\n")
