#!/usr/bin/env python3
"""Generate 6 BADASS polished charts for PROMETHEUS YouTube video."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.collections import PatchCollection
import matplotlib.patheffects as path_effects
import numpy as np
from pathlib import Path

# Output directory
output_dir = Path("/home/user/aimo3/video_assets")
output_dir.mkdir(exist_ok=True)

# =============================================================================
# STYLE CONFIG - Consistent dark theme
# =============================================================================
plt.rcParams.update({
    'figure.facecolor': '#0a0a1a',
    'axes.facecolor': '#0a0a1a',
    'axes.edgecolor': '#ffffff',
    'axes.labelcolor': '#ffffff',
    'text.color': '#ffffff',
    'xtick.color': '#ffffff',
    'ytick.color': '#ffffff',
    'grid.color': '#1a1a3a',
    'font.family': 'sans-serif',
    'font.weight': 'bold',
})

# Color palette
ORANGE = '#FF6B35'
TEAL = '#4ECDC4'
YELLOW = '#FFE66D'
PURPLE = '#A855F7'
PINK = '#EC4899'
GREEN = '#10B981'
BG_DARK = '#0a0a1a'
BG_CARD = '#12122a'
WHITE = '#FFFFFF'
GRAY = '#6B7280'

def add_glow(ax, text_obj, color, alpha=0.3):
    """Add glow effect to text."""
    text_obj.set_path_effects([
        path_effects.withStroke(linewidth=8, foreground=color, alpha=alpha),
        path_effects.Normal()
    ])

# =============================================================================
# CHART 1: HERO COMPARISON - Before vs After (Dramatic)
# =============================================================================
def chart1_hero_comparison():
    fig, ax = plt.subplots(figsize=(19.2, 10.8), facecolor=BG_DARK)
    ax.set_facecolor(BG_DARK)

    # Data
    metrics = ['FORMAT\nCOMPLIANCE', 'REASONING\nTRACES', 'COHERENCE\nSCORE']
    before = [5.5, 11.0, 1.0]
    after = [89.0, 98.5, 1.23]

    x = np.arange(len(metrics))
    width = 0.35

    # Bars with gradient effect (simulated with multiple bars)
    bars1 = ax.bar(x - width/2, before, width, label='BEFORE',
                   color='#333344', edgecolor=GRAY, linewidth=2, zorder=3)
    bars2 = ax.bar(x + width/2, after, width, label='AFTER PROMETHEUS',
                   color=ORANGE, edgecolor=WHITE, linewidth=2, zorder=3)

    # Add percentage labels
    for bar, val in zip(bars1, before):
        height = bar.get_height()
        txt = ax.text(bar.get_x() + bar.get_width()/2, height + 3,
                f'{val:.1f}%' if val > 2 else f'{val:.2f}',
                ha='center', va='bottom', fontsize=28, fontweight='bold', color=GRAY)

    for bar, val in zip(bars2, after):
        height = bar.get_height()
        txt = ax.text(bar.get_x() + bar.get_width()/2, height + 3,
                f'{val:.1f}%' if val > 2 else f'{val:.2f}',
                ha='center', va='bottom', fontsize=32, fontweight='bold', color=ORANGE)
        add_glow(ax, txt, ORANGE, 0.5)

    # Improvement arrows
    for i, (b, a) in enumerate(zip(before, after)):
        improvement = a - b if a > 2 else (a - b) / b * 100
        arrow_text = f'+{improvement:.0f}%' if before[i] > 2 else f'+{improvement:.0f}%'
        ax.annotate('', xy=(x[i] + width/2, a - 5), xytext=(x[i] - width/2, b + 5),
                   arrowprops=dict(arrowstyle='->', color=TEAL, lw=3,
                                  connectionstyle='arc3,rad=0.3'))

    # Styling
    ax.set_ylabel('', fontsize=1)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=24, fontweight='bold')
    ax.set_ylim(0, 130)
    ax.set_xlim(-0.7, 2.7)

    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=False, bottom=False)
    ax.set_yticks([])

    # Legend
    legend = ax.legend(loc='upper left', fontsize=20, framealpha=0.9,
                       facecolor=BG_CARD, edgecolor=WHITE)

    # Title with glow
    title = ax.text(0.5, 1.05, 'PROMETHEUS RESULTS', transform=ax.transAxes,
                    fontsize=48, fontweight='bold', ha='center', color=WHITE)
    add_glow(ax, title, ORANGE, 0.4)

    subtitle = ax.text(0.5, 0.98, 'Single TPU Session • 3 Hours • Gemma 3 1B',
                       transform=ax.transAxes, fontsize=20, ha='center', color=GRAY)

    plt.tight_layout()
    plt.savefig(output_dir / 'yt_01_hero_comparison.png', dpi=100,
                facecolor=BG_DARK, edgecolor='none', bbox_inches='tight', pad_inches=0.5)
    plt.close()
    print("✓ Chart 1: Hero Comparison")

# =============================================================================
# CHART 2: ARCHITECTURE FLOWCHART
# =============================================================================
def chart2_architecture():
    fig, ax = plt.subplots(figsize=(19.2, 10.8), facecolor=BG_DARK)
    ax.set_facecolor(BG_DARK)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis('off')

    # Box positions and labels
    boxes = [
        {'x': 8, 'y': 30, 'w': 14, 'h': 18, 'label': 'GEMMA 3\n1B-IT', 'color': TEAL, 'sub': 'Base Model'},
        {'x': 28, 'y': 30, 'w': 14, 'h': 18, 'label': 'LoRA\nR=32', 'color': YELLOW, 'sub': 'Adapter'},
        {'x': 48, 'y': 30, 'w': 14, 'h': 18, 'label': 'GRPO\nTRAINER', 'color': ORANGE, 'sub': 'Tunix'},
        {'x': 68, 'y': 30, 'w': 14, 'h': 18, 'label': '6 REWARD\nFUNCTIONS', 'color': PURPLE, 'sub': 'Multi-Obj'},
        {'x': 88, 'y': 30, 'w': 14, 'h': 18, 'label': 'TRAINED\nMODEL', 'color': GREEN, 'sub': '89% Format'},
    ]

    # Draw boxes
    for box in boxes:
        # Shadow
        shadow = FancyBboxPatch((box['x']-7+0.5, box['y']-9-0.5), box['w'], box['h'],
                                boxstyle="round,pad=0.02,rounding_size=1",
                                facecolor='#000000', alpha=0.5)
        ax.add_patch(shadow)

        # Main box
        rect = FancyBboxPatch((box['x']-7, box['y']-9), box['w'], box['h'],
                              boxstyle="round,pad=0.02,rounding_size=1",
                              facecolor=BG_CARD, edgecolor=box['color'], linewidth=4)
        ax.add_patch(rect)

        # Label
        txt = ax.text(box['x'], box['y']+2, box['label'], ha='center', va='center',
                fontsize=18, fontweight='bold', color=box['color'])

        # Subtitle
        ax.text(box['x'], box['y']-6, box['sub'], ha='center', va='center',
                fontsize=12, color=GRAY)

    # Draw arrows
    arrow_style = dict(arrowstyle='->', color=WHITE, lw=3, mutation_scale=20)
    for i in range(len(boxes)-1):
        ax.annotate('', xy=(boxes[i+1]['x']-7, boxes[i+1]['y']),
                   xytext=(boxes[i]['x']+7, boxes[i]['y']),
                   arrowprops=arrow_style)

    # Data inputs (top)
    data_sources = ['OpenAssistant', 'Dolly', 'Alpaca', 'FLAN']
    for i, src in enumerate(data_sources):
        x_pos = 38 + i * 12
        ax.annotate('', xy=(48, 42), xytext=(x_pos, 52),
                   arrowprops=dict(arrowstyle='->', color=TEAL, lw=2, alpha=0.7))
        ax.text(x_pos, 55, src, ha='center', va='center', fontsize=12, color=TEAL)

    ax.text(55, 58, 'REAL USER DATA', ha='center', va='center', fontsize=16,
            fontweight='bold', color=TEAL)

    # Temperature (bottom)
    ax.text(48, 12, '🔥 T=1.2 → T=0.5 ❄️', ha='center', va='center', fontsize=16, color=YELLOW)
    ax.text(48, 7, 'TEMPERATURE ANNEALING', ha='center', va='center', fontsize=14, color=GRAY)

    # Title
    title = ax.text(50, 67, 'PROMETHEUS ARCHITECTURE', ha='center', va='center',
                    fontsize=42, fontweight='bold', color=WHITE)
    add_glow(ax, title, ORANGE, 0.3)

    ax.set_ylim(0, 72)

    plt.tight_layout()
    plt.savefig(output_dir / 'yt_02_architecture.png', dpi=100,
                facecolor=BG_DARK, edgecolor='none', bbox_inches='tight', pad_inches=0.3)
    plt.close()
    print("✓ Chart 2: Architecture")

# =============================================================================
# CHART 3: TEMPERATURE ANNEALING (Fire to Ice)
# =============================================================================
def chart3_temperature():
    fig, ax = plt.subplots(figsize=(19.2, 10.8), facecolor=BG_DARK)
    ax.set_facecolor(BG_DARK)

    # Generate temperature curve
    steps = np.linspace(0, 3738, 500)
    progress = steps / 3738
    temp_start, temp_end = 1.2, 0.5
    temp = temp_end + 0.5 * (temp_start - temp_end) * (1 + np.cos(np.pi * progress))

    # Create gradient line by plotting segments
    for i in range(len(steps)-1):
        color_val = (temp[i] - temp_end) / (temp_start - temp_end)
        # Interpolate from orange (hot) to teal (cold)
        r = int(255 * color_val + 78 * (1-color_val))
        g = int(107 * color_val + 205 * (1-color_val))
        b = int(53 * color_val + 196 * (1-color_val))
        color = f'#{r:02x}{g:02x}{b:02x}'
        ax.plot(steps[i:i+2], temp[i:i+2], color=color, linewidth=8, solid_capstyle='round')

    # Add glow effect (wider, transparent line behind)
    ax.plot(steps, temp, color=ORANGE, linewidth=20, alpha=0.2)

    # Annotations
    ax.annotate('🔥 EXPLORATION', xy=(300, 1.18), fontsize=28, color=ORANGE,
                fontweight='bold', ha='center')
    ax.annotate('High temperature = diverse reasoning strategies',
                xy=(300, 1.08), fontsize=14, color=GRAY, ha='center')

    ax.annotate('❄️ EXPLOITATION', xy=(3400, 0.55), fontsize=28, color=TEAL,
                fontweight='bold', ha='center')
    ax.annotate('Low temperature = refined, stable outputs',
                xy=(3400, 0.45), fontsize=14, color=GRAY, ha='center')

    # Key points
    ax.scatter([0, 3738], [1.2, 0.5], s=300, c=[ORANGE, TEAL], zorder=5, edgecolors=WHITE, linewidths=3)
    ax.text(0, 1.28, 'T=1.2', fontsize=18, color=ORANGE, ha='center', fontweight='bold')
    ax.text(3738, 0.42, 'T=0.5', fontsize=18, color=TEAL, ha='center', fontweight='bold')

    # Styling
    ax.set_xlabel('TRAINING STEPS', fontsize=20, fontweight='bold', labelpad=15)
    ax.set_ylabel('TEMPERATURE', fontsize=20, fontweight='bold', labelpad=15)
    ax.set_xlim(-100, 3850)
    ax.set_ylim(0.3, 1.4)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color(GRAY)
    ax.spines['left'].set_color(GRAY)
    ax.tick_params(labelsize=14)

    # Grid
    ax.grid(True, alpha=0.2, linestyle='--')

    # Title
    title = ax.text(0.5, 1.08, 'TEMPERATURE ANNEALING', transform=ax.transAxes,
                    fontsize=42, fontweight='bold', ha='center', color=WHITE)
    add_glow(ax, title, YELLOW, 0.3)

    subtitle = ax.text(0.5, 1.02, 'Cosine Decay: Explore First, Exploit Later',
                       transform=ax.transAxes, fontsize=18, ha='center', color=GRAY)

    plt.tight_layout()
    plt.savefig(output_dir / 'yt_03_temperature.png', dpi=100,
                facecolor=BG_DARK, edgecolor='none', bbox_inches='tight', pad_inches=0.5)
    plt.close()
    print("✓ Chart 3: Temperature")

# =============================================================================
# CHART 4: REWARD FUNCTIONS BREAKDOWN
# =============================================================================
def chart4_rewards():
    fig, ax = plt.subplots(figsize=(19.2, 10.8), facecolor=BG_DARK)
    ax.set_facecolor(BG_DARK)

    # Data
    rewards = [
        ('FORMAT EXACT', 3.0, ORANGE, 'Perfect tag structure'),
        ('FORMAT APPROX', 2.0, ORANGE, 'Partial tag credit'),
        ('COHERENCE', 3.0, TEAL, 'Logical connectors'),
        ('COMPLETENESS', 2.0, TEAL, 'Substantive answers'),
        ('RELEVANCE', 2.0, PURPLE, 'Stay on topic'),
        ('COMPRESSION', 1.5, PURPLE, 'Kolmogorov bonus'),
    ]

    y_pos = np.arange(len(rewards))
    values = [r[1] for r in rewards]
    colors = [r[2] for r in rewards]

    # Bars
    bars = ax.barh(y_pos, values, height=0.7, color=colors, edgecolor=WHITE, linewidth=2)

    # Add glow behind bars
    for bar, color in zip(bars, colors):
        ax.barh(bar.get_y() + bar.get_height()/2, bar.get_width(), height=0.9,
                color=color, alpha=0.2, zorder=1)

    # Labels
    for i, (name, val, color, desc) in enumerate(rewards):
        # Value on bar
        ax.text(val + 0.15, i, f'+{val}', va='center', ha='left',
                fontsize=24, fontweight='bold', color=color)
        # Description
        ax.text(val + 0.6, i, desc, va='center', ha='left',
                fontsize=14, color=GRAY)

    # Y-axis labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels([r[0] for r in rewards], fontsize=18, fontweight='bold')

    ax.set_xlim(0, 5)
    ax.set_ylim(-0.6, 5.6)
    ax.invert_yaxis()

    # Styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color(GRAY)
    ax.spines['left'].set_visible(False)
    ax.tick_params(left=False, labelsize=14)
    ax.set_xlabel('MAX POINTS', fontsize=16, fontweight='bold', color=GRAY)

    # Category labels on right
    ax.text(4.8, 0.5, 'FORMAT', fontsize=14, color=ORANGE, fontweight='bold',
            ha='right', rotation=0)
    ax.text(4.8, 2.5, 'QUALITY', fontsize=14, color=TEAL, fontweight='bold',
            ha='right', rotation=0)
    ax.text(4.8, 4.5, 'META', fontsize=14, color=PURPLE, fontweight='bold',
            ha='right', rotation=0)

    # Total
    total = sum(values)
    ax.text(0.98, 0.02, f'TOTAL: {total} PTS', transform=ax.transAxes,
            fontsize=24, fontweight='bold', color=GREEN, ha='right', va='bottom')

    # Title
    title = ax.text(0.5, 1.06, '6 REWARD FUNCTIONS', transform=ax.transAxes,
                    fontsize=42, fontweight='bold', ha='center', color=WHITE)
    add_glow(ax, title, PURPLE, 0.3)

    subtitle = ax.text(0.5, 1.0, 'Multi-Objective Training: Reward the Process',
                       transform=ax.transAxes, fontsize=18, ha='center', color=GRAY)

    plt.tight_layout()
    plt.savefig(output_dir / 'yt_04_rewards.png', dpi=100,
                facecolor=BG_DARK, edgecolor='none', bbox_inches='tight', pad_inches=0.5)
    plt.close()
    print("✓ Chart 4: Rewards")

# =============================================================================
# CHART 5: DATASET DISTRIBUTION
# =============================================================================
def chart5_datasets():
    fig, ax = plt.subplots(figsize=(19.2, 10.8), facecolor=BG_DARK)
    ax.set_facecolor(BG_DARK)

    # Data
    datasets = ['OpenAssistant', 'Dolly', 'Alpaca', 'FLAN']
    sizes = [25, 25, 30, 20]
    colors = [ORANGE, TEAL, YELLOW, PURPLE]
    descriptions = [
        'Human conversations',
        'Databricks instructions',
        'Stanford instructions',
        'Diverse tasks'
    ]

    # Donut chart
    wedges, texts, autotexts = ax.pie(sizes, labels=None, autopct='%1.0f%%',
                                       colors=colors, startangle=90,
                                       wedgeprops=dict(width=0.5, edgecolor=BG_DARK, linewidth=4),
                                       textprops={'fontsize': 20, 'fontweight': 'bold'},
                                       pctdistance=0.75)

    # Style percentage text
    for autotext, color in zip(autotexts, colors):
        autotext.set_color(WHITE)
        autotext.set_fontsize(22)
        autotext.set_fontweight('bold')

    # Center text
    ax.text(0, 0, 'REAL\nUSER\nDATA', ha='center', va='center',
            fontsize=28, fontweight='bold', color=WHITE)

    # Legend on right side
    legend_x = 1.3
    for i, (name, size, color, desc) in enumerate(zip(datasets, sizes, colors, descriptions)):
        y = 0.7 - i * 0.35
        # Color box
        ax.add_patch(FancyBboxPatch((legend_x - 0.15, y - 0.08), 0.1, 0.16,
                                    boxstyle="round,pad=0.02", facecolor=color,
                                    transform=ax.transAxes, clip_on=False))
        # Name
        ax.text(legend_x, y, name, transform=ax.transAxes, fontsize=20,
                fontweight='bold', color=WHITE, va='center')
        # Description
        ax.text(legend_x, y - 0.12, desc, transform=ax.transAxes, fontsize=14,
                color=GRAY, va='center')

    # Title
    title = ax.text(0.5, 1.08, 'TRAINING DATA', transform=ax.transAxes,
                    fontsize=42, fontweight='bold', ha='center', color=WHITE)
    add_glow(ax, title, TEAL, 0.3)

    subtitle = ax.text(0.5, 1.01, 'Not Math Benchmarks — Real Human Questions',
                       transform=ax.transAxes, fontsize=18, ha='center', color=GRAY)

    ax.set_xlim(-1.5, 2)

    plt.tight_layout()
    plt.savefig(output_dir / 'yt_05_datasets.png', dpi=100,
                facecolor=BG_DARK, edgecolor='none', bbox_inches='tight', pad_inches=0.5)
    plt.close()
    print("✓ Chart 5: Datasets")

# =============================================================================
# CHART 6: KEY TAKEAWAYS
# =============================================================================
def chart6_takeaways():
    fig, ax = plt.subplots(figsize=(19.2, 10.8), facecolor=BG_DARK)
    ax.set_facecolor(BG_DARK)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis('off')

    takeaways = [
        {'icon': '👥', 'title': 'REAL USER DATA', 'desc': 'Train on what humans actually ask,\nnot academic benchmarks', 'color': ORANGE},
        {'icon': '🌡️', 'title': 'TEMP ANNEALING', 'desc': 'Start hot for exploration,\nend cool for exploitation', 'color': YELLOW},
        {'icon': '⚖️', 'title': 'REWARD PROCESS', 'desc': "Can't verify correctness?\nReward reasoning structure", 'color': TEAL},
    ]

    card_width = 28
    start_x = 8
    gap = 4

    for i, item in enumerate(takeaways):
        x = start_x + i * (card_width + gap)

        # Card shadow
        shadow = FancyBboxPatch((x+0.5, 9.5), card_width, 36,
                                boxstyle="round,pad=0.02,rounding_size=2",
                                facecolor='#000000', alpha=0.5)
        ax.add_patch(shadow)

        # Card
        card = FancyBboxPatch((x, 10), card_width, 36,
                              boxstyle="round,pad=0.02,rounding_size=2",
                              facecolor=BG_CARD, edgecolor=item['color'], linewidth=4)
        ax.add_patch(card)

        # Icon
        ax.text(x + card_width/2, 38, item['icon'], ha='center', va='center',
                fontsize=48)

        # Number
        ax.text(x + 3, 42, str(i+1), ha='center', va='center',
                fontsize=24, fontweight='bold', color=item['color'])

        # Title
        ax.text(x + card_width/2, 28, item['title'], ha='center', va='center',
                fontsize=22, fontweight='bold', color=item['color'])

        # Description
        ax.text(x + card_width/2, 18, item['desc'], ha='center', va='center',
                fontsize=14, color=GRAY, linespacing=1.5)

    # Title
    title = ax.text(50, 55, 'KEY TAKEAWAYS', ha='center', va='center',
                    fontsize=48, fontweight='bold', color=WHITE)
    add_glow(ax, title, GREEN, 0.3)

    # Bottom text
    ax.text(50, 3, 'Fork it. Experiment. Push reasoning forward. 🚀',
            ha='center', va='center', fontsize=18, color=GRAY)

    plt.tight_layout()
    plt.savefig(output_dir / 'yt_06_takeaways.png', dpi=100,
                facecolor=BG_DARK, edgecolor='none', bbox_inches='tight', pad_inches=0.3)
    plt.close()
    print("✓ Chart 6: Takeaways")

# =============================================================================
# GENERATE ALL
# =============================================================================
if __name__ == '__main__':
    print("\n🔥 GENERATING BADASS CHARTS 🔥\n")
    chart1_hero_comparison()
    chart2_architecture()
    chart3_temperature()
    chart4_rewards()
    chart5_datasets()
    chart6_takeaways()
    print("\n✅ ALL 6 CHARTS GENERATED!\n")
    print(f"📁 Location: {output_dir}")
