#!/usr/bin/env python3
"""
Figure Generation Script for "The Mathematics of Intelligence"
Generates all figures for the book in PNG format at 300 DPI.

Usage:
    python generate_figures.py          # Generate all figures
    python generate_figures.py --part 1 # Generate only Part 1 figures
    python generate_figures.py --fig 01_01  # Generate specific figure

Requirements:
    pip install matplotlib numpy pillow scipy networkx
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle, FancyBboxPatch
from matplotlib.collections import LineCollection
import numpy as np
from pathlib import Path
import argparse
from typing import Optional, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configuration
DPI = 300
FIGURE_DIR = Path(__file__).parent
FIGURE_DIR.mkdir(exist_ok=True)

# Color schemes
COLORS = {
    'primary': '#2563eb',      # Blue
    'secondary': '#7c3aed',    # Purple
    'accent': '#f59e0b',       # Amber
    'success': '#10b981',      # Green
    'danger': '#ef4444',       # Red
    'dark': '#1f2937',         # Dark gray
    'light': '#f3f4f6',        # Light gray
    'bg': '#ffffff',           # White background
    'text': '#111827',         # Near black
}

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False


def save_figure(fig, name: str, tight: bool = True):
    """Save figure with consistent settings."""
    path = FIGURE_DIR / f"{name}.png"
    if tight:
        fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white', edgecolor='none')
    else:
        fig.savefig(path, dpi=DPI, facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"Generated: {path}")


# =============================================================================
# PART 0: BEGINNER'S GUIDE
# =============================================================================

def fig_00_01():
    """Traditional vs ML Programming"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Traditional Programming
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Traditional Programming', fontsize=14, fontweight='bold', pad=20)

    # Boxes
    boxes = [
        (2, 7, 'Rules'),
        (2, 4, 'Data'),
        (7, 5.5, 'Output'),
    ]
    for x, y, text in boxes:
        rect = FancyBboxPatch((x-0.8, y-0.5), 1.6, 1, boxstyle="round,pad=0.05",
                              facecolor=COLORS['primary'], edgecolor='none')
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=11, color='white', fontweight='bold')

    # Computer box
    rect = FancyBboxPatch((4, 4.5), 2, 2, boxstyle="round,pad=0.1",
                          facecolor=COLORS['light'], edgecolor=COLORS['dark'], linewidth=2)
    ax.add_patch(rect)
    ax.text(5, 5.5, 'Computer\n(follows rules)', ha='center', va='center', fontsize=10)

    # Arrows
    ax.annotate('', xy=(4, 7), xytext=(2.8, 7),
                arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=2))
    ax.annotate('', xy=(4, 4), xytext=(2.8, 4),
                arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=2))
    ax.annotate('', xy=(7-0.8, 5.5), xytext=(6, 5.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=2))

    # Machine Learning
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Machine Learning', fontsize=14, fontweight='bold', pad=20)

    # Boxes
    boxes = [
        (2, 7, 'Data'),
        (2, 4, 'Output\n(labels)'),
        (7, 5.5, 'Rules'),
    ]
    for x, y, text in boxes:
        color = COLORS['accent'] if 'Rules' in text else COLORS['primary']
        rect = FancyBboxPatch((x-0.8, y-0.5), 1.6, 1, boxstyle="round,pad=0.05",
                              facecolor=color, edgecolor='none')
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=11, color='white', fontweight='bold')

    # Computer box
    rect = FancyBboxPatch((4, 4.5), 2, 2, boxstyle="round,pad=0.1",
                          facecolor=COLORS['light'], edgecolor=COLORS['dark'], linewidth=2)
    ax.add_patch(rect)
    ax.text(5, 5.5, 'Computer\n(learns rules)', ha='center', va='center', fontsize=10)

    # Arrows
    ax.annotate('', xy=(4, 7), xytext=(2.8, 7),
                arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=2))
    ax.annotate('', xy=(4, 4), xytext=(2.8, 4),
                arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=2))
    ax.annotate('', xy=(7-0.8, 5.5), xytext=(6, 5.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['accent'], lw=2))

    plt.tight_layout()
    save_figure(fig, 'fig_00_01')


def fig_00_02():
    """Three Flavors of ML"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Supervised Learning
    ax = axes[0]
    np.random.seed(42)
    X1 = np.random.randn(20, 2) * 0.5 + [1, 1]
    X2 = np.random.randn(20, 2) * 0.5 + [3, 3]
    ax.scatter(X1[:, 0], X1[:, 1], c=COLORS['primary'], s=60, label='Class A')
    ax.scatter(X2[:, 0], X2[:, 1], c=COLORS['accent'], s=60, label='Class B')
    ax.plot([0, 4], [4, 0], 'k--', lw=2, label='Decision boundary')
    ax.set_title('Supervised Learning', fontsize=14, fontweight='bold')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, 4.5)

    # Unsupervised Learning
    ax = axes[1]
    np.random.seed(42)
    X1 = np.random.randn(15, 2) * 0.4 + [1, 1]
    X2 = np.random.randn(15, 2) * 0.4 + [3, 1]
    X3 = np.random.randn(15, 2) * 0.4 + [2, 3]
    ax.scatter(X1[:, 0], X1[:, 1], c=COLORS['primary'], s=60)
    ax.scatter(X2[:, 0], X2[:, 1], c=COLORS['accent'], s=60)
    ax.scatter(X3[:, 0], X3[:, 1], c=COLORS['success'], s=60)
    # Draw ellipses around clusters
    for X, color in [(X1, COLORS['primary']), (X2, COLORS['accent']), (X3, COLORS['success'])]:
        center = X.mean(axis=0)
        circle = plt.Circle(center, 0.8, fill=False, color=color, linestyle='--', linewidth=2)
        ax.add_patch(circle)
    ax.set_title('Unsupervised Learning', fontsize=14, fontweight='bold')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, 4.5)

    # Reinforcement Learning
    ax = axes[2]
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    # Grid world
    for i in range(6):
        ax.axhline(y=i, color='gray', linewidth=0.5)
        ax.axvline(x=i, color='gray', linewidth=0.5)
    # Agent
    ax.add_patch(plt.Circle((0.5, 0.5), 0.3, color=COLORS['primary']))
    ax.text(0.5, 0.5, 'A', ha='center', va='center', color='white', fontweight='bold')
    # Goal
    ax.add_patch(plt.Rectangle((4, 4), 1, 1, color=COLORS['success'], alpha=0.7))
    ax.text(4.5, 4.5, 'G', ha='center', va='center', fontweight='bold', fontsize=14)
    # Path with arrows
    path = [(0.5, 0.5), (1.5, 0.5), (2.5, 0.5), (2.5, 1.5), (2.5, 2.5), (3.5, 2.5), (3.5, 3.5), (4.5, 3.5), (4.5, 4.5)]
    for i in range(len(path)-1):
        ax.annotate('', xy=path[i+1], xytext=path[i],
                   arrowprops=dict(arrowstyle='->', color=COLORS['accent'], lw=2))
    ax.set_title('Reinforcement Learning', fontsize=14, fontweight='bold')
    ax.set_xlabel('Environment')
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    save_figure(fig, 'fig_00_02')


def fig_00_03():
    """Learning Loop Diagram"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Circular layout
    cx, cy = 5, 4
    radius = 2.5
    angles = [90, 0, -90, 180]  # Predict, Compare, Adjust, Input
    labels = ['1. PREDICT', '2. COMPARE', '3. ADJUST', '4. REPEAT']
    descriptions = [
        'Make a guess',
        'Check if correct',
        'Update weights',
        'Try again'
    ]

    for i, (angle, label, desc) in enumerate(zip(angles, labels, descriptions)):
        rad = np.radians(angle)
        x = cx + radius * np.cos(rad)
        y = cy + radius * np.sin(rad)

        # Box
        rect = FancyBboxPatch((x-0.9, y-0.5), 1.8, 1, boxstyle="round,pad=0.05",
                              facecolor=COLORS['primary'], edgecolor='none')
        ax.add_patch(rect)
        ax.text(x, y+0.1, label, ha='center', va='center', fontsize=10,
                color='white', fontweight='bold')

        # Description outside
        desc_x = cx + (radius + 1.2) * np.cos(rad)
        desc_y = cy + (radius + 1.2) * np.sin(rad)
        ax.text(desc_x, desc_y, desc, ha='center', va='center', fontsize=9,
                style='italic', color=COLORS['dark'])

    # Arrows between boxes
    for i in range(4):
        start_angle = angles[i] - 20
        end_angle = angles[(i+1) % 4] + 20
        start_rad = np.radians(start_angle)
        end_rad = np.radians(end_angle)

        start_x = cx + radius * np.cos(start_rad)
        start_y = cy + radius * np.sin(start_rad)
        end_x = cx + radius * np.cos(end_rad)
        end_y = cy + radius * np.sin(end_rad)

        ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                   arrowprops=dict(arrowstyle='->', color=COLORS['accent'],
                                   lw=2, connectionstyle='arc3,rad=0.2'))

    # Center text
    ax.text(cx, cy, 'LEARNING\nLOOP', ha='center', va='center', fontsize=12,
            fontweight='bold', color=COLORS['dark'])

    ax.set_title('The Machine Learning Feedback Loop', fontsize=14, fontweight='bold', y=0.95)
    save_figure(fig, 'fig_00_03')


# =============================================================================
# PART 0.5: NEURAL NETWORK FOUNDATIONS
# =============================================================================

def fig_05_01():
    """Biological vs Artificial Neuron"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Biological Neuron (simplified)
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('Biological Neuron', fontsize=14, fontweight='bold')

    # Cell body
    circle = plt.Circle((5, 4), 1.2, color=COLORS['primary'], alpha=0.8)
    ax.add_patch(circle)
    ax.text(5, 4, 'Cell\nBody', ha='center', va='center', color='white', fontsize=9, fontweight='bold')

    # Dendrites (inputs)
    for angle in [150, 180, 210]:
        rad = np.radians(angle)
        end_x = 5 + 2.5 * np.cos(rad)
        end_y = 4 + 2.5 * np.sin(rad)
        ax.plot([5 + 1.2*np.cos(rad), end_x], [4 + 1.2*np.sin(rad), end_y],
                color=COLORS['success'], lw=3)
        ax.text(end_x - 0.3, end_y, 'input', fontsize=8, ha='right')

    # Axon (output)
    ax.plot([6.2, 9], [4, 4], color=COLORS['accent'], lw=4)
    ax.annotate('', xy=(9.5, 4), xytext=(9, 4),
               arrowprops=dict(arrowstyle='->', color=COLORS['accent'], lw=3))
    ax.text(9.5, 4.3, 'output', fontsize=9)

    # Artificial Neuron
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('Artificial Neuron', fontsize=14, fontweight='bold')

    # Inputs
    input_y = [6, 4, 2]
    for i, y in enumerate(input_y):
        ax.text(0.5, y, f'x{i+1}', fontsize=11, ha='center', va='center')
        ax.text(1.5, y+0.3, f'w{i+1}', fontsize=9, color=COLORS['secondary'])
        ax.plot([0.8, 3.5], [y, 4], color=COLORS['dark'], lw=1)

    # Summation
    circle = plt.Circle((4.5, 4), 0.7, color=COLORS['light'], ec=COLORS['dark'], lw=2)
    ax.add_patch(circle)
    ax.text(4.5, 4, 'Σ', fontsize=18, ha='center', va='center')

    # Activation function
    ax.plot([5.2, 6], [4, 4], color=COLORS['dark'], lw=2)
    rect = FancyBboxPatch((6, 3.5), 1.5, 1, boxstyle="round,pad=0.05",
                          facecolor=COLORS['primary'], edgecolor='none')
    ax.add_patch(rect)
    ax.text(6.75, 4, 'f(·)', ha='center', va='center', color='white', fontsize=11)

    # Output
    ax.plot([7.5, 8.5], [4, 4], color=COLORS['dark'], lw=2)
    ax.annotate('', xy=(9.2, 4), xytext=(8.5, 4),
               arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=2))
    ax.text(9.5, 4, 'y', fontsize=11, ha='center', va='center')

    # Equation
    ax.text(5, 1, r'$y = f\left(\sum_{i} w_i x_i + b\right)$', fontsize=12, ha='center')

    plt.tight_layout()
    save_figure(fig, 'fig_05_01')


def fig_05_03():
    """XOR Problem Visualization"""
    fig, ax = plt.subplots(figsize=(6, 6))

    # XOR points
    points = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]
    for x, y, label in points:
        color = COLORS['primary'] if label == 1 else COLORS['accent']
        marker = 'o' if label == 1 else 's'
        ax.scatter(x, y, c=color, s=300, marker=marker, edgecolors='white', linewidths=2, zorder=5)
        ax.annotate(f'({x},{y})\nXOR={label}', (x, y),
                   textcoords='offset points', xytext=(0, 25 if y == 0 else -35),
                   ha='center', fontsize=10)

    # Try to draw separating lines (they fail)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Attempted boundaries')
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.plot([-.2, 1.2], [1.2, -.2], 'gray', linestyle='--', alpha=0.5)

    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(-0.3, 1.3)
    ax.set_xlabel('Input A', fontsize=12)
    ax.set_ylabel('Input B', fontsize=12)
    ax.set_title('XOR Problem: No Single Line Can Separate Classes', fontsize=12, fontweight='bold')
    ax.set_aspect('equal')

    # Legend
    ax.scatter([], [], c=COLORS['primary'], s=100, marker='o', label='XOR = 1')
    ax.scatter([], [], c=COLORS['accent'], s=100, marker='s', label='XOR = 0')
    ax.legend(loc='upper right')

    ax.grid(True, alpha=0.3)
    save_figure(fig, 'fig_05_03')


def fig_05_06():
    """Activation Functions"""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    x = np.linspace(-3, 3, 100)

    activations = [
        ('Sigmoid', lambda x: 1 / (1 + np.exp(-x)), r'$\sigma(x) = \frac{1}{1+e^{-x}}$'),
        ('Tanh', lambda x: np.tanh(x), r'$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$'),
        ('ReLU', lambda x: np.maximum(0, x), r'$\text{ReLU}(x) = \max(0, x)$'),
        ('GELU', lambda x: x * 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3))),
         r'$\text{GELU}(x) \approx x \cdot \Phi(x)$'),
    ]

    for ax, (name, func, formula) in zip(axes.flat, activations):
        y = func(x)
        ax.plot(x, y, color=COLORS['primary'], lw=3)
        ax.axhline(y=0, color='gray', lw=0.5)
        ax.axvline(x=0, color='gray', lw=0.5)
        ax.set_title(name, fontsize=14, fontweight='bold')
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.text(0.05, 0.95, formula, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.set_xlim(-3, 3)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, 'fig_05_06')


def fig_05_07():
    """Residual Connection"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Input
    ax.text(1, 4, 'Input\nx', ha='center', va='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor=COLORS['light'], edgecolor=COLORS['dark']))

    # Layer block
    rect = FancyBboxPatch((3, 3), 3, 2, boxstyle="round,pad=0.1",
                          facecolor=COLORS['primary'], edgecolor='none')
    ax.add_patch(rect)
    ax.text(4.5, 4, 'Layer\nF(x)', ha='center', va='center', color='white', fontsize=11, fontweight='bold')

    # Main path arrow
    ax.annotate('', xy=(3, 4), xytext=(1.7, 4),
               arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=2))
    ax.annotate('', xy=(7.5, 4), xytext=(6, 4),
               arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=2))

    # Skip connection (curved)
    ax.annotate('', xy=(7.5, 4.5), xytext=(1.7, 4.5),
               arrowprops=dict(arrowstyle='->', color=COLORS['accent'], lw=2,
                             connectionstyle='arc3,rad=-0.4'))
    ax.text(4.5, 6.5, 'Skip Connection (x)', ha='center', va='center',
            fontsize=10, color=COLORS['accent'], fontweight='bold')

    # Addition
    circle = plt.Circle((8, 4), 0.5, color=COLORS['light'], ec=COLORS['dark'], lw=2)
    ax.add_patch(circle)
    ax.text(8, 4, '+', fontsize=20, ha='center', va='center')

    # Output
    ax.annotate('', xy=(10, 4), xytext=(8.5, 4),
               arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=2))
    ax.text(10.5, 4, 'Output\nx + F(x)', ha='center', va='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor=COLORS['success'], edgecolor='none', alpha=0.7))

    # Equation
    ax.text(6, 1.5, 'Output = x + F(x)', fontsize=14, ha='center', fontweight='bold')
    ax.text(6, 0.8, 'Gradients can flow through skip connection', fontsize=10, ha='center', style='italic')

    ax.set_title('Residual (Skip) Connection', fontsize=14, fontweight='bold', y=0.95)
    save_figure(fig, 'fig_05_07')


def fig_05_11():
    """Word Embedding Space"""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Example word positions (2D projection)
    words = {
        'king': (2, 4),
        'queen': (4, 4),
        'man': (2, 2),
        'woman': (4, 2),
        'prince': (1.5, 3.5),
        'princess': (4.5, 3.5),
        'boy': (1.5, 1.5),
        'girl': (4.5, 1.5),
    }

    # Plot words
    for word, (x, y) in words.items():
        ax.scatter(x, y, s=100, c=COLORS['primary'], zorder=5)
        ax.annotate(word, (x, y), textcoords='offset points', xytext=(5, 5),
                   fontsize=11, fontweight='bold')

    # Draw arrows showing relationships
    # King - Man + Woman = Queen
    ax.annotate('', xy=words['queen'], xytext=words['king'],
               arrowprops=dict(arrowstyle='->', color=COLORS['accent'], lw=2, ls='--'))
    ax.annotate('', xy=words['woman'], xytext=words['man'],
               arrowprops=dict(arrowstyle='->', color=COLORS['accent'], lw=2, ls='--'))

    # Same relationship in another pair
    ax.annotate('', xy=words['princess'], xytext=words['prince'],
               arrowprops=dict(arrowstyle='->', color=COLORS['secondary'], lw=2, ls='--'))
    ax.annotate('', xy=words['girl'], xytext=words['boy'],
               arrowprops=dict(arrowstyle='->', color=COLORS['secondary'], lw=2, ls='--'))

    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(1, 5)
    ax.set_xlabel('Dimension 1', fontsize=12)
    ax.set_ylabel('Dimension 2', fontsize=12)
    ax.set_title('Word Embedding Space: Capturing Relationships', fontsize=14, fontweight='bold')

    # Add equation
    ax.text(3, 4.7, 'king − man + woman ≈ queen', fontsize=12, ha='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    save_figure(fig, 'fig_05_11')


# =============================================================================
# ATTENTION AND TRANSFORMER FIGURES
# =============================================================================

def fig_07_01():
    """Attention Mechanism"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.set_title('Self-Attention Mechanism', fontsize=16, fontweight='bold', y=0.95)

    # Input sequence
    words = ['The', 'cat', 'sat', 'on', 'mat']
    for i, word in enumerate(words):
        x = 1 + i * 2.4
        rect = FancyBboxPatch((x-0.5, 8), 1, 0.8, boxstyle="round,pad=0.05",
                              facecolor=COLORS['light'], edgecolor=COLORS['dark'])
        ax.add_patch(rect)
        ax.text(x, 8.4, word, ha='center', va='center', fontsize=11, fontweight='bold')

    ax.text(0.2, 8.4, 'Input:', fontsize=10, ha='right', fontweight='bold')

    # Q, K, V projections
    for i, (label, color) in enumerate([('Q', COLORS['primary']), ('K', COLORS['accent']), ('V', COLORS['success'])]):
        y = 6 - i * 1.5
        ax.text(0.5, y, label, fontsize=14, ha='center', va='center', fontweight='bold', color=color)
        for j in range(5):
            x = 1 + j * 2.4
            rect = FancyBboxPatch((x-0.4, y-0.3), 0.8, 0.6, boxstyle="round,pad=0.02",
                                  facecolor=color, edgecolor='none', alpha=0.7)
            ax.add_patch(rect)

    # Attention computation box
    rect = FancyBboxPatch((0.5, 1.5), 5, 1.5, boxstyle="round,pad=0.1",
                          facecolor=COLORS['light'], edgecolor=COLORS['dark'], lw=2)
    ax.add_patch(rect)
    ax.text(3, 2.25, r'Attention = softmax$\left(\frac{QK^T}{\sqrt{d_k}}\right)V$',
            fontsize=12, ha='center', va='center')

    # Output
    ax.text(8, 2.25, 'Output:', fontsize=10, ha='right', fontweight='bold')
    for j in range(5):
        x = 8.5 + j * 1
        rect = FancyBboxPatch((x-0.3, 2), 0.6, 0.5, boxstyle="round,pad=0.02",
                              facecolor=COLORS['secondary'], edgecolor='none')
        ax.add_patch(rect)

    # Arrows
    ax.annotate('', xy=(3, 3), xytext=(3, 5.3),
               arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=2))
    ax.annotate('', xy=(8, 2.25), xytext=(5.5, 2.25),
               arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=2))

    save_figure(fig, 'fig_07_01')


def fig_07_02():
    """Attention Weights Heatmap"""
    fig, ax = plt.subplots(figsize=(8, 7))

    # Example attention weights
    words = ['The', 'cat', 'sat', 'on', 'the', 'mat']
    n = len(words)

    # Create synthetic attention pattern
    np.random.seed(42)
    attention = np.random.rand(n, n)
    # Make it more interesting - cat attends to mat
    attention[1, 5] = 0.9  # cat → mat
    attention[5, 1] = 0.8  # mat → cat
    attention[2, 3] = 0.7  # sat → on
    # Normalize rows
    attention = attention / attention.sum(axis=1, keepdims=True)

    im = ax.imshow(attention, cmap='Blues', aspect='auto')
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(words, fontsize=11)
    ax.set_yticklabels(words, fontsize=11)
    ax.set_xlabel('Key Position', fontsize=12)
    ax.set_ylabel('Query Position', fontsize=12)
    ax.set_title('Attention Weights: Which Words Attend to Which', fontsize=14, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attention Weight', fontsize=10)

    # Add values
    for i in range(n):
        for j in range(n):
            text = ax.text(j, i, f'{attention[i, j]:.2f}',
                          ha='center', va='center', fontsize=9,
                          color='white' if attention[i, j] > 0.5 else 'black')

    save_figure(fig, 'fig_07_02')


def fig_10_01():
    """CIC Functional Diagram"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Main equation
    ax.text(7, 9, r'$F[T] = \Phi(T) - \lambda H(T|X) + \gamma C_{multi}(T)$',
            fontsize=20, ha='center', fontweight='bold')

    # Three components
    components = [
        (2.5, 5.5, 'Φ(T)', 'Information\nCohesion', COLORS['primary'],
         'Measures how well\nsamples compress together'),
        (7, 5.5, 'H(T|X)', 'Representation\nEntropy', COLORS['accent'],
         'Penalizes high\nuncertainty in encoding'),
        (11.5, 5.5, 'Cmulti(T)', 'Multi-scale\nCoherence', COLORS['success'],
         'Rewards consistency\nacross scales'),
    ]

    for x, y, symbol, name, color, desc in components:
        # Box
        rect = FancyBboxPatch((x-1.5, y-1.5), 3, 3, boxstyle="round,pad=0.1",
                              facecolor=color, edgecolor='none', alpha=0.8)
        ax.add_patch(rect)
        ax.text(x, y+0.5, symbol, ha='center', va='center', fontsize=16,
                color='white', fontweight='bold')
        ax.text(x, y-0.5, name, ha='center', va='center', fontsize=10,
                color='white')

        # Description below
        ax.text(x, y-2.5, desc, ha='center', va='top', fontsize=9, style='italic')

    # Arrows connecting
    ax.annotate('', xy=(4.5, 5.5), xytext=(4, 5.5),
               arrowprops=dict(arrowstyle='-', color=COLORS['dark'], lw=2))
    ax.text(4.25, 5.9, '−λ', fontsize=12, ha='center')

    ax.annotate('', xy=(10, 5.5), xytext=(9.5, 5.5),
               arrowprops=dict(arrowstyle='-', color=COLORS['dark'], lw=2))
    ax.text(9.75, 5.9, '+γ', fontsize=12, ha='center')

    # Output
    ax.text(7, 1.5, 'High F[T] → High Confidence, Reliable Output',
            fontsize=12, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.set_title('The CIC Functional: Three Components of Intelligent Inference',
                fontsize=14, fontweight='bold', y=0.98)
    save_figure(fig, 'fig_10_01')


def fig_21_01():
    """Knowledge Quadrants"""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Draw quadrants
    ax.axhline(y=5, color=COLORS['dark'], lw=2)
    ax.axvline(x=5, color=COLORS['dark'], lw=2)

    # Quadrant labels and content
    quadrants = [
        (2.5, 7.5, 'KNOWN\nKNOWNS', COLORS['success'],
         'Things we know we know\n\n• Verified data\n• Validated models\n• Ground truth'),
        (7.5, 7.5, 'KNOWN\nUNKNOWNS', COLORS['primary'],
         'Things we know we don\'t know\n\n• Identified uncertainties\n• Quantified risks\n• Open questions'),
        (2.5, 2.5, 'UNKNOWN\nKNOWNS', COLORS['accent'],
         'Things we don\'t know we know\n\n• Implicit knowledge\n• Hidden assumptions\n• Untapped expertise'),
        (7.5, 2.5, 'UNKNOWN\nUNKNOWNS', COLORS['danger'],
         'Things we don\'t know we don\'t know\n\n• Black swans\n• Model blind spots\n• Unseen failure modes'),
    ]

    for x, y, title, color, content in quadrants:
        # Background
        rect = FancyBboxPatch((x-2.3, y-2.3), 4.6, 4.6, boxstyle="round,pad=0.1",
                              facecolor=color, edgecolor='none', alpha=0.2)
        ax.add_patch(rect)

        # Title
        ax.text(x, y+1.5, title, ha='center', va='center', fontsize=12,
                fontweight='bold', color=color)

        # Content
        ax.text(x, y-0.5, content, ha='center', va='top', fontsize=9)

    # Axis labels
    ax.text(5, 10.3, 'WE KNOW', ha='center', fontsize=11, fontweight='bold')
    ax.text(5, -0.3, 'WE DON\'T KNOW', ha='center', fontsize=11, fontweight='bold')
    ax.text(-0.3, 5, 'IT\'S KNOWN', ha='center', va='center', fontsize=11,
            fontweight='bold', rotation=90)
    ax.text(10.3, 5, 'IT\'S UNKNOWN', ha='center', va='center', fontsize=11,
            fontweight='bold', rotation=90)

    ax.set_title('The Knowledge Quadrants', fontsize=16, fontweight='bold', y=1.02)
    save_figure(fig, 'fig_21_01')


def fig_24_01():
    """Self-Hosted Architecture"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.set_title('Serverless LLM Inference Architecture', fontsize=16, fontweight='bold', y=0.98)

    # User
    ax.text(1, 8, '👤 User', fontsize=12, ha='center')

    # Next.js App Box
    rect = FancyBboxPatch((2.5, 6), 4, 3, boxstyle="round,pad=0.1",
                          facecolor=COLORS['light'], edgecolor=COLORS['dark'], lw=2)
    ax.add_patch(rect)
    ax.text(4.5, 8.5, 'Next.js App', fontsize=11, ha='center', fontweight='bold')

    # Inside Next.js
    for i, (label, y) in enumerate([('React UI', 7.8), ('API Routes', 7.2), ('Edge Runtime', 6.6)]):
        small_rect = FancyBboxPatch((2.8, y-0.2), 1.5, 0.4, boxstyle="round,pad=0.02",
                                    facecolor=COLORS['primary'], edgecolor='none', alpha=0.7)
        ax.add_patch(small_rect)
        ax.text(3.55, y, label, ha='center', va='center', fontsize=8, color='white')

    # Arrows from User
    ax.annotate('', xy=(2.5, 7.5), xytext=(1.5, 8),
               arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=2))

    # Serverless Function
    rect = FancyBboxPatch((8, 7), 2.5, 2, boxstyle="round,pad=0.1",
                          facecolor=COLORS['accent'], edgecolor='none')
    ax.add_patch(rect)
    ax.text(9.25, 8, 'Serverless\nFunction', ha='center', va='center', fontsize=10,
            color='white', fontweight='bold')

    # WASM Module
    rect = FancyBboxPatch((8, 4), 2.5, 2, boxstyle="round,pad=0.1",
                          facecolor=COLORS['success'], edgecolor='none')
    ax.add_patch(rect)
    ax.text(9.25, 5, 'WASM\nModule', ha='center', va='center', fontsize=10,
            color='white', fontweight='bold')

    # Rust Engine
    rect = FancyBboxPatch((8, 1), 2.5, 2, boxstyle="round,pad=0.1",
                          facecolor=COLORS['secondary'], edgecolor='none')
    ax.add_patch(rect)
    ax.text(9.25, 2, 'Rust Inference\nEngine', ha='center', va='center', fontsize=10,
            color='white', fontweight='bold')

    # Model
    rect = FancyBboxPatch((12, 3), 1.5, 4, boxstyle="round,pad=0.1",
                          facecolor=COLORS['danger'], edgecolor='none', alpha=0.7)
    ax.add_patch(rect)
    ax.text(12.75, 5, 'LLM\n\n(GGUF\nQ4_K_M)', ha='center', va='center', fontsize=9,
            color='white', fontweight='bold')

    # Arrows
    ax.annotate('', xy=(8, 8), xytext=(6.5, 7.5),
               arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=2))
    ax.annotate('', xy=(8, 5), xytext=(6.5, 6.5),
               arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=2))
    ax.annotate('', xy=(9.25, 4), xytext=(9.25, 7),
               arrowprops=dict(arrowstyle='<->', color=COLORS['dark'], lw=1.5))
    ax.annotate('', xy=(9.25, 1), xytext=(9.25, 4),
               arrowprops=dict(arrowstyle='<->', color=COLORS['dark'], lw=1.5))
    ax.annotate('', xy=(12, 5), xytext=(10.5, 2),
               arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=2))

    # Labels
    ax.text(7.25, 8.2, 'API call', fontsize=9, ha='center', style='italic')
    ax.text(7.25, 5.8, 'Browser/Edge', fontsize=9, ha='center', style='italic')
    ax.text(11.25, 3.8, 'Load\nModel', fontsize=8, ha='center', style='italic')

    # Legend
    legend_items = [
        (COLORS['primary'], 'Frontend'),
        (COLORS['accent'], 'Serverless'),
        (COLORS['success'], 'WebAssembly'),
        (COLORS['secondary'], 'Native Rust'),
        (COLORS['danger'], 'Model'),
    ]
    for i, (color, label) in enumerate(legend_items):
        y = 3 - i * 0.5
        rect = FancyBboxPatch((0.5, y), 0.4, 0.3, boxstyle="round,pad=0.01",
                              facecolor=color, edgecolor='none', alpha=0.7)
        ax.add_patch(rect)
        ax.text(1.1, y+0.15, label, fontsize=9, va='center')

    save_figure(fig, 'fig_24_01')


# =============================================================================
# COVER DESIGNS
# =============================================================================

def cover_design_01():
    """Neural Cosmos Cover"""
    fig, ax = plt.subplots(figsize=(10, 16))  # 1.6:1 ratio
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 16)
    ax.axis('off')

    # Deep space background
    ax.set_facecolor('#0a0a1a')
    fig.patch.set_facecolor('#0a0a1a')

    # Neural network as stars
    np.random.seed(42)
    n_nodes = 50
    nodes_x = np.random.uniform(1, 9, n_nodes)
    nodes_y = np.random.uniform(2, 14, n_nodes)
    sizes = np.random.uniform(20, 200, n_nodes)

    # Draw connections first
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            if np.random.random() < 0.1:  # 10% chance of connection
                ax.plot([nodes_x[i], nodes_x[j]], [nodes_y[i], nodes_y[j]],
                       color='cyan', alpha=0.1, lw=0.5)

    # Draw nodes
    ax.scatter(nodes_x, nodes_y, s=sizes, c='white', alpha=0.8, edgecolors='none')
    ax.scatter(nodes_x, nodes_y, s=sizes*0.3, c='cyan', alpha=0.5, edgecolors='none')

    # Title
    ax.text(5, 12, 'THE MATHEMATICS OF', fontsize=24, ha='center', va='center',
            color='white', fontweight='bold', family='sans-serif')
    ax.text(5, 10.5, 'INTELLIGENCE', fontsize=36, ha='center', va='center',
            color='white', fontweight='bold', family='sans-serif')

    # Subtitle
    ax.text(5, 8.5, 'FROM TRANSFORMERS TO TRUTH', fontsize=14, ha='center', va='center',
            color='white', alpha=0.8, family='sans-serif')

    # Tagline
    ax.text(5, 3, 'The Complete Guide to Understanding,\nBuilding, and Aligning\nLarge Language Models',
            fontsize=11, ha='center', va='center', color='white', alpha=0.6, family='sans-serif')

    # Author (placeholder)
    ax.text(5, 1.2, '[Author Name]', fontsize=12, ha='center', va='center',
            color='white', alpha=0.8, family='sans-serif')

    save_figure(fig, 'cover_design_01', tight=False)


def cover_design_04():
    """Emergence Cover"""
    fig, ax = plt.subplots(figsize=(10, 16))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 16)
    ax.axis('off')

    # Black background
    ax.set_facecolor('#0a0a0a')
    fig.patch.set_facecolor('#0a0a0a')

    # Particle transition from chaos to order
    np.random.seed(42)

    # Left side: chaos
    n_chaos = 100
    chaos_x = np.random.uniform(0.5, 3, n_chaos)
    chaos_y = np.random.uniform(3, 13, n_chaos)
    ax.scatter(chaos_x, chaos_y, s=10, c='white', alpha=0.3)

    # Middle: transition
    n_trans = 50
    trans_x = np.random.uniform(4, 6, n_trans)
    trans_y = np.random.uniform(3, 13, n_trans)
    # Slightly more organized
    for i in range(n_trans):
        trans_y[i] = trans_y[i] + 0.5 * np.sin(trans_x[i] * 2)
    ax.scatter(trans_x, trans_y, s=15, c='cyan', alpha=0.5)

    # Right side: order (network structure)
    n_order = 20
    order_x = np.linspace(7, 9, 4)
    order_y = np.linspace(4, 12, 5)

    for x in order_x:
        for y in order_y:
            ax.scatter(x, y, s=50, c='gold', alpha=0.9)
            # Connect to neighbors
            if x < 9:
                ax.plot([x, x+0.67], [y, y], color='gold', alpha=0.3, lw=1)
            if y < 12:
                ax.plot([x, x], [y, y+2], color='gold', alpha=0.3, lw=1)

    # Title
    ax.text(5, 14.5, 'THE MATHEMATICS', fontsize=28, ha='center', va='center',
            color='white', fontweight='bold', family='sans-serif')
    ax.text(5, 13.2, 'OF INTELLIGENCE', fontsize=28, ha='center', va='center',
            color='white', fontweight='bold', family='sans-serif')

    # Subtitle
    ax.text(5, 1.5, 'FROM PARAMETERS TO UNDERSTANDING', fontsize=11, ha='center',
            color='white', alpha=0.7, family='sans-serif')
    ax.text(5, 0.8, 'A Complete Guide to Large Language Models', fontsize=10, ha='center',
            color='white', alpha=0.5, family='sans-serif')

    save_figure(fig, 'cover_design_04', tight=False)


# =============================================================================
# MAIN
# =============================================================================

def generate_all():
    """Generate all figures."""
    print("Generating all figures...")

    # Part 0
    fig_00_01()
    fig_00_02()
    fig_00_03()

    # Part 0.5
    fig_05_01()
    fig_05_03()
    fig_05_06()
    fig_05_07()
    fig_05_11()

    # Part II
    fig_07_01()
    fig_07_02()

    # Part III
    fig_10_01()

    # Part IV
    fig_21_01()

    # Part V
    fig_24_01()

    # Covers
    cover_design_01()
    cover_design_04()

    print("\nDone! Generated figures in:", FIGURE_DIR)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate book figures')
    parser.add_argument('--all', action='store_true', help='Generate all figures')
    parser.add_argument('--fig', type=str, help='Generate specific figure (e.g., 05_03)')
    args = parser.parse_args()

    if args.fig:
        func_name = f'fig_{args.fig}'
        if func_name in globals():
            globals()[func_name]()
        else:
            print(f"Figure function {func_name} not found")
    else:
        generate_all()
