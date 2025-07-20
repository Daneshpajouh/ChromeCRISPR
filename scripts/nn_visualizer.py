#!/usr/bin/env python3
"""
Professional Neural Network Architecture Visualizer
• clean_blocks      – flat horizontal blocks (Medium-style)
• 3d_blocks         – isometric depth blocks (PlotNeuralNet-style)
• node_network      – classic node/edge view
Output: PNG, SVG, PDF at 300 DPI, GitHub-README ready.
"""
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, ConnectionPatch
import argparse
import numpy as np
from typing import List, Dict, Tuple

# ---------- colour palettes --------------------------------------------------
PALETTES = {
    "blue":  dict(input="#E3F2FD", conv="#2196F3", pool="#64B5F6",
                  dense="#1976D2", output="#0D47A1", dropout="#90CAF9",
                  batch_norm="#42A5F5", lstm="#4FC3F7", gru="#29B6F6",
                  attention="#81D4FA", gc="#B3E5FC"),
    "green": dict(input="#E8F5E9", conv="#4CAF50", pool="#81C784",
                  dense="#388E3C", output="#1B5E20", dropout="#A5D6A7",
                  batch_norm="#66BB6A", lstm="#66BB6A", gru="#4CAF50",
                  attention="#81C784", gc="#A5D6A7"),
    "purple":dict(input="#F3E5F5", conv="#9C27B0", pool="#BA68C8",
                  dense="#7B1FA2", output="#4A148C", dropout="#CE93D8",
                  batch_norm="#AB47BC", lstm="#BA68C8", gru="#9C27B0",
                  attention="#CE93D8", gc="#E1BEE7"),
}

# ---------- parser for "Conv2D(64)->MaxPool(2x2)->Dense(128)->Dense(10)" ----
def parse(arch: str) -> List[Dict]:
    spec, out = arch.split("->"), []
    for i, token in enumerate(spec):
        t = token.strip()
        if "Conv" in t:
            f = int(t[t.find("(")+1:t.find(")")].split(",")[0])
            out += [dict(type="conv", name="Conv2D", txt=f"{f} ch")]
        elif "Pool" in t:
            k = t[t.find("(")+1:t.find(")") or None] or "2x2"
            out += [dict(type="pool", name="Pool", txt=k)]
        elif "Dense" in t or "FC" in t:
            u = int(t[t.find("(")+1:t.find(")")])
            out += [dict(type="output" if i==len(spec)-1 else "dense",
                         name="Dense", txt=f"{u}")]
        elif "Dropout" in t:
            r = t[t.find("(")+1:t.find(")")] or "0.5"
            out += [dict(type="dropout", name="Dropout", txt=r)]
        elif "Batch" in t or "BN" in t:
            out += [dict(type="batch_norm", name="BatchNorm", txt="")]
        elif "LSTM" in t:
            u = int(t[t.find("(")+1:t.find(")")]) if "(" in t else 128
            out += [dict(type="lstm", name="LSTM", txt=f"{u}")]
        elif "GRU" in t:
            u = int(t[t.find("(")+1:t.find(")")]) if "(" in t else 128
            out += [dict(type="gru", name="GRU", txt=f"{u}")]
        elif "Attention" in t or "GC" in t:
            out += [dict(type="attention", name="Attention", txt="")]
    out.insert(0, dict(type="input",  name="Input",  txt=""))
    return out

# ---------- flat Medium-style blocks -----------------------------------------
def clean_blocks(layers: List[Dict], palette: str, out: str, size: Tuple[int,int]):
    w, h, gap = 2.5, 3, 1.5
    fig, ax = plt.subplots(figsize=size, dpi=300)
    colours = PALETTES[palette]
    
    # Professional styling
    plt.style.use('default')
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['axes.edgecolor'] = '#333333'
    
    for i, L in enumerate(layers):
        x = i*(w+gap)
        rect = FancyBboxPatch((x,0), w,h, boxstyle="round,pad=0.1",
                               facecolor=colours.get(L["type"], colours["conv"]),
                               edgecolor='#1A237E', linewidth=2, alpha=0.9)
        ax.add_patch(rect)
        
        # Text color: black for input, white for others
        text_color = 'black' if L["type"] == "input" else 'white'
        
        # Main label (layer name)
        ax.text(x+w/2, h*0.60, L["name"], ha="center", va="center",
                color=text_color, weight="bold", size=11)
        
        # Sub label (parameters/dimensions)
        ax.text(x+w/2, h*0.35, L["txt"], ha="center", va="center",
                color=text_color, size=9, alpha=0.8)
        
        # Draw arrows between layers
        if i > 0:
            start_x = x - gap + 0.1
            end_x = x - 0.1
            start_y = h/2
            end_y = h/2
            
            arrow = ConnectionPatch((start_x, start_y), (end_x, end_y), "data", "data",
                                   arrowstyle="->", shrinkA=5, shrinkB=5,
                                   mutation_scale=20, fc='#263238', ec='#263238', linewidth=2)
            ax.add_patch(arrow)
    
    # Title and styling
    plt.title('Neural Network Architecture', 
             fontsize=18, fontweight='bold', color='#1A237E', pad=25)
    
    ax.set_xlim(-1, len(layers)*(w+gap) - gap + 1)
    ax.set_ylim(-0.5, h + 0.5)
    ax.axis("off")
    ax.set_aspect("equal")
    fig.patch.set_facecolor('white')
    
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor='white', edgecolor='none')
    plt.close()

# ---------- 3D isometric blocks ----------------------------------------------
def blocks_3d(layers: List[Dict], palette: str, out: str, size: Tuple[int,int]):
    w, h, gap = 2.5, 3, 1.5
    fig, ax = plt.subplots(figsize=size, dpi=300)
    colours = PALETTES[palette]
    
    for i, L in enumerate(layers):
        x = i*(w+gap)
        y = 0
        
        # Create 3D effect with multiple rectangles
        for depth in range(3):
            alpha = 0.9 - depth * 0.2
            offset = depth * 0.1
            
            rect = FancyBboxPatch((x + offset, y + offset), w, h, 
                                 boxstyle="round,pad=0.1",
                                 facecolor=colours.get(L["type"], colours["conv"]),
                                 edgecolor='#1A237E', linewidth=1.5, alpha=alpha)
            ax.add_patch(rect)
        
        # Text on top layer
        text_color = 'black' if L["type"] == "input" else 'white'
        ax.text(x+w/2, h/2, L["name"], ha="center", va="center",
                color=text_color, weight="bold", size=11)
        ax.text(x+w/2, h/2 - 0.5, L["txt"], ha="center", va="center",
                color=text_color, size=9, alpha=0.8)
        
        # 3D arrows
        if i > 0:
            start_x = x - gap + 0.1
            end_x = x - 0.1
            start_y = h/2
            end_y = h/2
            
            arrow = ConnectionPatch((start_x, start_y), (end_x, end_y), "data", "data",
                                   arrowstyle="->", shrinkA=5, shrinkB=5,
                                   mutation_scale=20, fc='#263238', ec='#263238', linewidth=2)
            ax.add_patch(arrow)
    
    plt.title('3D Neural Network Architecture', 
             fontsize=18, fontweight='bold', color='#1A237E', pad=25)
    
    ax.set_xlim(-1, len(layers)*(w+gap) - gap + 1)
    ax.set_ylim(-0.5, h + 0.5)
    ax.axis("off")
    ax.set_aspect("equal")
    fig.patch.set_facecolor('white')
    
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor='white', edgecolor='none')
    plt.close()

# ---------- node network view ------------------------------------------------
def node_network(layers: List[Dict], palette: str, out: str, size: Tuple[int,int]):
    fig, ax = plt.subplots(figsize=size, dpi=300)
    colours = PALETTES[palette]
    
    # Calculate positions
    n_layers = len(layers)
    layer_positions = []
    
    for i in range(n_layers):
        x = i * 3
        layer_positions.append(x)
    
    # Draw nodes
    for i, (L, x) in enumerate(zip(layers, layer_positions)):
        color = colours.get(L["type"], colours["conv"])
        
        # Draw node
        circle = Circle((x, 0), 0.8, facecolor=color, edgecolor='#1A237E', 
                       linewidth=2, alpha=0.9)
        ax.add_patch(circle)
        
        # Node label
        text_color = 'black' if L["type"] == "input" else 'white'
        ax.text(x, 0, L["name"], ha="center", va="center",
                color=text_color, weight="bold", size=10)
        
        # Parameter text below node
        ax.text(x, -1.5, L["txt"], ha="center", va="center",
                color='#333333', size=8)
    
    # Draw connections
    for i in range(len(layer_positions) - 1):
        start_x = layer_positions[i] + 0.8
        end_x = layer_positions[i + 1] - 0.8
        
        arrow = ConnectionPatch((start_x, 0), (end_x, 0), "data", "data",
                               arrowstyle="->", shrinkA=5, shrinkB=5,
                               mutation_scale=15, fc='#263238', ec='#263238', linewidth=2)
        ax.add_patch(arrow)
    
    plt.title('Neural Network Node View', 
             fontsize=18, fontweight='bold', color='#1A237E', pad=25)
    
    ax.set_xlim(-1, layer_positions[-1] + 1)
    ax.set_ylim(-2, 1)
    ax.axis("off")
    ax.set_aspect("equal")
    fig.patch.set_facecolor('white')
    
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor='white', edgecolor='none')
    plt.close()

# ---------- main CLI ----------------------------------------------------------
if __name__ == "__main__":
    P = argparse.ArgumentParser(description="Professional Neural Network Architecture Visualizer")
    P.add_argument("-a","--architecture", required=True, 
                   help="Architecture string like 'Conv2D(64)->MaxPool(2x2)->Dense(10)'")
    P.add_argument("-s","--style", choices=["clean","node","3d"], default="clean",
                   help="Visualization style")
    P.add_argument("-o","--output", default="nn.png",
                   help="Output filename (supports .png, .svg, .pdf)")
    P.add_argument("-c","--palette", choices=list(PALETTES.keys()), default="blue",
                   help="Color palette")
    P.add_argument("--figsize", default="18,5",
                   help="Figure size as 'width,height'")
    args = P.parse_args()

    print(f"🎨 Creating {args.style} style visualization...")
    print(f"📊 Architecture: {args.architecture}")
    print(f"🎨 Palette: {args.palette}")
    
    layers = parse(args.architecture)
    w,h = map(int,args.figsize.split(","))
    
    if args.style == "clean":
        clean_blocks(layers, args.palette, args.output, (w,h))
    elif args.style == "3d":
        blocks_3d(layers, args.palette, args.output, (w,h))
    elif args.style == "node":
        node_network(layers, args.palette, args.output, (w,h))
    
    print(f"✅ Diagram saved → {args.output}")
    print(f"📏 Size: {w}x{h} inches, 300 DPI")
    print(f"🎯 Ready for GitHub README, papers, or presentations!")
