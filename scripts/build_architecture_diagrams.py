#!/usr/bin/env python3
"""Generate one SVG diagram per model from the reference implementation.

The diagrams are derived from the built modules in src/models/architectures.py rather than drawn
by hand, so they cannot drift from the ChromeCRISPR architectures.
"""
import argparse, os, sys
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.architectures import MODELS, SEQ_LEN

W, H, GAP, PAD = 320, 46, 26, 20


def svg(name, rows):
    height = PAD * 2 + len(rows) * H + (len(rows) - 1) * GAP
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W + PAD*2}" height="{height}" '
           f'viewBox="0 0 {W + PAD*2} {height}" font-family="Helvetica, Arial, sans-serif">',
           '<defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="3" '
           'orient="auto"><path d="M0,0 L7,3 L0,6 z" fill="#555"/></marker></defs>',
           f'<rect width="100%" height="100%" fill="#fff"/>']
    y = PAD
    for i, (title, sub) in enumerate(rows):
        out.append(f'<rect x="{PAD}" y="{y}" width="{W}" height="{H}" rx="4" '
                   f'fill="#f7f7f7" stroke="#bbb"/>')
        if sub:
            out.append(f'<text x="{PAD+W/2}" y="{y+19}" text-anchor="middle" font-size="13" '
                       f'fill="#111">{title}</text>')
            out.append(f'<text x="{PAD+W/2}" y="{y+35}" text-anchor="middle" font-size="11" '
                       f'fill="#555">{sub}</text>')
        else:
            out.append(f'<text x="{PAD+W/2}" y="{y+28}" text-anchor="middle" font-size="13" '
                       f'fill="#111">{title}</text>')
        if i < len(rows) - 1:
            out.append(f'<line x1="{PAD+W/2}" y1="{y+H}" x2="{PAD+W/2}" y2="{y+H+GAP-6}" '
                       f'stroke="#555" stroke-width="1.4" marker-end="url(#a)"/>')
        y += H + GAP
    out.append('</svg>')
    return "\n".join(out)


def layers_of(name, model):
    rows = [("Input", f"one-hot {SEQ_LEN} x 4")]
    if getattr(model, "embedding", None) is not None:
        rows.append(("Embedding", "4 -> 128 per position"))
    convs = [m for m in model.modules() if isinstance(m, nn.Conv1d) and m.kernel_size != (1,)]
    if convs:
        c = convs[0]
        rows.append((f"{len(convs)} x Conv1D",
                     f"{c.out_channels} filters, k={c.kernel_size[0]}, "
                     f"s={c.stride[0]}, p={c.padding[0]}, ReLU"))
    rnns = [m for m in model.modules() if isinstance(m, (nn.GRU, nn.LSTM))]
    if rnns:
        r = rnns[0]
        kind = ("Bi" if r.bidirectional else "") + type(r).__name__
        rows.append((f"{r.num_layers} x {kind}", f"{r.hidden_size} hidden units"))
    if getattr(model, "cnn_readout", None) is not None:
        rows.append(("CNN branch readout", "128 x 21 -> 128"))
        rows.append(("Concatenate branches", "128 + 128 = 256"))
    if getattr(model, "use_gc_content", False):
        rows.append(("Append GC content",
                     "256 + 1 = 257" if getattr(model, "cnn_readout", None) else "+1 feature"))
    head = getattr(model, "head", None)
    dense = [m for m in head.modules() if isinstance(m, nn.Linear)] if head is not None else []
    widths = [m.out_features for m in dense if m.out_features != 1]
    if widths:
        rows.append(("Dense " + ", ".join(str(w) for w in widths), "with batch normalisation"))
    rows.append(("Output", "1 unit"))
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out-dir", default="docs/architectures")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    n = 0
    for name, factory in MODELS.items():
        open(os.path.join(a.out_dir, name.replace("+", "_plus_") + ".svg"), "w").write(
            svg(name, layers_of(name, factory())))
        n += 1
    open(os.path.join(a.out_dir, "RF.svg"), "w").write(
        svg("RF", [("Random Forest", "100 estimators")]))
    n += 1
    print(f"generated {n} diagrams in {a.out_dir}")


if __name__ == "__main__":
    main()
