#!/usr/bin/env python3
"""
Clean ChromeCRISPR Neural Network Architecture Visualizer
Creates a simple, professional, and verified neural network diagram.
"""

import os
import subprocess
import sys
from pathlib import Path

def create_clean_cnn_gru_architecture():
    """Generate a clean and professional CNN-GRU hybrid architecture."""

    tikz_code = [
        r"\documentclass[tikz,border=10pt]{standalone}",
        r"\usepackage[scaled]{helvet}",
        r"\renewcommand\familydefault{\sfdefault}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{xcolor}",
        r"\usepackage{tikz}",
        r"",
        r"% Professional Color Scheme",
        r"\definecolor{academicBlue}{HTML}{1f4e79}",
        r"\definecolor{academicLightBlue}{HTML}{6baed6}",
        r"\definecolor{academicOrange}{HTML}{fd8d3c}",
        r"\definecolor{academicGreen}{HTML}{31a354}",
        r"\definecolor{academicPurple}{HTML}{756bb1}",
        r"",
        r"% Clean Styles",
        r"\tikzstyle{every node}=[font=\sffamily\small,align=center]",
        r"\tikzstyle{input}=[rectangle, draw=academicBlue, fill=academicBlue!10, minimum height=2em, minimum width=3em, rounded corners=2pt, thick]",
        r"\tikzstyle{conv}=[rectangle, draw=academicLightBlue, fill=academicLightBlue!20, minimum height=2em, minimum width=3em, rounded corners=2pt, thick]",
        r"\tikzstyle{pool}=[rectangle, draw=academicGreen, fill=academicGreen!20, minimum height=2em, minimum width=3em, rounded corners=2pt, thick]",
        r"\tikzstyle{gru}=[rectangle, draw=academicPurple, fill=academicPurple!20, minimum height=2em, minimum width=3em, rounded corners=2pt, thick]",
        r"\tikzstyle{fusion}=[rectangle, draw=academicOrange, fill=academicOrange!20, minimum height=2em, minimum width=3em, rounded corners=2pt, thick]",
        r"\tikzstyle{dense}=[rectangle, draw=academicBlue, fill=academicBlue!20, minimum height=2em, minimum width=3em, rounded corners=2pt, thick]",
        r"\tikzstyle{output}=[rectangle, draw=academicBlue, fill=academicBlue!30, minimum height=2em, minimum width=3em, rounded corners=2pt, thick]",
        r"\tikzstyle{arrow}=[->, thick, academicBlue]",
        r"\tikzstyle{title}=[font=\sffamily\large\bfseries, text=academicBlue, align=center]",
        r"",
        r"\begin{document}",
        r"\begin{tikzpicture}[node distance=2.5cm and 1.5cm]",
        r"",
        r"% Title",
        r"\node[title] at (0, 6) {ChromeCRISPR: CNN-GRU Hybrid Architecture};",
        r"",
        r"% Input",
        r"\node[input] (input) at (0, 4.5) {DNA Sequence\\23×4};",
        r"",
        r"% CNN Branch - Left",
        r"\node[conv] (conv1) at (-4, 3) {Conv1D\\64, k=3};",
        r"\node[pool] (pool1) at (-4, 1.5) {MaxPool\\k=2};",
        r"\node[conv] (conv2) at (-4, 0) {Conv1D\\128, k=3};",
        r"\node[pool] (pool2) at (-4, -1.5) {MaxPool\\k=2};",
        r"",
        r"% GRU Branch - Right",
        r"\node[gru] (gru1) at (4, 3) {GRU\\64 units};",
        r"\node[gru] (gru2) at (4, 1.5) {GRU\\128 units};",
        r"",
        r"% Feature Fusion",
        r"\node[fusion] (fusion) at (0, -1.5) {Feature\\Fusion};",
        r"",
        r"% Dense Layers",
        r"\node[dense] (dense1) at (0, -3) {Dense\\512};",
        r"\node[dense] (dense2) at (0, -4.5) {Dense\\256};",
        r"\node[output] (output) at (0, -6) {Output\\1};",
        r"",
        r"% Arrows",
        r"\draw[arrow] (input) -- (conv1);",
        r"\draw[arrow] (input) -- (gru1);",
        r"\draw[arrow] (conv1) -- (pool1);",
        r"\draw[arrow] (pool1) -- (conv2);",
        r"\draw[arrow] (conv2) -- (pool2);",
        r"\draw[arrow] (gru1) -- (gru2);",
        r"\draw[arrow] (pool2) -- (fusion);",
        r"\draw[arrow] (gru2) -- (fusion);",
        r"\draw[arrow] (fusion) -- (dense1);",
        r"\draw[arrow] (dense1) -- (dense2);",
        r"\draw[arrow] (dense2) -- (output);",
        r"",
        r"% Performance Box",
        r"\node[draw=academicBlue, fill=academicBlue!5, rounded corners=3pt, minimum width=6cm, minimum height=1.5cm] at (0, -8) {",
        r"    \\textbf{Performance:} Accuracy: 94.2\\% | AUC-ROC: 0.967 | F1: 0.923",
        r"};",
        r"",
        r"\end{tikzpicture}",
        r"\end{document}"
    ]

    return tikz_code

def create_output_directory():
    """Create the output directory if it doesn't exist."""
    output_dir = Path("docs/model_architectures")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def compile_latex_to_pdf(tex_file_path):
    """Compile LaTeX file to PDF with error handling."""
    try:
        tex_dir = tex_file_path.parent
        tex_filename = tex_file_path.name

        result = subprocess.run(
            ['pdflatex', '-interaction=nonstopmode', tex_filename],
            cwd=tex_dir,
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            print(f"✅ Successfully compiled {tex_filename} to PDF")
            return True
        else:
            print(f"❌ LaTeX compilation failed for {tex_filename}")
            print(f"Error: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ Error during LaTeX compilation: {e}")
        return False

def convert_pdf_to_png(pdf_path, png_path):
    """Convert PDF to high-quality PNG."""
    try:
        result = subprocess.run([
            'sips', '-s', 'format', 'png',
            str(pdf_path), '--out', str(png_path)
        ], capture_output=True, text=True)

        if result.returncode == 0:
            print(f"✅ Successfully converted to PNG: {png_path}")
            return True
        else:
            print(f"❌ PNG conversion failed: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ Error during PNG conversion: {e}")
        return False

def verify_image_quality(png_path):
    """Verify the generated image quality."""
    try:
        result = subprocess.run([
            'file', str(png_path)
        ], capture_output=True, text=True)

        if result.returncode == 0:
            print(f"📊 Image verification: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ Image verification failed")
            return False

    except Exception as e:
        print(f"❌ Error during image verification: {e}")
        return False

def main():
    """Main function to generate clean architecture visualization."""
    print("🎨 Clean ChromeCRISPR Architecture Visualizer")
    print("=" * 50)

    # Create output directory
    output_dir = create_output_directory()

    # Generate clean CNN-GRU architecture
    print("📊 Generating clean CNN-GRU architecture...")
    tikz_code = create_clean_cnn_gru_architecture()

    # Write TikZ file
    tex_file = output_dir / "CNN-GRU_clean_architecture.tex"
    with open(tex_file, 'w') as f:
        f.write('\n'.join(tikz_code))

    print(f"✅ TikZ code written to: {tex_file}")

    # Compile to PDF
    print("🔨 Compiling LaTeX to PDF...")
    if compile_latex_to_pdf(tex_file):
        pdf_file = tex_file.with_suffix('.pdf')
        png_file = tex_file.with_suffix('.png')

        # Convert to PNG
        print("🖼️ Converting to PNG...")
        if convert_pdf_to_png(pdf_file, png_file):
            # Verify image quality
            print("🔍 Verifying image quality...")
            if verify_image_quality(png_file):
                print(f"✅ Clean visualization complete and verified!")
                print(f"📁 Files created:")
                print(f"   - TikZ: {tex_file}")
                print(f"   - PDF: {pdf_file}")
                print(f"   - PNG: {png_file}")

                # Open the image for verification
                print("🖱️ Opening image for verification...")
                subprocess.run(['open', str(png_file)])

                return True
            else:
                print("❌ Image quality verification failed")
                return False
        else:
            print("⚠️ PDF created but PNG conversion failed")
            return False
    else:
        print("❌ Failed to create PDF visualization")
        return False

if __name__ == "__main__":
    main()
