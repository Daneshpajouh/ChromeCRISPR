#!/usr/bin/env python3
"""
Setup script for ChromeCRISPR.

ChromeCRISPR: Hybrid Machine Learning Model for CRISPR/Cas9 On-Target Activity Prediction
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="chromecrispr",
    version="1.0.0",
    author="ChromeCRISPR Contributors",
    description="Deep learning framework for CRISPR guide RNA efficiency prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/ChromeCRISPR",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    include_package_data=True,
    package_data={
        "": ["*.pth", "*.json", "*.md"],
    },
)
