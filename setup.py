#!/usr/bin/env python3
"""
Setup script for ChromeCRISPR.

ChromeCRISPR: Hybrid Machine Learning Model for CRISPR/Cas9 On-Target Activity Prediction
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text() if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    requirements = [line.strip() for line in requirements_path.read_text().splitlines()
                   if line.strip() and not line.startswith("#")]

setup(
    name="chromecrispr",
    version="1.0.0",
    author="Amirhossein Daneshpajouh, Megan Fowler, Kay C. Wiese",
    author_email="amir_dp@sfu.ca",
    description="Hybrid Machine Learning Model for CRISPR/Cas9 On-Target Activity Prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/ChromeCRISPR",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pre-commit>=3.3.0",
        ],
        "docs": [
            "sphinx>=7.1.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
        "gpu": [
            "torch>=2.0.0+cu118",
            "torchvision>=0.15.0+cu118",
        ],
    },
    entry_points={
        "console_scripts": [
            "chromecrispr-train=scripts.train_model:main",
            "chromecrispr-generate-data=scripts.generate_sample_data:main",
        ],
    },
    include_package_data=True,
    package_data={
        "chromecrispr": ["config/*.yaml", "data/*.csv"],
    },
    zip_safe=False,
    keywords="crispr, machine learning, deep learning, bioinformatics, genomics",
    project_urls={
        "Bug Reports": "https://github.com/your-username/ChromeCRISPR/issues",
        "Source": "https://github.com/your-username/ChromeCRISPR",
        "Documentation": "https://chromecrispr.readthedocs.io/",
    },
)
