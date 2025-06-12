#!/usr/bin/env python3
"""
Setup script for Meta-Optimization Framework
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="meta-optimization-framework",
    version="0.1.0",
    author="Ryan Oates",
    author_email="ryan.oates@ucsb.edu",
    description="A comprehensive framework for cognitive-inspired deep learning optimization",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/Surfer12/meta-optimization-framework",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=22.0",
            "flake8>=5.0",
            "mypy>=1.0",
            "pre-commit>=2.20",
            "sphinx>=5.0",
            "sphinx-rtd-theme>=1.2",
            "jupyter>=1.0",
            "notebook>=6.5",
        ],
        "docs": [
            "sphinx>=5.0",
            "sphinx-rtd-theme>=1.2",
            "myst-parser>=0.18",
            "sphinx-autodoc-typehints>=1.19",
        ],
        "experiments": [
            "wandb>=0.13",
            "optuna>=3.0",
            "mlflow>=2.0",
            "tensorboard>=2.10",
        ],
    },
    entry_points={
        "console_scripts": [
            "meta-optimize=core.meta_optimization:main",
            "randomness-study=experiments.randomness_impact_study:main",
            "failure-museum=utils.failure_documentation:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    project_urls={
        "Bug Reports": "https://github.com/Surfer12/meta-optimization-framework/issues",
        "Source": "https://github.com/Surfer12/meta-optimization-framework",
        "Documentation": "https://meta-optimization-framework.readthedocs.io/",
    },
)