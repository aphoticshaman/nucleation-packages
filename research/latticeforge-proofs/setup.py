#!/usr/bin/env python3
"""
Setup script for latticeforge-proofs package.
Apache 2.0 License
"""

from setuptools import setup, find_packages

setup(
    name="cic-inference",
    version="1.0.0",
    description="CIC Functional and Value Clustering for ML Inference Optimization",
    long_description=open("README.md").read() if __import__("os").path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    author="Crystalline Labs",
    author_email="research@crystallinelabs.io",
    license="Apache-2.0",
    url="https://github.com/crystallinelabs/cic-inference",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[],  # No required dependencies
    extras_require={
        "numpy": ["numpy>=1.20.0"],
        "test": ["pytest>=7.0.0", "pytest-cov>=4.0.0"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    keywords="machine-learning inference ensemble clustering phase-transition",
)
