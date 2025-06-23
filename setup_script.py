"""
Setup script for AI Personal Finance Assistant
"""

from setuptools import setup, find_packages
import os

# Read README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ai-finance-assistant",
    version="1.0.0",
    author="Your Name",
    author_email="your-email@example.com",
    description="AI-powered personal finance assistant with transaction analysis and predictive modeling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ai-finance-assistant",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "advanced": [
            "tensorflow>=2.13.0",
            "torch>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "finance-assistant=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.yaml", "*.yml"],
    },
    project_urls={
        "Bug Reports": "https://github.com/yourusername/ai-finance-assistant/issues",
        "Source": "https://github.com/yourusername/ai-finance-assistant",
        "Documentation": "https://github.com/yourusername/ai-finance-assistant/wiki",
    },
)
