"""Setup script for Nyx."""
from __future__ import annotations

from setuptools import find_packages, setup

setup(
    name="nyx",
    version="0.2.0",
    description="Zero-dependency agentic coding CLI — MCP, subagents, skills, web search, multi-provider",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Nyx Contributors",
    url="https://github.com/nyx-cli/nyx",
    project_urls={
        "Source": "https://github.com/nyx-cli/nyx",
        "Issues": "https://github.com/nyx-cli/nyx/issues",
    },
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=[],
    extras_require={
        "tui": ["rich>=13.0.0"],
        "dev": ["pytest>=7.0", "mypy>=1.0", "ruff>=0.1", "rich>=13.0.0"],
    },
    entry_points={
        "console_scripts": [
            "nyx=nyx.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development",
        "Topic :: Software Development :: Code Generators",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    license="MIT",
)