"""
Nyx — a powerful open-source agentic coding assistant.
"""
from __future__ import annotations

__version__ = "0.2.1"

from nyx.config import Config
from nyx.agent import Agent

__all__ = ["Agent", "Config", "__version__"]
