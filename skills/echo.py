"""
Echo skill — a minimal skill example.

Simply echoes back the input. Use as a template for creating new skills.
"""
from __future__ import annotations

name = "echo"
description = "Echoes back the input text. Useful for testing the skill system."
parameters = {
    "type": "object",
    "properties": {
        "text": {
            "type": "string",
            "description": "The text to echo back",
        },
    },
    "required": ["text"],
}


def execute(arguments: dict) -> str:
    text = arguments.get("text", "")
    return f"Echo: {text}"