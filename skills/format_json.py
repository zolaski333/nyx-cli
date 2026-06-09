"""
Format JSON skill — pretty-prints or minifies JSON data.
"""
from __future__ import annotations

import json

name = "format_json"
description = "Format, pretty-print, or minify JSON data."
parameters = {
    "type": "object",
    "properties": {
        "json_string": {
            "type": "string",
            "description": "The JSON string to format",
        },
        "indent": {
            "type": "integer",
            "description": "Indentation level (0 for minified, default: 2)",
            "default": 2,
        },
    },
    "required": ["json_string"],
}


def execute(arguments: dict) -> str:
    json_string = arguments.get("json_string", "")
    indent = arguments.get("indent", 2)

    try:
        parsed = json.loads(json_string)
        if indent == 0:
            result = json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
        else:
            result = json.dumps(parsed, ensure_ascii=False, indent=indent)
        return result
    except json.JSONDecodeError as e:
        return f"Invalid JSON: {e}"