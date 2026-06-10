# Plan: Fix Markdown Rendering in Terminal (ANSI Fallback)

## Problem

When Rich is **not** installed (or `--no-rich` is used), the AI agent's responses are printed **raw** to the terminal. Since the LLM generates responses in Markdown format (headings `##`, bold `**`, tables `|`, code blocks `` ``` ``, etc.), the user sees unformatted Markdown syntax instead of a clean plain-text response.

### Root Cause

| File | Line(s) | What happens |
|------|---------|--------------|
| [`nyx/cli_rich.py`](nyx/cli_rich.py:217) | 217-225 | `format_content()` wraps response in Rich's `Markdown(content)` — renders beautifully |
| [`nyx/cli_rich.py`](nyx/cli_rich.py:504) | 504-505 | Rich mode calls `console.print(format_content(result))` — Markdown is rendered |
| [`nyx/cli.py`](nyx/cli.py:467) | 467 | ANSI fallback prints `result` directly: `print(f"...> {result}")` — raw Markdown shown |
| [`nyx/cli.py`](nyx/cli.py:481) | 481 | Same issue in single-prompt mode |

## Solution Options

### Option A: Strip Markdown in the ANSI fallback (Recommended)

Add a `strip_markdown()` function in [`nyx/cli.py`](nyx/cli.py) that converts Markdown to plain text before printing.

**What it would strip:**
- `#` headings → plain text (remove `#` prefix)
- `**bold**` and `*italic*` → remove `*` markers
- `` `code` `` → remove backticks
- ```` ```code blocks```` → remove fences, keep content
- `| table | rows |` → remove table formatting, keep text
- `---` horizontal rules → remove
- `> blockquotes` → remove `>` prefix
- `- list items` → keep `-` as bullet
- `[text](url)` links → keep only `text`
- `![alt](url)` images → keep only `alt`

**Files to modify:**
- [`nyx/cli.py`](nyx/cli.py) — Add `strip_markdown()` function and use it in `run_ansi_interactive()` and `run_ansi_single()`

**Pros:**
- No change to the LLM's behavior or system prompt
- Works with any LLM that outputs Markdown
- Backward compatible
- Simple, predictable output

**Cons:**
- Imperfect stripping (some edge cases may slip through)
- Adds ~30 lines of code

---

### Option B: Add a system prompt instruction

Modify the default system prompt in [`nyx/config.py`](nyx/config.py) (line 29) to tell the AI to avoid Markdown.

**New system prompt:**
```
"You are a powerful agentic CLI assistant... Be concise, precise, and helpful. IMPORTANT: Do NOT use Markdown formatting in your responses. The terminal does not support Markdown rendering. Use plain text only."
```

**Files to modify:**
- [`nyx/config.py`](nyx/config.py) — Update `DEFAULT_CONFIG["system_prompt"]`

**Pros:**
- Very simple change (1 line)
- No post-processing needed

**Cons:**
- LLMs often ignore this instruction (they're trained to use Markdown)
- Inconsistent: Rich mode users would lose Markdown formatting too
- Doesn't fix existing responses or subagent responses
- Not reliable

---

### Option C: Both A + B (Best approach)

Combine both: strip Markdown in the ANSI fallback **and** add a system prompt hint.

**Files to modify:**
- [`nyx/cli.py`](nyx/cli.py) — Add `strip_markdown()` function
- [`nyx/config.py`](nyx/config.py) — Update system prompt (optional, as a hint)

## Recommended: Option A (Strip Markdown in ANSI fallback)

This is the most robust solution because:
1. It works regardless of what the LLM outputs
2. It doesn't affect Rich users (they still get beautiful Markdown rendering)
3. It's a pure client-side fix — no dependency on LLM compliance

### Implementation Details

**New function in [`nyx/cli.py`](nyx/cli.py):**

```python
import re

def strip_markdown(text: str) -> str:
    """Strip Markdown formatting for plain-text terminal display."""
    # Remove code blocks (```...```)
    text = re.sub(r'```[\s\S]*?```', '', text)
    # Remove inline code (`code`)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    # Remove images ![alt](url)
    text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'\1', text)
    # Remove links [text](url) -> text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # Remove heading markers (#)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # Remove bold/italic markers
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    text = re.sub(r'_([^_]+)_', r'\1', text)
    # Remove horizontal rules
    text = re.sub(r'^[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
    # Remove blockquote markers
    text = re.sub(r'^>\s?', '', text, flags=re.MULTILINE)
    # Clean up excessive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()
```

**Changes in [`nyx/cli.py`](nyx/cli.py):**
- Line 467: Change `print(f"{c('Agent', ASSISTANT_COLOR)}> {result}")` to `print(f"{c('Agent', ASSISTANT_COLOR)}> {strip_markdown(result)}")`
- Line 481: Same change for single-prompt mode