<div align="center">

# ⚡ Nyx

**The zero-dependency agentic coding CLI**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**Zero external dependencies • MCP native • Multi-provider • Subagents • Skills • Web search**

</div>

---

## ✨ Why Nyx?

Nyx is an **agentic coding CLI** that runs on **pure Python 3.10+ standard library** — no `pip install`, no `node_modules`, no `cargo`. Clone it and it works instantly.

| Feature | Nyx | Claude Code | Codex CLI | Open Interpreter | Aider | Goose |
|---------|:---:|:-----------:|:---------:|:----------------:|:-----:|:-----:|
| Zero dependencies | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ (Go) |
| MCP native | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ |
| Plugin skills | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ |
| Subagents (sync + parallel) | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Web search (no API key) | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ |
| Multi-provider | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ |
| Open source | ✅ MIT | ❌ | ✅ MIT | ✅ AGPL | ✅ Apache 2.0 | ✅ Apache 2.0 |

---

## 🚀 Quick Start

### 1. Prérequis

- **Python 3.10+** ([télécharger](https://www.python.org/downloads/))
- Une clé API chez [OpenRouter](https://openrouter.ai/keys), [OpenAI](https://platform.openai.com/api-keys) ou [Anthropic](https://console.anthropic.com/)

### 2. Installation

<details>
<summary><b>🐧 Linux / macOS</b></summary>

```bash
# Cloner le projet
git clone https://github.com/nyx-cli/nyx.git
cd nyx

# Configurer ta clé API
export OPENROUTER_API_KEY="sk-or-..."
# ou : export OPENAI_API_KEY="sk-..."
# ou : export ANTHROPIC_API_KEY="sk-ant-..."

# Installation globale (recommandée)
pip install -e ".[tui]"

# Lancer depuis n'importe quel dossier
cd /chemin/vers/mon/projet
nyx
```

> **Note** : Si tu es sur Ubuntu/Debian et que `pip` bloque avec `externally-managed-environment`, utilise `pipx` :
> ```bash
> pipx install ".[tui]"
> nyx
> ```
</details>

<details>
<summary><b>🪟 Windows</b></summary>

```powershell
# Cloner le projet
git clone https://github.com/nyx-cli/nyx.git
cd nyx

# Configurer ta clé API
$env:OPENROUTER_API_KEY = "sk-or-..."
# ou : $env:OPENAI_API_KEY = "sk-..."
# ou : $env:ANTHROPIC_API_KEY = "sk-ant-..."

# Installation globale (recommandée)
pip install -e ".[tui]"

# Lancer depuis n'importe quel dossier
cd C:\chemin\vers\mon\projet
nyx
```
</details>

<details>
<summary><b>🐳 Docker / sans installation</b></summary>

```bash
# Sans installation — depuis le dossier du projet
python -m nyx.cli
```
</details>

---

## 🎮 Usage

### Interactive REPL

```bash
# Va dans le dossier où tu veux travailler
cd /chemin/vers/mon/projet

# Lance Nyx — il détecte automatiquement le répertoire courant
nyx
```

### Single prompt

```bash
nyx -p "explain how this Python code works"
```

### Pipe mode

```bash
cat main.py | nyx -p "review this code for bugs"
```

### Custom config

```bash
nyx -c ./myconfig.json
```

### Override model/provider

```bash
nyx -m "openai/gpt-4o"
nyx --provider anthropic
```

### Working directory

```bash
# Par défaut, Nyx utilise le répertoire courant
cd /mon/projet
nyx

# Tu peux aussi spécifier un répertoire différent
nyx --dir /autre/chemin
nyx --project /autre/chemin
```

### Flags

| Flag | Description |
|------|-------------|
| `-p, --prompt` | Run a single prompt and exit |
| `-c, --config` | Path to custom config.json |
| `-m, --model` | Override model (e.g. `openai/gpt-4o`) |
| `--provider` | Override provider (`openrouter`, `openai`, `anthropic`) |
| `-d, --dir` | Working directory for the AI (default: current dir) |
| `--project` | Alias for `--dir` |
| `--no-stream` | Disable streaming output |
| `--no-color` | Disable ANSI color output |
| `--no-rich` | Force basic CLI even if Rich is installed |

### Interactive Commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/model` | Show current model |
| `/model <name>` | Change model on the fly |
| `/clear` | Clear conversation context |
| `/tools` | List all available tools |
| `/memory` | Show memory status |
| `/conversations` | List saved conversations |
| `/reset` | Reset agent + disconnect MCP |
| `/exit` | Quit |

---

## 🧠 Features

### 🔌 MCP Support

Connect any [Model Context Protocol](https://modelcontextprotocol.io) server:

```json
{
  "mcp_servers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/dir"],
      "enabled": true
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "enabled": true,
      "env": { "GITHUB_TOKEN": "ghp_..." }
    }
  }
}
```

### 🎯 Skill System

Create Python skills in the `skills/` directory — they're auto-discovered and exposed as tools:

```python
# skills/my_skill.py
name = "my_skill"
description = "Does something useful"
parameters = {
    "type": "object",
    "properties": {
        "input": {"type": "string", "description": "Some input"}
    },
    "required": ["input"],
}

def execute(arguments: dict) -> str:
    return f"Processed: {arguments['input']}"
```

### 👥 Subagents

Spawn child agents for complex tasks — synchronously or in parallel:

```
You> analyse the codebase and generate a refactoring plan

Agent> I'll spawn a subagent for code analysis...
[Subagent:code-analysis] Analysing...
```

### 🌐 Web Search

Built-in DuckDuckGo search — no API key needed. The agent can search the web and fetch pages.

### 💾 Persistent Memory

Conversations are automatically saved to disk with smart summarisation. Switch between conversations, search past context, and never lose your work.

### 🔧 Built-in Tools

| Tool | Description |
|------|-------------|
| `web_search` | Search the internet (DuckDuckGo, free) |
| `web_fetch` | Fetch and extract text from any URL |
| `subagent_run` | Spawn a subagent for a subtask |
| `parallel_subagents` | Run multiple subagents in parallel |
| `memory_save` | Save notes to persistent memory |
| `memory_recall` | Search past conversations |
| `execute_command` | Run shell commands |
| `read_file` | Read files from the filesystem |
| `write_file` | Write/create files |
| `append_file` | Append to existing files |
| `list_files` | List directory contents |
| `finish` | Signal task completion |

---

## ⚙️ Configuration

### Environment Variables (recommended)

```bash
export OPENROUTER_API_KEY="sk-or-..."
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export NYX_MODEL="openai/gpt-4o"       # Override default model
export NYX_PROVIDER="anthropic"         # Override default provider
```

### config.json

See [`config.example.json`](config.example.json) for all available options.

Priority chain: **Environment variables > config.json > Defaults**

---

## 📁 Project Structure

```
nyx/
├── __init__.py          # Package init
├── cli.py               # CLI entry point (argparse, REPL)
├── cli_rich.py          # Rich TUI (optional dependency)
├── config.py            # Configuration (env, JSON, defaults)
├── agent.py             # Agentic loop + built-in tools
├── mcp_client.py        # MCP server connection (JSON-RPC stdio)
├── skill_manager.py     # Dynamic skill loading
├── subagent.py          # Subagent spawning & management
├── async_subagent.py    # Parallel subagent execution
├── web_search.py        # DuckDuckGo search + web fetch
├── memory.py            # Persistent memory with summarisation
└── providers/
    ├── __init__.py      # Provider factory
    ├── base.py          # Abstract base + types
    ├── openrouter.py    # OpenRouter/OpenAI-compatible
    ├── openai_provider.py
    └── anthropic_provider.py
skills/                  # User-defined skills (auto-discovered)
├── echo.py              # Example skill
└── format_json.py       # Example skill
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      Nyx CLI                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐  │
│  │  Agent   │  │  MCP     │  │  Skills  │  │ Memory │  │
│  │  (loop)  │◄─┤  Client  │  │  Manager │  │ Manager│  │
│  └────┬─────┘  └──────────┘  └──────────┘  └────────┘  │
│       │                                                  │
│  ┌────▼─────┐  ┌──────────┐  ┌──────────────────────┐   │
│  │ Provider │  │Subagents │  │  Built-in Tools       │   │
│  │ (LLM)    │  │(sync/par)│  │  (web, file, shell…)  │   │
│  └──────────┘  └──────────┘  └──────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

---

## 🤝 Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Quick start for contributors

```bash
git clone https://github.com/nyx-cli/nyx.git
cd nyx
pip install -e ".[dev]"
# Make your changes, then:
ruff check .
mypy nyx/
pytest
```

---

## 📜 License

MIT — see [LICENSE](LICENSE) for details.

---

<div align="center">
  <sub>Built with ❤️ and zero dependencies.</sub>
</div>