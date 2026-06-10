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

> **⚡ Confort de saisie (REPL)** :
> - **Autocomplétion (Tab)** : Appuie sur `Tab` pour compléter automatiquement les commandes commençant par `/` (comme `/help`, `/model`, `/tools`) ainsi que les chemins de fichiers et dossiers.
> - **Historique des commandes** : Utilise les flèches `↑` et `↓` pour naviguer dans l'historique de tes commandes passées. L'historique est sauvegardé automatiquement entre les sessions dans le fichier `~/.nyx_history` (jusqu'à 1000 entrées).

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

### Pipe mode (stdin)

Nyx supports pipe mode — when stdin is not a TTY (e.g., piped data), the piped content is
automatically prepended to your prompt context:

```bash
# Pipe file content as context
cat main.py | nyx -p "review this code for bugs"

# Pipe command output
git diff | nyx -p "summarise these changes"

# Multi-line pipe
curl -s https://api.example.com/data.json | nyx -p "explain this JSON structure"
```

> **Note**: Pipe mode works by reading all stdin into a context preamble. The `-p` flag is still
> required to specify the actual prompt. If no `-p` is given with piped input, Nyx falls back
> to interactive mode.

### 🚀 CI/CD Integration (`--json`)

For CI/CD pipelines, use `--json` mode to get structured output:

```bash
# JSON output with cost tracking, session ID, timing
nyx --json -p "lint all Python files"

# Example output:
# {
#   "status": "success",
#   "prompt": "lint all Python files",
#   "result": "Fixed 3 issues...",
#   "duration_seconds": 12.45,
#   "session_id": "a1b2c3d4e5f6",
#   "cost": 0.0023,
#   "llm_calls": 2,
#   "tool_calls": 5
# }
```

The `--json` flag requires `--prompt`/`-p` and outputs newline-delimited JSON with:
- `status`: `"success"` or `"error"`
- `result` / `error`: The output or error message
- `duration_seconds`: Wall-clock time
- `session_id`: Unique session identifier
- `cost`: Estimated USD cost
- `llm_calls`, `tool_calls`: Usage statistics

### Flags

| Flag | Description |
|------|-------------|
| `-p, --prompt` | Run a single prompt and exit |
| `-c, --config` | Path to custom config.json |
| `-m, --model` | Override model (e.g. `openai/gpt-4o`) |
| `--provider` | Override provider (`openrouter`, `openai`, `anthropic`) |
| `-d, --dir` | Working directory for the AI (default: current dir) |
| `--project` | Alias for `--dir` |
| `--json` | JSON output mode for CI/CD (requires `--prompt`) |
| `--no-stream` | Disable streaming output |
| `--no-color` | Disable ANSI color output |
| `--no-rich` | Force basic CLI even if Rich is installed |
| `-v, --verbose` | Enable verbose/debug logging |

### Interactive Commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/model` | Show current model |
| `/model <name>` | Change model on the fly |
| `/clear` | Clear conversation context |
| `/tools [N]` | List all available tools (paginated, optional page N) |
| `/memory [N]` | Show memory status with paginated entries |
| `/conversations [N]` | List saved conversations (paginated, optional page N) |
| `/switch <id>` | Switch to a saved conversation (supports partial ID) |
| `/reset` | Reset agent + disconnect MCP |
| `/exit` | Quit |

> **Tip**: `/tools 2`, `/memory 3`, `/conversations 2` navigate paginated views.
> `/switch abc` switches to conversation with ID starting with "abc".

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