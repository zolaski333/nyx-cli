# Release Checklist

Use this before sharing Nyx widely or cutting a public release.

## Package Identity

- Distribution name: `nyx-cli`
- Import package: `nyx`
- Console command: `nyx`

The `nyx` name is already used on PyPI by the Tor terminal monitor, so publish this project as `nyx-cli`.

## Local Verification

```bash
python -m pip install -e ".[dev,tui]"
python -m pytest
python -m ruff check .
python -m build
python -m twine check dist/*
nyx doctor
nyx --help
```

`mypy` is intentionally not a release gate yet. It is useful as an informational check, but the current codebase still has historical typing debt to pay down before strict type checking can be enforced in CI.

## Pre-Share Checks

- Confirm `git status` contains only intentional changes.
- Search for accidental secrets: `rg -n "sk-|api_key|token|password|secret" .`
- Verify `config.json`, `.nyx/`, `.nyx_memory/`, audit logs, build artifacts, and virtual environments are ignored.
- Test a clean install in a fresh virtual environment.
- Confirm README examples match the current CLI.
- Confirm `SECURITY.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, and `LICENSE` are present.
- Open GitHub Issues and Discussions if you want feedback from external users.
- Keep CI green with `pytest` and `ruff`; track `mypy` cleanup separately.

## Suggested First Release

Publish a GitHub release before publishing to PyPI. Mark it as beta/experimental and invite trusted users to install from Git:

```bash
pipx install "nyx-cli[tui] @ git+https://github.com/zolaski333/nyx-cli.git"
```

Move to PyPI only after a few external users have installed and run `nyx doctor` successfully.

## Announcement Template

Title:

```text
Nyx: an experimental, standard-library-first agentic coding CLI
```

Short post:

```text
I am sharing Nyx, an experimental agentic coding CLI written in Python.

It focuses on being small and hackable: standard-library-first core, optional Rich UI, multiple LLM providers, guarded file edits, MCP stdio support, local skills, subagents, repo maps, code search, and CI-friendly JSON output.

It is not a mature replacement for established coding agents yet. The best audience right now is developers who enjoy local tooling, provider flexibility, and inspecting how the agent loop works.

Install from Git:

pipx install "nyx-cli[tui] @ git+https://github.com/zolaski333/nyx-cli.git"

Then run:

nyx doctor
nyx

Feedback on safety, UX, provider support, and packaging would be especially useful.
```

Good first places to share:

- GitHub release notes.
- A small circle of developer friends or a private Discord/Slack first.
- Hacker News "Show HN" only after the install path is clean.
- Reddit communities such as r/Python or r/LocalLLaMA, with a clear experimental label.
- Python Discord or relevant AI tooling communities, asking specifically for testers.
