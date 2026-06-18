# Security Policy

Nyx is an experimental local coding agent. It can read files, write files, run shell commands, call configured MCP servers, and execute local Python skills. Treat it as developer tooling for trusted workspaces, not as a security boundary.

## Supported Versions

Security fixes are handled on the default branch until the project starts publishing stable releases. If packaged releases are published, only the latest minor version is expected to receive fixes.

## Reporting a Vulnerability

Please report security issues privately by opening a GitHub security advisory for this repository. If that is not available, contact the maintainer privately before opening a public issue.

Include:

- Affected version or commit SHA.
- Operating system and Python version.
- Minimal reproduction steps.
- Whether the issue requires a malicious prompt, malicious repository, malicious MCP server, malicious skill, or only normal usage.
- Expected impact, such as arbitrary file write, command execution, secret exposure, sandbox escape, or denial of service.

Please do not include real API keys, private source code, or user data in reports.

## Security Boundaries

Nyx includes guardrails, but they are not complete isolation:

- Shell command prompts and deny rules are best-effort UX controls.
- File sandbox checks reduce accidental writes outside a project, but OS-level isolation is required for hostile inputs.
- MCP servers and Python skills are trusted local code.
- `--yolo` disables most approval prompts and should be limited to disposable or fully trusted projects.
- Rollback/history files are convenience features, not backups.

For higher-risk testing, run Nyx inside a VM, container, or throwaway checkout with a least-privilege API key.
