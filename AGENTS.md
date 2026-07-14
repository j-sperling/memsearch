# memsearch Instructions

## Scope

Semantic memory search package and agent plugin sources.

## Commands

```bash
uv run pytest
uv run ruff check src tests
```

## Conventions

- Keep CLI, Python API, and plugin behavior aligned when changing shared memory semantics.
- Markdown memory files are source of truth; vector stores are rebuildable indexes.
- Avoid unattended runtime paths that install or upgrade tools dynamically.
- Update docs when plugin commands, config keys, or install paths change.
