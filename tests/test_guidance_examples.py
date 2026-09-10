"""Offline guidance contracts; no provider calls or normal memory-store writes."""

import os
import re
import subprocess
from pathlib import Path

from click.testing import CliRunner

ROOT = Path(__file__).resolve().parents[1]


def test_candidate_heredocs_preserve_content(tmp_path):
    fake = tmp_path / "memsearch"
    fake.write_text('#!/bin/sh\ncat > "$CAPTURE_BODY"\n')
    fake.chmod(0o755)
    body = "## Literal example\n\n1. Keep `$HOME` and $(not_a_command).\n2. Preserve apostrophe: user's.\n"
    paths = list((ROOT / "plugins").glob("*/skills/memory-to-skill/SKILL.md"))
    assert len(paths) == 4
    for path in paths:
        text = path.read_text()
        command = next(
            block for block in re.findall(r"```bash\n(.*?)```", text, re.S) if "memsearch skills add" in block
        )
        command = re.sub(r"(?<=<<'SKILL_BODY'\n).*?(?=SKILL_BODY)", body, command, flags=re.S)
        capture = tmp_path / (path.parts[-4] + ".md")
        result = subprocess.run(
            ["bash", "-c", command],
            env={**os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"], "CAPTURE_BODY": str(capture)},
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert capture.read_text() == body
    claude = ROOT / "plugins/claude-code/skills/memory-to-skill/SKILL.md"
    assert "context: fork" not in claude.read_text().split("---", 2)[1]


def test_project_collection_round_trip_with_offline_embeddings(tmp_path, monkeypatch):
    from memsearch import config, core
    from memsearch.cli import cli

    class OfflineEmbedder:
        dimension = 8
        model_name = "synthetic-guidance-fixture"
        batch_size = 10

        async def embed(self, texts):
            return [[1.0] + [0.1] * 7 for _ in texts]

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "GLOBAL_CONFIG_PATH", tmp_path / "absent-global.toml")
    monkeypatch.setattr(core, "get_provider", lambda *a, **k: OfflineEmbedder())
    (tmp_path / ".memsearch.toml").write_text('[milvus]\ncollection = "isolated_guidance_fixture"\n')
    notes = tmp_path / ".research/notes"
    notes.mkdir(parents=True)
    (notes / "test.md").write_text("# Synthetic experiment\n\nThe azure otter fixture preserves collection identity.\n")
    database = str(tmp_path / "isolated.db")
    runner = CliRunner()
    indexed = runner.invoke(cli, ["index", str(notes), "--milvus-uri", database])
    assert indexed.exit_code == 0, indexed.output + repr(indexed.exception)
    assert "Indexed 1 chunks." in indexed.output
    found = runner.invoke(cli, ["search", "azure otter", "--milvus-uri", database, "--json-output"])
    assert found.exit_code == 0, found.output + repr(found.exception)
    assert "azure otter fixture" in found.output
    # Explicit overrides also remain symmetric; a different collection is empty.
    empty = runner.invoke(
        cli, ["search", "azure otter", "--milvus-uri", database, "--collection", "different_fixture", "--json-output"]
    )
    assert empty.exit_code == 0, empty.output
    assert "azure otter fixture" not in empty.output
