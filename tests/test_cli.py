"""Smoke test for the CLI ingest path (no LLM / network needed)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("chromadb")

from ragloop.cli import main  # noqa: E402


def test_cli_ingest_indexes_files(tmp_path, capsys):
    (tmp_path / "policy.md").write_text(
        "Refunds within 30 days of delivery.\n\nFree shipping on orders over $50."
    )
    rc = main(["ingest", str(tmp_path)])
    assert rc == 0
    assert "Ingested" in capsys.readouterr().out


def test_cli_requires_subcommand():
    with pytest.raises(SystemExit):
        main([])
