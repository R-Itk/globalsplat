"""Test the optional suppression of the vendored loader's "Skipped bad example"
log spam (``globalsplat.main._quiet_upstream_logs``)."""
import builtins

import pytest

pytest.importorskip("torch")
pytest.importorskip("pytorch_lightning")
pytest.importorskip("hydra")

from omegaconf import OmegaConf
import globalsplat.main as m


def _run_with_filter(flag):
    cfg = OmegaConf.create({"quiet_skipped_examples": flag})
    captured = []
    orig = builtins.print
    builtins.print = lambda *a, **k: captured.append(a[0] if a else "")
    try:
        m._quiet_upstream_logs(cfg)          # may re-wrap builtins.print
        builtins.print("Skipped bad example deadbeef. Context shape was ...")
        builtins.print("a normal log line")
    finally:
        builtins.print = orig
    return captured


def test_suppresses_skipped_example_lines_when_enabled():
    captured = _run_with_filter(True)
    assert "a normal log line" in captured
    assert not any(str(x).startswith("Skipped bad example") for x in captured)


def test_keeps_everything_when_disabled():
    captured = _run_with_filter(False)
    assert "a normal log line" in captured
    assert any(str(x).startswith("Skipped bad example") for x in captured)


if __name__ == "__main__":
    test_suppresses_skipped_example_lines_when_enabled()
    test_keeps_everything_when_disabled()
    print("test_logging: OK")
