"""Make the repo root importable so the tests can ``import globalsplat``
regardless of where pytest is launched from."""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
