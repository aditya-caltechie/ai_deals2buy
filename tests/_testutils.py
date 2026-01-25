from __future__ import annotations

import sys
from pathlib import Path


def add_src_to_syspath() -> None:
    """
    Ensure `import core.framework` etc. works when tests are run from repo root.
    """
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

