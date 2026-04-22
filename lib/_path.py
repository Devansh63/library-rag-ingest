"""
Ensure the project root is on sys.path so scripts can import from lib/.

Import this module at the top of any script under scripts/ before any
other project imports:

    import lib._path  # noqa: F401 (side-effect import)

This is needed because uv runs scripts with the project root available but
does not install the project as an editable package by default (package=false
in pyproject.toml). The explicit sys.path insert is more portable than
relying on PYTHONPATH being set in the shell environment.
"""

import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
