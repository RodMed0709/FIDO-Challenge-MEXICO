"""Pytest bootstrap for the FIDO repo.

Two things must happen before any test module is imported, or the suite either
fails to import or crashes the interpreter outright:

1. `src/` on `sys.path`, so `import fido...` resolves. There is no
   `src/__init__.py`, and the repo is used without being pip-installed.

2. `KMP_DUPLICATE_LIB_OK`. On Windows the Anaconda MKL runtime and the one
   vendored inside torch both load libiomp5, and the duplicate aborts the
   process. The abort surfaces as a bare interpreter crash with a dump of
   extension modules and no traceback, which reads like a bug in the test
   rather than an environment clash. The same workaround was already needed to
   run the vendored scorer (see submissions/r04-interim-joint/README.md).

Must be set before torch is imported, hence a root-level conftest.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
