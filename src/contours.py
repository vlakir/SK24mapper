from __future__ import annotations

"""
Compatibility shim: make `contours` a proper package facade.

This file used to be a standalone module that conflicted with the
`contours/` package. We now treat it as a package shim that:
- exposes a package search path so that `import contours.seeds` and
  `import contours.labels_overlay` work as expected;
- re-exports public helpers for backward compatibility.
"""

import os

# Expose package path for submodules (treat this module as a package)
__path__ = [os.path.join(os.path.dirname(__file__), 'contours')]

# Backward-compatible re-exports

try:
    pass
except Exception:  # pragma: no cover - during early startup
    # If submodule isn't available yet, leave name undefined; callers importing
    # from contours.seeds will still work thanks to __path__ above.
    pass
