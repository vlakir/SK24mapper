from __future__ import annotations

from pathlib import Path

"""
Compatibility shim: make `contours` a proper package facade.

This file used to be a standalone module that conflicted with the
`contours/` package. We now treat it as a package shim that:
- exposes a package search path so that `import contours.seeds` and
  `import contours.labels_overlay` work as expected;
- re-exports public helpers for backward compatibility.
"""

# Expose package path for submodules (treat this module as a package)
__path__ = [str(Path(__file__).parent / 'contours')]
