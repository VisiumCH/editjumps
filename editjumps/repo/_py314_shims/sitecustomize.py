"""Compat shim for a Python 3.14 / mlflow 3.14.0 import crash.

Python 3.14 removed importlib.abc.Traversable / TraversableResources
(deprecated since 3.12, moved to importlib.resources.abc). mlflow 3.14.0's
mlflow.assistant.skill_installer still imports the old location, so `mlflow
server` / `mlflow ui` crash with `ImportError: cannot import name 'Traversable'
from 'importlib.abc'` on Python 3.14. This is a confirmed, currently open
upstream bug: https://github.com/mlflow/mlflow/issues/24155

Not needed on Python <3.14, or once mlflow ships a fix - safe to delete this
whole editjumps/repo/_py314_shims/ directory (and its use in editjumps/repo/mlflow_ui.sh)
when that lands.

`sitecustomize.py` is auto-imported by Python at startup if its directory is
on PYTHONPATH (see editjumps/repo/mlflow_ui.sh), so this patches importlib.abc
before mlflow ever imports it - no code changes to mlflow itself needed.
"""
import importlib.abc
import importlib.resources.abc as _resources_abc

# setattr (not direct `importlib.abc.Traversable = ...`) because the whole
# point of this shim is that these names no longer exist on importlib.abc in
# 3.14 - a direct assignment trips static type checkers (ty) with an
# unresolved-attribute error on names the stubs correctly say are gone.
if not hasattr(importlib.abc, "Traversable"):
    setattr(importlib.abc, "Traversable", _resources_abc.Traversable)
if not hasattr(importlib.abc, "TraversableResources"):
    setattr(importlib.abc, "TraversableResources", _resources_abc.TraversableResources)
