import sys
from pathlib import Path

# Add root directory to sys.path and extend package search path
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

# Extend this package's search path to also search the root directory
if _root not in __path__:
    __path__.append(_root)
