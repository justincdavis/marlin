"""Package which imeplements the Marlin algorithm.

Classes
-------
Marlin
    Main class for the Marlin algorithm.

References
----------
Paper: https://dl.acm.org/doi/pdf/10.1145/3356250.3360044
"""
from . import _change, _marlin, _tracker
from ._change import ChangeDetector
from ._marlin import Marlin
from ._tracker import LucasKanadeTracker, MultiBoxTracker

__all__ = ["Marlin", "MultiBoxTracker", "LucasKanadeTracker", "ChangeDetector"]
__version__ = "0.0.1"
