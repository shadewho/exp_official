# Exp_Game/developer/__init__.py
"""
Developer Tools Module

Provides debug toggles, console output controls, and diagnostic tools
for addon developers and advanced users.
"""

from .dev_properties import register_properties, unregister_properties
from .dev_panel import DEV_PT_DeveloperTools
from . import rig_analyzer

__all__ = [
    'register_properties',
    'unregister_properties',
    'DEV_PT_DeveloperTools',
    'rig_analyzer',
]


def register():
    """Register developer module components."""
    rig_analyzer.register()


def unregister():
    """Unregister developer module components."""
    rig_analyzer.unregister()
