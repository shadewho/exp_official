from __future__ import annotations
from typing import List, Optional, Tuple, Dict

class _SectionBase:
    key: str
    column: str  # "LEFT" | "RIGHT"
    order: int
    prop_toggle: Optional[str] = None
    def measure(self, scene, STATE, BUS, scale: int, lh: int, width: int) -> int: ...
    def draw(self, x: int, y: int, scene, STATE, BUS, scale: int, lh: int, width: int) -> int: ...

class _Registry:
    """
    Duplicate-safe, order-stable registry for HUD sections.
    Keyed by (column, key). Re-registering replaces the old instance.
    """
    def __init__(self):
        self._left: List[_SectionBase] = []
        self._right: List[_SectionBase] = []
        self._by_key: Dict[Tuple[str, str], _SectionBase] = {}

    def clear(self):
        self._left.clear()
        self._right.clear()
        self._by_key.clear()

    def _arr_for(self, col: str) -> List[_SectionBase]:
        return self._left if col == "LEFT" else self._right

    def register(self, section: _SectionBase):
        col = getattr(section, "column", "LEFT")
        key = getattr(section, "key", section.__class__.__name__)
        k = (col, key)

        # Replace existing entry (avoid duplicates across reloads)
        old = self._by_key.get(k)
        if old is not None:
            arr = self._arr_for(col)
            try:
                arr.remove(old)
            except ValueError:
                pass

        self._by_key[k] = section
        arr = self._arr_for(col)
        arr.append(section)
        arr.sort(key=lambda s: int(getattr(s, "order", 0)))

    def active(self, scene, column: str):
        arr = self._left if column == "LEFT" else self._right
        for s in arr:
            if s.prop_toggle:
                try:
                    if not bool(getattr(scene, s.prop_toggle, False)):
                        continue
                except Exception:
                    continue
            yield s

REGISTRY = _Registry()

def register_section(section: _SectionBase):
    REGISTRY.register(section)
