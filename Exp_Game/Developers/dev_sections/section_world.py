from __future__ import annotations
from ..dev_registry import register_section
from ..dev_draw_prims import draw_text
from ..dev_state import world_counts
from ..dev_utils import safe_modal

class WorldSection:
    key = "world"
    column = "LEFT"
    order = 20
    prop_toggle = "dev_hud_show_world"

    def measure(self, scene, STATE, BUS, scale, lh, width):
        return 0 if not getattr(scene, "dev_hud_show_world", False) else 1*lh

    def draw(self, x, y, scene, STATE, BUS, scale, lh, width):
        modal = safe_modal()
        dyn_active, dyn_bvhs, dyn_total, stat_total = world_counts(scene, modal)
        draw_text(x, y, f"World  Dyn Active {dyn_active}/{dyn_total}   Dyn BVHs {dyn_bvhs}   Static {stat_total}", 12*scale); y -= lh
        return y

register_section(WorldSection())
