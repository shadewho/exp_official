from __future__ import annotations
from ..dev_registry import register_section
from ..dev_draw_prims import draw_text
from ..dev_state import STATE, catalog_update_from_bus

def _channel_map(scene):
    m = {}
    try:
        for item in getattr(scene, "dev_hud_channels", []):
            m[str(item.key)] = bool(item.enabled)
    except Exception:
        pass
    return m

def sync_channels_from_catalog(scene):
    try:
        coll = getattr(scene, "dev_hud_channels", None)
        if coll is None:
            return
        have = {str(it.key) for it in coll}
        for grp in STATE.custom_groups_order:
            for full_key in STATE.custom_keys_by_group.get(grp, []):
                if full_key not in have:
                    it = coll.add()
                    it.key = str(full_key)
                    it.enabled = True
    except Exception:
        pass

class CustomSection:
    key = "custom"
    column = "RIGHT"
    order = 10
    prop_toggle = "dev_hud_show_custom"

    def measure(self, scene, STATE, BUS, scale, lh, width):
        if not getattr(scene, "dev_hud_show_custom", False): return 0
        catalog_update_from_bus(BUS)
        sync_channels_from_catalog(scene)

        filt_enabled = bool(getattr(scene, "dev_hud_custom_filter", False))
        filt_group   = str(getattr(scene, "dev_hud_channel_filter_group", "") or "").strip().upper()
        cmap = _channel_map(scene) if filt_enabled else None

        count = 0
        for grp in STATE.custom_groups_order:
            if filt_group and not grp.startswith(filt_group):
                continue
            keys = STATE.custom_keys_by_group.get(grp) or []
            if not keys:
                continue
            count += 1
            for k in keys:
                if cmap is not None and not cmap.get(k, True):
                    continue
                count += 1
        return count * lh

    def draw(self, x, y, scene, STATE, BUS, scale, lh, width):
        NORMAL = (1.0, 1.0, 1.0, 1.0)
        STALE  = (0.78, 0.82, 0.90, 0.90)

        filt_enabled = bool(getattr(scene, "dev_hud_custom_filter", False))
        filt_group   = str(getattr(scene, "dev_hud_channel_filter_group", "") or "").strip().upper()
        cmap = _channel_map(scene) if filt_enabled else None

        for grp in STATE.custom_groups_order:
            if filt_group and not grp.startswith(filt_group):
                continue
            keys = STATE.custom_keys_by_group.get(grp) or []
            if not keys:
                continue
            draw_text(x, y, f"{grp}", 12*scale); y -= lh
            for full_key in keys:
                if cmap is not None and not cmap.get(full_key, True):
                    continue
                short = full_key.split(".", 1)[-1]
                val   = STATE.custom_last_val.get(full_key, "—")
                age   = STATE.frame_idx - int(STATE.custom_last_seen.get(full_key, 0))
                colr  = NORMAL if age == 0 else STALE
                draw_text(x + int(16*scale), y, f"{short}: {val}", 12*scale, colr); y -= lh
        return y

register_section(CustomSection())
