########################################
# File: exp_custom_ui.py
# Purpose: Draw ephemeral or indefinite text in 3D Viewport,
#          using ratio-based margin & scale for resolution independence.
########################################

import bpy
import blf
import time
from .exp_time import get_game_time

_ui_draw_handler = None

_reaction_texts = []  # store ratio-based data
_ui_messages = []     # optional legacy debug messages

def add_text_reaction(
    text_str,
    anchor,
    margin_x,
    margin_y,
    scale,
    end_time=None,
    color=(1,1,1,1)
):
    item = {
        "text": text_str,
        "anchor": anchor,
        "margin_x": margin_x,
        "margin_y": margin_y,
        "scale": scale,
        "end_time": end_time,
        "color": color,
    }
    _reaction_texts.append(item)
    return item  # <--- return it so we can modify it further


def update_text_reactions():
    now = get_game_time()
    keep_list = []

    for it in _reaction_texts:
        sub = it.get("subtype", "STATIC")

        # ===============================
        #  A) OBJECTIVE_TIMER_DISPLAY
        # ===============================
        if sub == "OBJECTIVE_TIMER_DISPLAY":
            idx_str = it.get("objective_index", "")
            if idx_str.isdigit():
                idx = int(idx_str)
                scene = bpy.context.scene
                if 0 <= idx < len(scene.objectives):
                    objv = scene.objectives[idx]

                    # Either way, we can format it as “seconds”:
                    current_time_val = max(0.0, objv.timer_value)
                    it["text"] = format_hms(current_time_val)

            # Check indefinite vs ephemeral
            e_time = it.get("end_time")
            if e_time is None:
                keep_list.append(it)  # keep forever
            else:
                if now < e_time:
                    keep_list.append(it)  # still within time

        # ===============================
        #  B) OBJECTIVE (Simple Counter)
        # ===============================
        elif sub == "OBJECTIVE":
            idx_str = it.get("objective_index", "")
            if idx_str.isdigit():
                idx = int(idx_str)
                scene = bpy.context.scene
                if 0 <= idx < len(scene.objectives):
                    objv = scene.objectives[idx]
                    # Retrieve the new properties. Provide sensible defaults if they aren’t set.
                    prefix = it.get("custom_text_prefix", "")
                    suffix = it.get("custom_text_suffix", "")
                    include_counter = it.get("custom_text_include_counter", True)
                    if include_counter:
                        it["text"] = f"{prefix}{objv.current_value}{suffix}"
                    else:
                        it["text"] = f"{prefix}{suffix}"

            # Indefinite vs ephemeral
            e_time = it.get("end_time")
            if e_time is None:
                keep_list.append(it)
            else:
                if now < e_time:
                    keep_list.append(it)

        # ===============================
        #  C) STATIC
        # ===============================
        elif sub == "STATIC":
            e_time = it.get("end_time")
            if e_time is None:
                keep_list.append(it)
            else:
                if now < e_time:
                    keep_list.append(it)

    _reaction_texts[:] = keep_list






def add_ui_message(text, duration=3.0, indefinite=False):
    if indefinite:
        end_time = None
    else:
        end_time = time.time() + duration
    _ui_messages.append((text, end_time, indefinite))

def update_ui_messages():
    now = time.time()
    keep = []
    for (txt, et, indef) in _ui_messages:
        if indef or (et and et > now):
            keep.append((txt, et, indef))
    _ui_messages[:] = keep

def clear_ui_messages():
    _ui_messages.clear()

def clear_all_text():
    clear_text_reactions()
    clear_ui_messages()

def draw_ui_callback():
    # Find the 3D View "WINDOW" region
    region_3d = None
    window = bpy.context.window
    if not window or not window.screen:
        return

    for area in window.screen.areas:
        if area.type == 'VIEW_3D':
            for reg in area.regions:
                if reg.type == 'WINDOW':
                    region_3d = reg
                    break
            if region_3d:
                break

    if not region_3d:
        # Can't find a 3D 'WINDOW' region => bail
        return

    width = region_3d.width
    height = region_3d.height

    # ------------------------------------------------------------------------
    # Define a reference resolution (design target)
    ref_width = 1920.0
    ref_height = 1080.0

    # Compute a uniform scaling factor using the smaller ratio so that it fits both axes.
    scale_factor_w = width / ref_width
    scale_factor_h = height / ref_height
    scale_factor = min(scale_factor_w, scale_factor_h)

    # ------------------------------------------------------------------------
    # Define the grid based on the reference resolution.
    grid_columns = 20
    ref_grid_unit = ref_width / grid_columns          # grid unit at reference resolution
    grid_unit = ref_grid_unit * scale_factor           # horizontally scaled grid unit
    grid_unit_y = (ref_height / 20.0) * scale_factor     # vertical grid unit; adjust divisor as needed

    # ------------------------------------------------------------------------
    # Define maximum font size at the reference resolution and scale it.
    base_max_font_px = 180  # maximum font size when scale_int is 20 at the ref resolution
    max_font_px = base_max_font_px * scale_factor

    # ------------------------------------------------------------------------
    # 1) Reaction-based text drawing using the grid system and integer scaling.
    for it in _reaction_texts:
        text_str = it["text"]
        anchor   = it["anchor"]

        # Assume it["margin_x"] and it["margin_y"] are now integer offsets.
        margin_x_int = it["margin_x"]  # e.g., 0, 1, -1, etc.
        margin_y_int = it["margin_y"]

        # Convert grid offsets to pixel margins.
        mx_px = margin_x_int * grid_unit
        my_px = margin_y_int * grid_unit_y

        # Use the integer scale (value from 0 to 20) to compute font size.
        scale_int = it["scale"]  # integer in the range [0, 20]
        font_size_px = int((scale_int / 20.0) * max_font_px)
        if font_size_px < 10:
            font_size_px = 10  # Ensure a minimum for legibility

        color_rgba = it["color"]
        font_id = 0  # Default Blender font
        blf.size(font_id, font_size_px)

        # Get the dimensions of the text.
        tw, th = blf.dimensions(font_id, text_str)

        # Compute the anchored position using the pixel-based margins.
        draw_x, draw_y = compute_anchored_position(
            width, height, tw, th, anchor, mx_px, my_px
        )

        # Position the font and draw the text.
        blf.position(font_id, draw_x, draw_y, 0)
        r, g, b, a = color_rgba
        blf.color(font_id, r, g, b, a)
        blf.draw(font_id, text_str)

    # ------------------------------------------------------------------------
    # 2) Optional legacy debug messages (unchanged)
    font_id = 0
    blf.size(font_id, 20)
    line_gap = 24
    center_x = width // 2
    center_y = height // 2
    reversed_msgs = list(reversed(_ui_messages))

    for i, (msg_txt, msg_et, msg_indef) in enumerate(reversed_msgs):
        tw, th = blf.dimensions(font_id, msg_txt)
        x = center_x - (tw / 2)
        y = center_y + (i * line_gap)
        blf.position(font_id, x, y, 0)
        blf.color(font_id, 1, 1, 1, 1)
        blf.draw(font_id, msg_txt)




def compute_anchored_position(width, height, text_w, text_h, anchor, margin_x_px, margin_y_px):
    if anchor == 'TOP_LEFT':
        draw_x = margin_x_px
        draw_y = height - margin_y_px - text_h
    elif anchor == 'TOP_CENTER':
        draw_x = (width - text_w) / 2 + margin_x_px  # Add horizontal offset.
        draw_y = height - margin_y_px - text_h
    elif anchor == 'TOP_RIGHT':
        draw_x = width - margin_x_px - text_w
        draw_y = height - margin_y_px - text_h
    elif anchor == 'MIDDLE_LEFT':
        draw_x = margin_x_px
        draw_y = (height - text_h) / 2 + margin_y_px
    elif anchor == 'MIDDLE_CENTER':
        draw_x = (width - text_w) / 2 + margin_x_px  # Also adjust center horizontally.
        draw_y = (height - text_h) / 2 + margin_y_px
    elif anchor == 'MIDDLE_RIGHT':
        draw_x = width - margin_x_px - text_w
        draw_y = (height - text_h) / 2 + margin_y_px
    elif anchor == 'BOTTOM_LEFT':
        draw_x = margin_x_px
        draw_y = margin_y_px
    elif anchor == 'BOTTOM_CENTER':
        draw_x = (width - text_w) / 2 + margin_x_px  # Add the offset here.
        draw_y = margin_y_px
    elif anchor == 'BOTTOM_RIGHT':
        draw_x = width - margin_x_px - text_w
        draw_y = margin_y_px
    else:
        draw_x = margin_x_px
        draw_y = height - margin_y_px - text_h

    return draw_x, draw_y


def register_ui_draw():
    global _ui_draw_handler
    if _ui_draw_handler is None:
        _ui_draw_handler = bpy.types.SpaceView3D.draw_handler_add(
            draw_ui_callback, (), 'WINDOW', 'POST_PIXEL'
        )

def unregister_ui_draw():
    global _ui_draw_handler
    if _ui_draw_handler is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_ui_draw_handler, 'WINDOW')
        _ui_draw_handler = None
    clear_all_text()


def format_hms(seconds_left: float) -> str:
    """
    Convert total seconds => conditional 'Xh Xm X.Xs' format.
    Only shows hours if hours>0, only shows minutes if minutes>0, etc.
    e.g. '10.0s', '5m 2.3s', '1h 15.2s', or '1h 30m 12.0s'.
    If all are zero => '0.0s'.
    """
    if seconds_left < 0:
        seconds_left = 0.0

    hours = int(seconds_left // 3600)
    remain = seconds_left % 3600
    minutes = int(remain // 60)
    secs = remain % 60  # float

    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    # For seconds, you might want to show it even if 0.0,
    # or only if > 0.  Here we show it only if > 0:
    if secs > 0:
        parts.append(f"{secs:.1f}s")

    if not parts:
        # Means hours=0, minutes=0, secs=0 => show "0.0s"
        return "0.0s"

    return " ".join(parts)

def clear_text_reactions():
    """
    Removes all text items from the _reaction_texts list.
    """
    _reaction_texts.clear()




class EXPLORE_OT_PreviewCustomText(bpy.types.Operator):
    """Preview all custom text items for 3 seconds without starting the game."""
    bl_idname = "exploratory.preview_custom_text"
    bl_label = "Preview Custom Text"
    bl_options = {'REGISTER'}

    duration: bpy.props.FloatProperty(
        name="Duration (seconds)",
        default=5.0,
        description="How long the text is displayed"
    )

    def execute(self, context):
        print("Starting custom text preview operator...")
        
        # Clear any existing text reactions.
        print("Clearing all existing text reactions.")
        clear_all_text()
        
        # Get a reference time.
        current_time = get_game_time() or time.time()
        end_time = current_time + self.duration
        print(f"Current time: {current_time:.3f}, duration: {self.duration:.1f} sec, computed end_time: {end_time:.3f}")

        # Search for custom UI text reactions defined by the user.
        interactions = getattr(context.scene, "custom_interactions", None)
        custom_text_count = 0
        if interactions:
            print(f"Found {len(interactions)} custom interactions in the scene.")
            for inter in interactions:
                print("Processing interaction:", inter.name)
                for reaction in inter.reactions:
                    if reaction.reaction_type == "CUSTOM_UI_TEXT":
                        print("  -> Found custom UI text reaction:", reaction.name,
                              "Subtype:", reaction.custom_text_subtype)
                        # Local import to avoid circular dependency.
                        from .exp_reactions import execute_custom_ui_text_reaction
                        execute_custom_ui_text_reaction(reaction)
                        custom_text_count += 1
        else:
            print("No custom interactions found in the scene.")

        print(f"Total custom UI text reactions found and added: {custom_text_count}")

        # For previewing objective counters and timer displays, override the text field with dummy text.
        for item in _reaction_texts:
            subtype = item.get("subtype", "STATIC")
            if subtype == "OBJECTIVE":
                # Construct text from prefix, counter, and suffix.
                # In the preview, you can use a dummy counter value (e.g. 42)
                prefix = item.get("custom_text_prefix", "Objective: ")
                suffix = item.get("custom_text_suffix", " pts")
                # Optionally, check if the counter should be included.
                include_counter = item.get("custom_text_include_counter", True)
                if include_counter:
                    item["text"] = f"{prefix}42{suffix}"
                else:
                    item["text"] = f"{prefix}{suffix}"
                print(f"Set dummy objective counter text: {item['text']}")
            elif subtype == "OBJECTIVE_TIMER_DISPLAY":
                item["text"] = "00:42s"
                print(f"Set dummy timer text: {item['text']}")

        print("Total preview text reactions in _reaction_texts:", len(_reaction_texts))
        
        # Ensure the draw handler is registered so that text is drawn.
        from .exp_custom_ui import register_ui_draw
        register_ui_draw()

        # Request an immediate redraw so the UI is updated.
        context.area.tag_redraw()
        self.report({'INFO'}, f"Preview text added for {self.duration:.1f} seconds. (Custom texts: {custom_text_count})")
        
        # ----------------------------------------------------------------
        # Register a timer callback that clears all preview text after the duration.
        def clear_preview():
            print("Timer reached: clearing preview text items...")
            clear_all_text()
            # Request redraw on all VIEW_3D areas.
            for area in bpy.context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
            return None  # Returning None stops the timer.
        
        bpy.app.timers.register(clear_preview, first_interval=self.duration)
        # ----------------------------------------------------------------
        
        return {'FINISHED'}