# Exploratory/Exp_UI/interface/drawing/detail_content.py
import os, bpy, gpu, blf
from gpu_extras.batch import batch_for_shader
from .utilities import format_relative_time, build_text_item
from .config import BACK_BUTTON_PATH, EXPLORE_BUTTON_PATH, MISSING_THUMB
from .fonts import get_font_id

# in-memory image/texture cache (no disk DB)
_LOADED_IMAGES, _LOADED_TEX = {}, {}

def _get_image(path):
    if not path:
        return None
    img = _LOADED_IMAGES.get(path)
    try:
        _ = img.name
    except Exception:
        img = None

    if img is None:
        try:
            img = bpy.data.images.load(path, check_existing=True)
        except Exception:
            return None

    # Ensure UI-safe settings even for cached images
    try:
        cs = getattr(img, "colorspace_settings", None)
        if cs and getattr(cs, "name", None) != "Non-Color":
            cs.name = "Non-Color"   # critical: bypass color management for UI art
    except Exception:
        pass
    try:
        img.alpha_mode = 'STRAIGHT'  # harmless for JPG (no alpha), prevents fringes for PNGs
    except Exception:
        pass

    _LOADED_IMAGES[path] = img
    return img


def _get_texture(img):
    if not img:
        return None
    key = img.name
    tex = _LOADED_TEX.get(key)
    if tex:
        return tex
    try:
        tex = gpu.texture.from_image(img)
        _LOADED_TEX[key] = tex
        return tex
    except Exception:
        return None

def build_detail_content(template_item):
    """
    DETAIL view only:
      - Back (closes the UI)
      - Explore (runs your explore pipeline)
      - Large thumbnail
      - Text block (title/author/desc/likes/downloads/date[/votes])
    """
    if not template_item:
        return []

    data_list = []
    x1, y1, x2, y2 = template_item["pos"]

    # Back button
    try:
        img = _get_image(BACK_BUTTON_PATH)
        if img:
            tex    = _get_texture(img)
            shader = gpu.shader.from_builtin('IMAGE')
            w, h   = img.size
            aspect = (w / h) if h else 1.0
            bw = 0.10 * (x2 - x1)
            bh = bw / aspect
            mtop = 0.05 * (y2 - y1)
            mleft= 0.02 * (x2 - x1)
            bx1  = x1 + mleft
            bx2  = bx1 + bw
            by2  = y2 - mtop
            by1  = by2 - bh
            verts  = [(bx1,by1),(bx2,by1),(bx2,by2),(bx1,by2)]
            coords = [(0,0),(1,0),(1,1),(0,1)]
            batch  = batch_for_shader(shader, 'TRI_FAN', {"pos": verts, "texCoord": coords})
            data_list.append({"shader": shader, "batch": batch, "texture": tex, "name": "Back_Icon", "pos": (bx1,by1,bx2,by2)})
    except Exception as e:
        print(f"[Detail] Back button failed: {e}")

    # Explore button (always — shop removed)
    try:
        eimg = _get_image(EXPLORE_BUTTON_PATH)
        if eimg:
            tex    = _get_texture(eimg)
            shader = gpu.shader.from_builtin('IMAGE')
            w, h   = eimg.size
            aspect = (w / h) if h else 1.0
            btn_w  = 0.30 * (x2 - x1)
            btn_h  = btn_w / aspect
            off_x  = 0.25 * (x2 - x1)
            off_y  = -0.40 * (y2 - y1)
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            ex1 = cx + off_x - btn_w/2
            ey1 = cy + off_y - btn_h/2
            ex2 = ex1 + btn_w
            ey2 = ey1 + btn_h
            verts  = [(ex1,ey1),(ex2,ey1),(ex2,ey2),(ex1,ey2)]
            coords = [(0,0),(1,0),(1,1),(0,1)]
            batch  = batch_for_shader(shader, 'TRI_FAN', {"pos": verts, "texCoord": coords})
            data_list.append({"shader": shader, "batch": batch, "texture": tex, "name": "Explore_Icon", "pos": (ex1,ey1,ex2,ey2)})
    except Exception as e:
        print(f"[Detail] Explore button failed: {e}")

    # Enlarged thumbnail (uses Scene.selected_thumbnail → fallback)
    try:
        path = bpy.context.scene.selected_thumbnail or ""
        if not os.path.exists(path):
            path = MISSING_THUMB
        img = _get_image(path)
        tex = _get_texture(img) if img else None
        if tex:
            shader = gpu.shader.from_builtin('IMAGE')
            # square
            tw   = 0.05 * (x2 - x1)
            size = tw * 8
            cx   = (x1 + x2) / 2
            cy   = (y1 + y2) / 2
            off_x= -0.25 * (x2 - x1)
            off_y= -0.075 * (y2 - y1)
            ax1  = cx - size/2 + off_x
            ay1  = cy - size/2 + off_y
            ax2  = ax1 + size
            ay2  = ay1 + size
            verts  = [(ax1,ay1),(ax2,ay1),(ax2,ay2),(ax1,ay2)]
            coords = [(0,0),(1,0),(1,1),(0,1)]
            batch  = batch_for_shader(shader, 'TRI_FAN', {"pos": verts, "texCoord": coords})
            data_list.append({"shader": shader, "batch": batch, "texture": tex, "name": "Enlarged_Thumbnail", "pos": (ax1,ay1,ax2,ay2)})
    except Exception as e:
        print(f"[Detail] Thumbnail failed: {e}")

    # Text block
    build_item_detail_text(data_list, template_item)
    return data_list


def build_item_detail_text(data_list, template_item):
    """Keep your dynamic text fit logic; remove shop bits."""
    scene = bpy.context.scene
    ad    = scene.my_addon_data
    if getattr(ad, "file_id", 0) <= 0:
        return

    title        = getattr(ad, "package_name", "")
    author       = getattr(ad, "author", getattr(ad, "uploader", ""))
    description  = getattr(ad, "description", "")
    likes        = int(getattr(ad, "likes", 0) or 0)
    downloads    = int(getattr(ad, "download_count", 0) or 0)
    upload_date  = getattr(ad, "upload_date", "")
    file_type    = getattr(ad, "file_type", "world")
    votes        = int(getattr(ad, "vote_count", 0) or 0)
    uploaded_rel = format_relative_time(upload_date)

    lines = [
        f"Title: {title}", "",
        f"Author: {author}", "",
        f"Description: {description}", "",
        f"Likes: ♥ {likes}", "",
        f"Downloads: {downloads}",
    ]
    if file_type == "event":
        lines += ["", f"Votes: ★ {votes}"]
    lines += ["", f"Uploaded: {uploaded_rel} ago", ""]

    x1,y1,x2,y2 = template_item["pos"]
    t_w, t_h = (x2-x1), (y2-y1)

    # same right column geometry as before
    L,R,T,B = 0.50, 0.05, 0.25, 0.15
    col_x1, col_x2 = x1 + L*t_w, x2 - R*t_w
    col_y2 = y2 - T*t_h

    btn_top_y = None
    for itm in data_list:
        if itm.get("name") in ("Explore_Icon",):
            btn_top_y = itm["pos"][3]
            break

    PADDING = 0.02 * t_h
    col_y1  = max(y1 + B*t_h, (btn_top_y + PADDING) if btn_top_y else y1)
    col_w, col_h = col_x2 - col_x1, col_y2 - col_y1

    font_id = get_font_id()
    INIT_RATIO, MIN_RATIO = 0.075, 0.0325
    SHRINK = 0.95
    LEAD, BLANK_LEAD = 1.15, 0.40

    def line_h(size, blank=False): return size*(BLANK_LEAD if blank else LEAD)

    def wrap_para(text, max_w, size):
        blf.size(font_id, int(size))
        words, out, buf = text.split(), [], ""
        for w in words:
            test = (buf + " " + w).strip()
            if blf.dimensions(font_id, test)[0] > max_w and buf:
                out.append(buf); buf = w
            else: buf = test
        out.append(buf); return out

    def wrap_all(src, max_w, size):
        res = []
        for ln in src:
            if not ln.strip(): res.append("")
            else: res.extend(wrap_para(ln, max_w, size))
        return res

    font_sz = t_h*INIT_RATIO
    while True:
        wrapped = wrap_all(lines, col_w, font_sz)
        needed  = sum(line_h(font_sz, not ln.strip()) for ln in wrapped)
        if needed <= col_h or font_sz <= t_h*MIN_RATIO:
            break
        font_sz *= SHRINK

    blf.size(font_id, int(font_sz))
    cursor_y = col_y2
    for logical in wrap_all(lines, col_w, font_sz):
        h = line_h(font_sz, not logical.strip())
        if logical.strip():
            data_list.append(build_text_item(
                text=logical, x=col_x1, y=cursor_y,
                size=font_sz, color=(1,1,1,1), alignment='LEFT', multiline=False
            ))
        cursor_y -= h
