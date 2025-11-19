# Exploratory/Exp_UI/interface/drawing/explore_loading.py
from .utilities import build_text_item
from .animated_sequence import build_loading_frames

def build_loading_progress(template_item, progress):
    data_list = []
    x1, y1, x2, y2 = template_item["pos"]
    tw, th = (x2-x1), (y2-y1)

    # dim background
    data_list.append({"type": "rect", "pos": (x1,y1,x2,y2), "color": (0,0,0,0.6)})

    # centered box
    box_w, box_h = tw*0.7, th*0.5
    cx, cy = (x1+x2)/2, (y1+y2)/2
    bx1, by1 = cx - box_w/2, cy - box_h/2
    bx2, by2 = cx + box_w/2, cy + box_h/2

    # animated spinner
    spinner_side = box_w * 0.28
    data_list.append(build_loading_frames(cx, cy - box_h * 0.05, spinner_side))
    spinner_cy       = cy - box_h * 0.05
    spinner_top_y    = spinner_cy + spinner_side / 2
    spinner_bottom_y = spinner_cy - spinner_side / 2

    # box bg
    data_list.append({"type": "rect", "pos": (bx1,by1,bx2,by2), "color": (0.1,0.1,0.1,0.9)})

    # title
    title_fs = min(box_w, box_h) * 0.15
    data_list.append(build_text_item("Loading...", (bx1+bx2)/2, spinner_top_y + title_fs*0.6, title_fs, (1,1,1,1), 'CENTER', False))

    # progress bar
    bar_w, bar_h = box_w*0.8, box_h*0.15
    bar_x1, bar_y1 = (bx1+bx2)/2 - bar_w/2, by1 + box_h*0.25
    bar_x2, bar_y2 = bar_x1 + bar_w, bar_y1 + bar_h
    data_list.append({"type": "rect", "pos": (bar_x1,bar_y1,bar_x2,bar_y2), "color": (0.2,0.2,0.2,1)})
    data_list.append({"type": "rect", "pos": (bar_x1,bar_y1,bar_x1 + bar_w*progress,bar_y2), "color": (0.0,0.7,0.0,1)})

    # percent
    pct_fs = bar_h*0.7
    data_list.append(build_text_item(f"{int(progress*100)}%", (bar_x1+bar_x2)/2, spinner_bottom_y - pct_fs*1.25, pct_fs, (1,1,1,1), 'CENTER', False))

    # messages
    msg_fs = box_h * 0.10
    msg_y  = spinner_bottom_y - pct_fs*1.25 - msg_fs*1.6
    data_list.append(build_text_item("Your game will begin momentarily...", (bx1+bx2)/2, msg_y, msg_fs, (0.8,0.8,0.8,1), 'CENTER', False))
    tip_fs = msg_fs*0.8
    data_list.append(build_text_item("Please keep your cursor in the Blender window.", (bx1+bx2)/2, msg_y - tip_fs*1.35, tip_fs, (0.7,0.7,0.7,1), 'CENTER', False))
    return data_list
