import os

GRID_COLUMNS = 4
GRID_ROWS = 2
THUMBNAILS_PER_PAGE = 8
TEMPLATE_ASPECT_RATIO = 2.0

OFFSET_FACTOR = 0.1
THUMBNAIL_SPACING_FACTOR = 0.1

THUMBNAIL_TEXT_OFFSET_RATIO = 0.12 # 5% of thumbnail height
THUMBNAIL_TEXT_SIZE_RATIO   = 0.12  # 12% of thumbnail height as font size

THUMBNAIL_TEXT_SIZE = 14            # Font size for thumbnail titles
THUMBNAIL_TEXT_COLOR = (1, 1, 1, 1) # RGBA white
THUMBNAIL_TEXT_ALIGNMENT = 'CENTER'

TEMPLATE_MARGIN_HORIZONTAL = 0.05
TEMPLATE_MARGIN_TOP = 0.2
TEMPLATE_MARGIN_BOTTOM = 0.1

LOADING_IMAGE_SCALE = 0.2     # 20% of the template width
LOADING_IMAGE_OFFSET_X = -.40  # relative offset from center in X
LOADING_IMAGE_OFFSET_Y = 0.41 # relative offset from center in Y


# Paths
TEMPLATE_IMAGE_PATH = os.path.join(os.path.dirname(__file__), "ui_templates", "thumbnail_boarder.jpg")
ARROW_LEFT_PATH  = os.path.join(os.path.dirname(__file__), "ui_templates", "arrow_left.png")
ARROW_RIGHT_PATH = os.path.join(os.path.dirname(__file__), "ui_templates", "arrow_right.png")
CLOSE_WINDOW_PATH = os.path.join(os.path.dirname(__file__), "ui_templates", "close_page.png")
BACK_BUTTON_PATH = os.path.join(os.path.dirname(__file__), "ui_templates", "back_button.png")
EXPLORE_BUTTON_PATH = os.path.join(os.path.dirname(__file__), "ui_templates", "explore_button.png")
LOADING_IMAGE_PATH = os.path.join(os.path.dirname(__file__), "ui_templates", "loading.jpg")

LAST_VIEWPORT_SIZE = {"width": 0, "height": 0}
