# main_config.py
import os

# Determine the base URL based on the environment
# You can set an environment variable 'ADDON_ENV' to switch between environments
# Default to 'development' if not set
ENVIRONMENT = os.getenv('ADDON_ENV', 'development')

BASE_URLS = {
    'development': 'http://127.0.0.1:5000/blender_api/api',
    'production': 'https://exploratory.online/blender_api/api'
}


# Select the appropriate base URL
BASE_URL = BASE_URLS.get(ENVIRONMENT, BASE_URLS['development'])

# Define specific endpoints
LOGIN_ENDPOINT = f"{BASE_URL}/login"
LOGOUT_ENDPOINT = f"{BASE_URL}/logout"
DOWNLOAD_ENDPOINT = f"{BASE_URL}/download"
VALIDATE_TOKEN_ENDPOINT = f"{BASE_URL}/validate_token"
PACKAGE_DETAILS_ENDPOINT = f"{BASE_URL}/package_details"
PACKAGES_ENDPOINT = f"{BASE_URL}/packages"
LIKE_PACKAGE_ENDPOINT = f"{BASE_URL}/like"  # Example
COMMENT_PACKAGE_ENDPOINT = f"{BASE_URL}/comment"  # Example



# Define the path to the addon's folder
ADDON_FOLDER = os.path.dirname(__file__)

# Path to the login token file
TOKEN_FILE = os.path.join(ADDON_FOLDER, "login_token.txt")

# Subfolder in the addon directory for downloaded .blend files
WORLD_DOWNLOADS_FOLDER = os.path.join(ADDON_FOLDER, "World Downloads")
if not os.path.exists(WORLD_DOWNLOADS_FOLDER):
    os.makedirs(WORLD_DOWNLOADS_FOLDER)

SHOP_DOWNLOADS_FOLDER = os.path.join(ADDON_FOLDER, "Shop Downloads")

THUMBNAIL_CACHE_FOLDER = os.path.join(ADDON_FOLDER, "cache_thumbnails")
if not os.path.exists(THUMBNAIL_CACHE_FOLDER):
    os.makedirs(THUMBNAIL_CACHE_FOLDER)