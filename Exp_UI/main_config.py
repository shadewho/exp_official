# main_config.py
import os

# Determine the base URL based on the environment
# You can set an environment variable 'ADDON_ENV' to switch between environments
# Default to 'development' if not set
ENVIRONMENT = os.getenv('ADDON_ENV', 'development')

BASE_URLS = {
    'development': 'http://127.0.0.1:5000/blender_api/api',
    'production': 'https://exploratory.online//blender_api/api'
}

# Define a new configuration for user profiles:
USER_PROFILE_BASE_URLS = {
    'development': 'http://127.0.0.1:5000/user',
    'production': 'https://exploratory.online/user'
}

# Select the appropriate base URL
BASE_URL = BASE_URLS.get(ENVIRONMENT, BASE_URLS['development'])
USER_PROFILE_BASE_URL = USER_PROFILE_BASE_URLS.get(ENVIRONMENT, USER_PROFILE_BASE_URLS['development'])
# Define specific endpoints
# Define specific endpoints
LOGIN_ENDPOINT = f"{BASE_URL}/login"
LOGOUT_ENDPOINT = f"{BASE_URL}/logout"
DOWNLOAD_ENDPOINT = f"{BASE_URL}/download"
VALIDATE_TOKEN_ENDPOINT = f"{BASE_URL}/validate_token"
PACKAGE_DETAILS_ENDPOINT = f"{BASE_URL}/package_details"
PACKAGES_ENDPOINT = f"{BASE_URL}/packages"
LIKE_PACKAGE_ENDPOINT = f"{BASE_URL}/like"
COMMENT_PACKAGE_ENDPOINT = f"{BASE_URL}/comment"
USAGE_ENDPOINT = f"{BASE_URL}/usage"

if ENVIRONMENT == 'development':
    LOGIN_PAGE_ENDPOINT = "http://127.0.0.1:5000/login"
    # This is where Blender’s callback server will redirect the browser after login.
    BLENDER_CALLBACK_URL = "http://localhost:8000/callback"
    # This is the success page served by your Flask app.
    BLENDER_LOGIN_SUCCESS_ENDPOINT = "http://127.0.0.1:5000/blender_login_successful"
else:
    # Set production endpoints as needed.
    LOGIN_PAGE_ENDPOINT = "https://your-production-domain.com/login"
    BLENDER_CALLBACK_URL = "http://localhost:8000/callback"
    BLENDER_LOGIN_SUCCESS_ENDPOINT = "https://your-production-domain.com/blender_login_successful"


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