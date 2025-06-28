import os
import bpy
import requests
from ..main_config import (
    VALIDATE_TOKEN_ENDPOINT, TOKEN_FILE,
)
# -----------------------------------------------------------------------------
# Token File Helpers
# -----------------------------------------------------------------------------

def save_token(token):
    """Save the token to a file."""
    with open(TOKEN_FILE, "w") as file:
        file.write(token)

def load_token():
    """Load the token from a file."""
    if not os.path.exists(TOKEN_FILE):
        return None
    with open(TOKEN_FILE, "r") as file:
        return file.read().strip()

def clear_token():
    """Clear the token file."""
    if os.path.exists(TOKEN_FILE):
        os.remove(TOKEN_FILE)

# -----------------------------------------------------------------------------
# Token Validation and Expiry
# -----------------------------------------------------------------------------

def validate_token():
    """
    Validates the current authentication token.
    Returns True if token is valid, False otherwise.
    """
    token = load_token()
    if not token:
        print("Warning: No token found for validation.")
        return False
    from ..internet.helpers import is_internet_available
    if not is_internet_available():
        print("Error: No internet connection detected. Logging out.")
        clear_token()  # Log the user out immediately.
        return False

    headers = {"Authorization": f"Bearer {token}"}
    try:
        response = requests.get(VALIDATE_TOKEN_ENDPOINT, headers=headers, timeout=5)
        response.raise_for_status()
        data = response.json()
        if data.get("success"):
            print("Info: Token is valid.")
            return True
        else:
            print("Warning: Token validation failed.")
            clear_token()
            return False
    except requests.RequestException as e:
        print(f"Error: Token validation request failed: {e}")
        return False

def is_logged_in():
    """Check if the user is logged in by validating the token."""
    return validate_token()

def token_expiry_check():
    """
    Timer callback that checks whether the current login token is valid.
    If not, it clears the token and updates the UI.
    Returns the time (in seconds) until the next check.
    """
    if not validate_token():
        clear_token()
        bpy.context.scene.my_addon_data.is_from_webapp = False
        print("[INFO] Your login token has expired. You have been logged out automatically.")
    return 60.0  # Check every 60 seconds