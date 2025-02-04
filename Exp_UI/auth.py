# auth.py

import os
import requests
from .main_config import VALIDATE_TOKEN_ENDPOINT, LOGIN_ENDPOINT, LOGOUT_ENDPOINT, TOKEN_FILE
import logging

# Configure logging for auth.py
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Adjust as needed

if not logger.hasHandlers():
    logging.basicConfig(level=logging.DEBUG)

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

def validate_token():
    """
    Validates the current authentication token.
    Returns True if token is valid, False otherwise.
    """
    token = load_token()
    if not token:
        logger.warning("No token found for validation.")
        return False

    headers = {"Authorization": f"Bearer {token}"}
    try:
        response = requests.get(VALIDATE_TOKEN_ENDPOINT, headers=headers, timeout=5)
        response.raise_for_status()
        data = response.json()
        if data.get("success"):
            logger.info("Token is valid.")
            return True
        else:
            logger.warning("Token validation failed.")
            clear_token()
            return False
    except requests.RequestException as e:
        logger.error(f"Token validation request failed: {e}")
        return False

def is_logged_in():
    """Check if the user is logged in by validating the token."""
    return validate_token()
