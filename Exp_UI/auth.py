# auth.py

import os
import bpy
import requests
from .main_config import  (VALIDATE_TOKEN_ENDPOINT, LOGIN_ENDPOINT,
                            LOGOUT_ENDPOINT, TOKEN_FILE, LOGIN_PAGE_ENDPOINT,
                            BLENDER_LOGIN_SUCCESS_ENDPOINT, BLENDER_CALLBACK_URL

)
import logging
import threading
import http.server
from http.server import BaseHTTPRequestHandler
import socketserver
import urllib.parse
import webbrowser


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

class CallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urllib.parse.urlparse(self.path)
        if parsed_path.path == '/callback':
            params = urllib.parse.parse_qs(self.path)
            params = urllib.parse.parse_qs(parsed_path.query)
            token = params.get("token", [None])[0]
            if token:
                print(f"Received token: {token}")
                save_token(token)
                self.send_response(302)
                self.send_header("Location", BLENDER_LOGIN_SUCCESS_ENDPOINT)
                self.send_header("Connection", "close")
                self.end_headers()
                try:
                    self.wfile.flush()
                except Exception as e:
                    print("Error flushing response:", e)
                # Shutdown the server in a separate thread so the response can complete.
                threading.Thread(target=self.server.shutdown, daemon=True).start()
            else:
                self.send_error(400, "Missing token parameter.")
        else:
            self.send_error(404)


def start_local_server(port=8000):
    with socketserver.TCPServer(("", port), CallbackHandler) as httpd:
        print(f"Local server running on port {port}...")
        httpd.serve_forever()
        print("Local server shut down.")

def initiate_login():
    import urllib.parse, webbrowser
    callback_url = BLENDER_CALLBACK_URL
    login_url = f"{LOGIN_PAGE_ENDPOINT}?callback={urllib.parse.quote(callback_url)}"
    webbrowser.open(login_url)
    print("Opened browser for login at:", login_url)