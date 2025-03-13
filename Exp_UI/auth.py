import os
import bpy
import requests
import threading
import urllib.parse
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler

from .main_config import (
    VALIDATE_TOKEN_ENDPOINT, LOGIN_ENDPOINT, LOGOUT_ENDPOINT, TOKEN_FILE,
    LOGIN_PAGE_ENDPOINT, BLENDER_LOGIN_SUCCESS_ENDPOINT, BLENDER_CALLBACK_URL, BASE_URL
)

import socketserver
import socket
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

# -----------------------------------------------------------------------------
# Internet Connection Checks
# -----------------------------------------------------------------------------

def is_internet_available():
    """Check if the internet is available by connecting to a known server."""
    try:
        # Use Google's public DNS server to test connectivity.
        socket.setdefaulttimeout(3)
        host = socket.gethostbyname("8.8.8.8")
        s = socket.create_connection((host, 53), 2)
        s.close()
        print("internet connectivity check passed")
        return True
    except Exception as e:
        print("Internet connectivity check failed:", e)
        return False


def ensure_internet_connection(context):
    """
    Checks if the internet is available.
    If not, logs out the user, disables the web UI, and returns False.
    Otherwise, returns True.
    """
    if not is_internet_available():
        print("Error: No internet connection detected. Logging out and disabling web UI.")
        clear_token()
        if hasattr(bpy.context.scene, "my_addon_data"):
            bpy.context.scene.my_addon_data.is_from_webapp = False
        bpy.context.scene.ui_current_mode = "GAME"
        return False
    return True

# -----------------------------------------------------------------------------
# Callback Server
# -----------------------------------------------------------------------------

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
