#Exploratory/Exp_UI/internet/helpers.py
import bpy
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler

from ..main_config import (
    LOGIN_PAGE_ENDPOINT, BLENDER_LOGIN_SUCCESS_ENDPOINT, BLENDER_CALLBACK_URL
)

import socketserver
import socket
import time
from ..auth.helpers import save_token, clear_token
_last_internet_check_time = 0
_cached_internet_available = False
_CACHE_DURATION = 10  # seconds

# -----------------------------------------------------------------------------
# Internet Connection Checks
# -----------------------------------------------------------------------------

def is_internet_available():
    """Check if the internet is available by connecting to a known server.
       Caches the result for _CACHE_DURATION seconds to avoid repeated calls.
    """
    global _last_internet_check_time, _cached_internet_available
    current_time = time.time()
    # If the last check was recent, return the cached result.
    if current_time - _last_internet_check_time < _CACHE_DURATION:
        return _cached_internet_available

    try:
        # Use Google's public DNS server to test connectivity.
        socket.setdefaulttimeout(3)
        host = socket.gethostbyname("8.8.8.8")
        s = socket.create_connection((host, 53), 2)
        s.close()
        print("internet connectivity check passed")
        _cached_internet_available = True
    except Exception as e:
        print("Internet connectivity check failed:", e)
        _cached_internet_available = False

    _last_internet_check_time = current_time
    return _cached_internet_available


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
