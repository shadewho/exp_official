# api.py
import os
import requests
import shutil
import traceback
from .main_config import (
    LOGIN_ENDPOINT,
    LOGOUT_ENDPOINT,
    DOWNLOAD_ENDPOINT,
    VALIDATE_TOKEN_ENDPOINT,
    PACKAGE_DETAILS_ENDPOINT,
    PACKAGES_ENDPOINT,
    LIKE_PACKAGE_ENDPOINT,
    COMMENT_PACKAGE_ENDPOINT,
    SHOP_DOWNLOADS_FOLDER,
    USAGE_ENDPOINT
)
from .helper_functions import download_blend_file, append_scene_from_blend
from .auth import load_token, save_token, clear_token, is_internet_available
from .cache_manager import cache_manager
from .main_config import PACKAGE_DETAILS_ENDPOINT, THUMBNAIL_CACHE_FOLDER


def login(username, password):
    """
    Sends a login request to the server.

    Args:
        username (str): User's username.
        password (str): User's password.

    Returns:
        dict: Response data.
    """
    url = LOGIN_ENDPOINT
    payload = {"username": username, "password": password}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        if data.get("success"):
            token = data.get("token")
            save_token(token)
            # Login successful.
        else:
            # Login failed.
            pass
        return data
    except requests.RequestException as e:
        # Login request failed.
        raise


def logout():
    """
    Sends a logout request to the server.

    Returns:
        dict: Response data.
    """
    url = LOGOUT_ENDPOINT
    token = load_token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    try:
        response = requests.post(url, headers=headers)
        if response.status_code == 404:
            # Logout endpoint not found.
            return {"success": False, "message": "Logout endpoint not found."}
        response.raise_for_status()
        data = response.json()
        if data.get("success"):
            clear_token()
            # Logout successful.
        else:
            # Logout failed.
            pass
        return data
    except requests.RequestException as e:
        return {"success": False, "message": str(e)}
    except ValueError:
        return {"success": False, "message": "Invalid response from server."}


def validate_token():
    """
    Validates the current authentication token.

    Returns:
        bool: True if token is valid, False otherwise.
    """
    url = VALIDATE_TOKEN_ENDPOINT
    token = load_token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        if data.get("success"):
            # Token is valid.
            return True
        else:
            clear_token()
            # Token is invalid.
            return False
    except requests.RequestException as e:
        return False


def fetch_packages(params):
    """
    Fetches packages from the server based on provided parameters.

    Args:
        params (dict): Query parameters for fetching packages.

    Returns:
        dict: Response data containing packages.
    """
    # Check for internet connectivity
    if not is_internet_available():
        clear_token()  # Log out the user by clearing the token.
        raise Exception("No internet connection detected. You have been logged out.")

    url = PACKAGES_ENDPOINT
    token = load_token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get("success"):
            # Fetched packages successfully.
            pass
        else:
            # Failed to fetch packages.
            pass
        return data
    except requests.RequestException as e:
        raise


def like_package(file_id):
    """
    POST /like/<file_id>
    If user already liked, server returns 400 + { success: false, message: "Already liked" }
    We'll parse that ourselves and return data without raising an error.
    """
    token = load_token()
    if not token:
        raise Exception("Not logged in")

    url = f"{LIKE_PACKAGE_ENDPOINT}/{file_id}"
    headers = {"Authorization": f"Bearer {token}"}

    resp = requests.post(url, headers=headers)

    # If 400, parse JSON but do not raise an exception
    if resp.status_code == 400:
        data = resp.json()  # e.g. { "success": false, "message": "Already liked" }
        return data

    # For any other status code >= 400 (except 400), raise an error.
    if resp.status_code >= 400:
        resp.raise_for_status()

    data = resp.json()
    return data


def comment_package(file_id, comment_text):
    """
    POST /comment/<file_id>
    JSON body: { "comment_text": ... }
    Returns dict { success: bool, comment: {...} } or error.
    """
    token = load_token()
    if not token:
        raise Exception("Not logged in")

    url = f"{COMMENT_PACKAGE_ENDPOINT}/{file_id}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    payload = {"comment_text": comment_text}

    try:
        resp = requests.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        return data
    except requests.RequestException as e:
        raise Exception(f"Comment request failed: {e}")


def download_package(download_code):
    """
    Requests the download URL for a package using its download code.

    Args:
        download_code (str): The download code for the package.

    Returns:
        dict: Response data containing the download URL.
    """
    url = DOWNLOAD_ENDPOINT
    token = load_token()
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"} if token else {}
    payload = {"download_code": download_code}

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        if data.get("success"):
            # Download URL retrieved successfully.
            pass
        else:
            # Failed to retrieve download URL.
            pass
        return data
    except requests.RequestException as e:
        raise


def explore_package(pkg):
    """
    Download and either append a world or store a shop item.
    pkg is the dictionary from the server with fields like:
      - file_type ( 'world' or 'shop_item' )
      - download_code
      - ...
    Returns {'FINISHED'} or {'CANCELLED'}, plus optionally a message.
    """
    token = load_token()
    if not token:
        print("[ERROR] Not logged in.")
        return {'CANCELLED'}, "Not logged in"

    file_type = pkg.get("file_type", "world")
    download_code = pkg.get("download_code")
    if not download_code:
        print("[ERROR] No download_code in package data.")
        return {'CANCELLED'}, "No download code"

    # 1) Request the .blend file download URL from your Flask endpoint
    url = DOWNLOAD_ENDPOINT
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    payload = {"download_code": download_code}

    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code != 200:
            err_msg = f"API Error {response.status_code}: {response.text}"
            print("[ERROR]", err_msg)
            return {'CANCELLED'}, err_msg

        data = response.json()
        if not data.get("success"):
            err_msg = data.get("message", "Download failed")
            print("[ERROR]", err_msg)
            return {'CANCELLED'}, err_msg

        download_url = data["download_url"]
        # 2) Download the file
        local_blend_path = download_blend_file(download_url)
        if not local_blend_path:
            err_msg = "Failed to download .blend file."
            print("[ERROR]", err_msg)
            return {'CANCELLED'}, err_msg

        # 3) If it's a world, append the scene
        if file_type == "world":
            print("[INFO] Appending scene from downloaded .blend...")
            result = append_scene_from_blend(local_blend_path)
            return result, "Scene appended." if result == {'FINISHED'} else "Failed to append scene."

        # 4) If it's a shop item, store it in the Shop Downloads folder
        elif file_type == "shop_item":
            base_name = os.path.basename(local_blend_path)
            final_path = os.path.join(SHOP_DOWNLOADS_FOLDER, base_name)
            print(f"[INFO] Copying from {local_blend_path} to {final_path}")
            shutil.copy2(local_blend_path, final_path)
            return {'FINISHED'}, f"Shop item downloaded to: {final_path}"
        else:
            print(f"[ERROR] Unknown file_type: {file_type}")
            return {'CANCELLED'}, "Unknown file type"

    except Exception as e:
        traceback.print_exc()
        return {'CANCELLED'}, str(e)


def fetch_detail_for_file(file_id):
    """
    Fetches detailed information for a specific file/package.
    
    Args:
        file_id (int): The ID of the file/package.
    
    Returns:
        dict or None: Detailed package data if successful, else None.
    """
    token = load_token()
    url = f"{PACKAGE_DETAILS_ENDPOINT}/{file_id}"
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get("success"):
            return data
        else:
            # Failed to fetch package details.
            return None
    except requests.RequestException as e:
        return None
    except ValueError:
        return None
    

def get_usage_data():
    token = load_token()
    if not token:
        raise Exception("Not logged in")
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(USAGE_ENDPOINT, headers=headers, timeout=5)
    response.raise_for_status()
    data = response.json()
    if not data.get("success"):
        raise Exception(data.get("message", "Failed to fetch usage data"))
    return data
