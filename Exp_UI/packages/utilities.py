#Exploratory/Exp_UI/packages/utilities.py

import requests
from ..main_config import (
    DOWNLOAD_ENDPOINT,
    LIKE_PACKAGE_ENDPOINT,
    COMMENT_PACKAGE_ENDPOINT,
)
from ..auth.helpers import load_token

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
