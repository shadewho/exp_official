# Exploratory/Exp_UI/interface/operators/utilities.py
import requests
from ...main_config import (
    PACKAGE_DETAILS_ENDPOINT, PACKAGES_ENDPOINT
)
from ...auth.helpers import load_token
from ...internet.helpers import is_internet_available, clear_token

#used in display operator
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
    

#used in display and apply operators
def build_filter_signature(scene):
    """
    Returns a stable string representing the current UI filters.
    Extend as you add new filter properties.
    """
    return "|".join([
        scene.package_item_type,
        scene.package_sort_by,
        scene.package_search_query.strip(),
        scene.event_stage,
        scene.selected_event,
    ])


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
        response = requests.get(url, headers=headers, params=params, timeout=30)
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