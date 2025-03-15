
# addon_data.py
import bpy
from bpy.types import PropertyGroup
from bpy.props import (
    BoolProperty,
    IntProperty,
    StringProperty,
    CollectionProperty
)
from .main_config import USER_PROFILE_BASE_URL

class MyAddonComment(PropertyGroup):
    """
    Stores a single comment (author, text, etc.).
    """
    author: StringProperty(name="Author", default="")
    text: StringProperty(name="Text", default="")
    timestamp: StringProperty(name="Timestamp", default="")

class MyAddonSceneProps(PropertyGroup):
    """
    Main property group that keeps track of 
    whether this scene is from Exploratory, 
    plus details like file_id, package name, comments, etc.
    """
    is_from_webapp: BoolProperty(
        name="From Exploratory Web App",
        description="Was this scene appended from Exploratory?",
        default=False
    )

    file_id: IntProperty(
        name="File ID",
        description="ID from the Exploratory web app database",
        default=0
    )

    package_name: StringProperty(
        name="Package Name",
        default=""
    )

    author: StringProperty(
        name="Author",
        default=""
    )

    likes: IntProperty(
        name="Likes",
        default=0
    )

    # If you want to store the author's profile link:
    profile_url: StringProperty(
        name="Profile URL",
        default=""
    )

    # **New Properties**
    description: StringProperty(
        name="Description",
        description="Detailed description of the package",
        default=""
    )
    upload_date: StringProperty(
        name="Upload Date",
        description="Date when the package was uploaded",
        default=""
    )

    # Comment storage:
    comments: CollectionProperty(type=MyAddonComment)
    comment_page: IntProperty(
        name="Comment Page",
        default=1,
        min=1
    )


    active_comment_index: IntProperty(
        name="Active Comment Index",
        default=0
    )


    subscription_tier: StringProperty(
        name="Subscription Tier",
        default="Free"
    )
    downloads_used: IntProperty(
        name="Downloads Used",
        default=0
    )
    downloads_limit: IntProperty(
        name="Downloads Limit",
        default=0
    )
    uploads_used: IntProperty(
        name="Uploads Used",
        default=0
    )
    download_count: IntProperty(
        name="Download Count",
        description="Total number of downloads",
        default=0
    )
    
    
    def init_from_package(self, pkg: dict):
        """
        A helper to initialize these props from the server's package dict (fetched_packages_data).
        Example usage after a scene is appended or a user goes to detail mode.
        """
        self.is_from_webapp = True
        self.file_id = pkg.get("file_id", 0)
        self.package_name = pkg.get("package_name", "")
        self.author = pkg.get("uploader", "Unknown")
        self.likes = pkg.get("likes", 0)
        self.description = pkg.get("description", "No description")
        self.upload_date = pkg.get("upload_date", "N/A")
        # Build out the profile link:
        self.profile_url = f"{USER_PROFILE_BASE_URL}/{self.author}"
        self.download_count = pkg.get("download_count", 0)


# This property will hold the entire events data fetched from the backend.
bpy.types.Scene.fetched_events = bpy.props.PointerProperty(type=bpy.types.PropertyGroup)

# A helper function that returns the items for the event dropdown.
def get_event_items(self, context):
    events_data = context.scene.get("fetched_events_data", {})  # This will store our fetched events.
    stage = context.scene.event_stage  # Already defined as an EnumProperty (e.g., 'submit', 'vote', 'winners')
    items = []
    stage_events = events_data.get(stage, [])
    for event in stage_events:
        # Each item is a tuple: (value, label, description)
        items.append((str(event["id"]), event["title"], event.get("description", "")))
    if not items:
        items = [("0", "No events", "No active event in this stage")]
    return items

# Add a property for selecting an event.
bpy.types.Scene.selected_event = bpy.props.EnumProperty(
    name="Event",
    description="Select an event to filter packages",
    items=get_event_items
)
