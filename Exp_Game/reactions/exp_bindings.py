# Exp_Game/reactions/exp_bindings.py
"""
Reaction Binding System - Connect data nodes to reaction parameters.

Architecture:
    1. DESIGN TIME: User connects data nodes to reaction input sockets
    2. GAME START: serialize_reaction_bindings() scans connections, bakes values
    3. RUNTIME: resolve_*() functions return bound values (no node access!)

This allows reaction parameters (location, rotation, duration, etc.) to receive
values from connected data nodes while respecting the "never read nodes at runtime" rule.

Binding Types:
    - STATIC: Value baked at game start from data node (Float, Vector, etc.)
    - TRACKER: Dynamic value resolved from tracker at execution time (future)

Performance:
    - Serialization: Once at game start (~1ms for 100 reactions)
    - Resolution: Dict lookup per parameter (~1µs each)
    - No node graph traversal at runtime
"""

import bpy
from mathutils import Vector, Euler
from ..developer.dev_logger import log_game

EXPL_TREE_ID = "ExploratoryNodesTreeType"

# ══════════════════════════════════════════════════════════════════════════════
# BINDING CACHE
# ══════════════════════════════════════════════════════════════════════════════
# Structure: {reaction_index: {param_name: {"type": str, "data_type": str, "value": any}}}
# Example: {0: {"transform_location": {"type": "STATIC", "data_type": "VECTOR", "value": (1.8, 1.8, 1.8)}}}

_reaction_bindings = {}


def reset_bindings():
    """Clear all bindings. Called on game start/reset."""
    global _reaction_bindings
    _reaction_bindings.clear()
    log_game("BINDINGS", "RESET cleared all bindings")


# ══════════════════════════════════════════════════════════════════════════════
# SERIALIZATION (Game Start)
# ══════════════════════════════════════════════════════════════════════════════

def serialize_reaction_bindings(scene) -> int:
    """
    Scan all reaction nodes and serialize their input socket connections.
    Called once at game start. Bakes connected data node values into bindings.

    Returns number of bindings created.
    """
    global _reaction_bindings
    _reaction_bindings.clear()

    binding_count = 0

    for ng in bpy.data.node_groups:
        if getattr(ng, 'bl_idname', '') != EXPL_TREE_ID:
            continue

        # Only process trees for this scene
        tree_scene = getattr(ng, 'scene', None)
        if tree_scene and tree_scene != scene:
            continue

        for node in ng.nodes:
            # Check if this is a reaction node (has reaction_index)
            reaction_idx = getattr(node, 'reaction_index', -1)
            if reaction_idx < 0:
                continue

            if reaction_idx not in _reaction_bindings:
                _reaction_bindings[reaction_idx] = {}

            # Check each input socket for connections
            for socket in node.inputs:
                if not socket.is_linked:
                    continue

                # Get the parameter name from socket's reaction_prop
                param_name = getattr(socket, 'reaction_prop', '')
                if not param_name:
                    # Fall back to socket name (lowercase, underscored)
                    param_name = socket.name.lower().replace(' ', '_')

                if not param_name:
                    continue

                # Trace back to source node
                src_node = socket.links[0].from_node
                binding = _serialize_source_node(src_node)

                if binding:
                    _reaction_bindings[reaction_idx][param_name] = binding
                    binding_count += 1
                    log_game("BINDINGS", f"BIND r={reaction_idx} '{param_name}' <- {binding['type']}:{binding['data_type']}")

    if binding_count > 0:
        log_game("BINDINGS", f"SERIALIZED {binding_count} bindings for {len(_reaction_bindings)} reactions")
    else:
        log_game("BINDINGS", "SERIALIZED 0 bindings (no connected data nodes)")

    return binding_count


def _serialize_source_node(node) -> dict:
    """
    Serialize a source node into a binding.
    Returns dict with type, data_type, and value - or None if not bindable.

    Supports:
        - FloatVectorDataNodeType → VECTOR
        - FloatDataNodeType → FLOAT
        - IntDataNodeType → INT
        - BoolDataNodeType → BOOL
        - ObjectDataNodeType → OBJECT (stores name or __CHARACTER__ marker)
    """
    node_type = getattr(node, 'bl_idname', '')

    # Float Vector Data Node
    if node_type == 'FloatVectorDataNodeType':
        return {
            "type": "STATIC",
            "data_type": "VECTOR",
            "value": tuple(node.value),
        }

    # Float Data Node
    elif node_type == 'FloatDataNodeType':
        return {
            "type": "STATIC",
            "data_type": "FLOAT",
            "value": float(node.value),
        }

    # Int Data Node
    elif node_type == 'IntDataNodeType':
        return {
            "type": "STATIC",
            "data_type": "INT",
            "value": int(node.value),
        }

    # Bool Data Node
    elif node_type == 'BoolDataNodeType':
        return {
            "type": "STATIC",
            "data_type": "BOOL",
            "value": bool(node.value),
        }

    # Object Data Node - store object name for runtime lookup
    elif node_type == 'ObjectDataNodeType':
        if getattr(node, 'use_character', False):
            return {
                "type": "STATIC",
                "data_type": "OBJECT",
                "value": "__CHARACTER__",  # Special marker for scene.target_armature
            }
        elif node.target_object:
            return {
                "type": "STATIC",
                "data_type": "OBJECT",
                "value": node.target_object.name,
            }

    # Future: Tracker nodes for dynamic bindings
    # elif node_type in ('DistanceTrackerNodeType', 'StateTrackerNodeType', ...):
    #     return {
    #         "type": "TRACKER",
    #         "data_type": "BOOL",  # or FLOAT depending on tracker
    #         "tracker_name": node.name,
    #     }

    return None


# ══════════════════════════════════════════════════════════════════════════════
# RESOLUTION (Runtime)
# ══════════════════════════════════════════════════════════════════════════════

def _find_reaction_index(reaction) -> int:
    """
    Find the index of a reaction in scene.reactions.
    Returns -1 if not found.
    """
    reactions = getattr(bpy.context.scene, "reactions", [])
    for idx, r in enumerate(reactions):
        if r == reaction:
            return idx
    return -1


def _get_reaction_idx(reaction_or_idx) -> int:
    """
    Convert reaction object or index to index.
    Accepts: int (returns as-is) or reaction object (finds index).
    """
    if isinstance(reaction_or_idx, int):
        return reaction_or_idx
    return _find_reaction_index(reaction_or_idx)


def has_binding(reaction_or_idx, param_name: str) -> bool:
    """Check if a parameter has a binding."""
    reaction_idx = _get_reaction_idx(reaction_or_idx)
    bindings = _reaction_bindings.get(reaction_idx, {})
    return param_name in bindings


def resolve_raw_binding(reaction_or_idx, param_name: str) -> str | None:
    """
    Return the raw binding value string without resolving markers like __CHARACTER__.
    Useful for checking whether a binding targets the character or a specific object.

    Returns the raw string value, or None if no binding exists.
    """
    reaction_idx = _get_reaction_idx(reaction_or_idx)
    bindings = _reaction_bindings.get(reaction_idx, {})
    binding = bindings.get(param_name)

    if binding and binding["type"] == "STATIC":
        return str(binding["value"])

    return None


def resolve_vector(reaction_or_idx, param_name: str, default) -> Vector:
    """
    Resolve a vector parameter for a reaction.
    Returns bound value if exists, otherwise converts default to Vector.

    Args:
        reaction_or_idx: Reaction object or index in scene.reactions
        param_name: Parameter name (e.g., "transform_location")
        default: Default value (tuple, list, or Vector)

    Returns:
        Vector with resolved or default value
    """
    reaction_idx = _get_reaction_idx(reaction_or_idx)
    bindings = _reaction_bindings.get(reaction_idx, {})
    binding = bindings.get(param_name)

    if binding and binding["type"] == "STATIC" and binding["data_type"] == "VECTOR":
        return Vector(binding["value"])

    # Return default as Vector
    if isinstance(default, Vector):
        return default.copy()
    return Vector(default)


def resolve_float(reaction_or_idx, param_name: str, default: float) -> float:
    """
    Resolve a float parameter for a reaction.
    Returns bound value if exists, otherwise default.

    Args:
        reaction_or_idx: Reaction object or index in scene.reactions
    """
    reaction_idx = _get_reaction_idx(reaction_or_idx)
    bindings = _reaction_bindings.get(reaction_idx, {})
    binding = bindings.get(param_name)

    if binding and binding["type"] == "STATIC" and binding["data_type"] == "FLOAT":
        return float(binding["value"])

    return float(default)


def resolve_int(reaction_or_idx, param_name: str, default: int) -> int:
    """
    Resolve an int parameter for a reaction.
    Returns bound value if exists, otherwise default.

    Args:
        reaction_or_idx: Reaction object or index in scene.reactions
    """
    reaction_idx = _get_reaction_idx(reaction_or_idx)
    bindings = _reaction_bindings.get(reaction_idx, {})
    binding = bindings.get(param_name)

    if binding and binding["type"] == "STATIC" and binding["data_type"] == "INT":
        return int(binding["value"])

    return int(default)


def resolve_bool(reaction_or_idx, param_name: str, default: bool) -> bool:
    """
    Resolve a bool parameter for a reaction.
    Returns bound value if exists, otherwise default.

    Args:
        reaction_or_idx: Reaction object or index in scene.reactions
    """
    reaction_idx = _get_reaction_idx(reaction_or_idx)
    bindings = _reaction_bindings.get(reaction_idx, {})
    binding = bindings.get(param_name)

    if binding and binding["type"] == "STATIC" and binding["data_type"] == "BOOL":
        return bool(binding["value"])

    return bool(default)


def resolve_object(reaction_or_idx, param_name: str, default_obj):
    """
    Resolve an object parameter for a reaction.
    Returns bound object if exists, otherwise default_obj.

    Handles __CHARACTER__ marker by returning scene.target_armature.

    Args:
        reaction_or_idx: Reaction object or index in scene.reactions
    """
    reaction_idx = _get_reaction_idx(reaction_or_idx)
    bindings = _reaction_bindings.get(reaction_idx, {})
    binding = bindings.get(param_name)

    if binding and binding["type"] == "STATIC" and binding["data_type"] == "OBJECT":
        obj_name = binding["value"]
        if obj_name == "__CHARACTER__":
            return bpy.context.scene.target_armature
        return bpy.data.objects.get(obj_name)

    return default_obj


def resolve_euler(reaction_or_idx, param_name: str, default) -> Euler:
    """
    Resolve a rotation parameter (stored as vector, returned as Euler).
    Returns bound value as Euler if exists, otherwise converts default.

    Args:
        reaction_or_idx: Reaction object or index in scene.reactions
    """
    reaction_idx = _get_reaction_idx(reaction_or_idx)
    bindings = _reaction_bindings.get(reaction_idx, {})
    binding = bindings.get(param_name)

    if binding and binding["type"] == "STATIC" and binding["data_type"] == "VECTOR":
        return Euler(binding["value"], 'XYZ')

    # Return default as Euler
    if isinstance(default, Euler):
        return default.copy()
    return Euler(default, 'XYZ')


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY
# ══════════════════════════════════════════════════════════════════════════════

def get_binding_count() -> int:
    """Get total number of bindings across all reactions."""
    return sum(len(bindings) for bindings in _reaction_bindings.values())


def get_bindings_for_reaction(reaction_idx: int) -> dict:
    """Get all bindings for a specific reaction (for debugging)."""
    return _reaction_bindings.get(reaction_idx, {}).copy()
