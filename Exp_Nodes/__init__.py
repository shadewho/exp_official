# Exp_Nodes/__init__.py
from .node_editor import register as node_editor_register, unregister as node_editor_unregister

def register():
    node_editor_register()

def unregister():
    node_editor_unregister()

if __name__ == "__main__":
    register()
