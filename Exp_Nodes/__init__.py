from .node_editor import register as register_nodes, unregister as unregister_nodes

def register():
    register_nodes()

def unregister():
    unregister_nodes()

if __name__ == "__main__":
    register()
