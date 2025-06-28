# version_info.py
import os
import ast

# locate the add-on’s __init__.py (one folder up from this file’s parent)
_root_dir  = os.path.dirname(os.path.dirname(__file__))
_init_path = os.path.join(_root_dir, "__init__.py")

# default version tuple
_version = (0, 0, 0)

with open(_init_path, "r", encoding="utf-8") as f:
    src = f.read()

module = ast.parse(src, _init_path)
for node in module.body:
    # find: bl_info = { ... }
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "bl_info":
                # scan its dict entries for "version"
                for key_node, val_node in zip(node.value.keys, node.value.values):
                    if isinstance(key_node, ast.Constant) and key_node.value == "version":
                        _version = ast.literal_eval(val_node)
                break
        break

# expose as "X.Y.Z"
CURRENT_VERSION = ".".join(str(n) for n in _version)