import os

# Path to your package folder
package_path = 'C:/Users/spenc/Desktop/Exploratory/addons/Exploratory/Exp_Game'

# Desired output path and file name
output_path = 'C:/Users/spenc/Desktop/exploratory_combine_game'
output_file = 'combined_package.txt'

# Ensure the output directory exists
os.makedirs(output_path, exist_ok=True)

# Full path to the output file
output_file_path = os.path.join(output_path, output_file)

# Configuration for filtering
ignored_files = {'ui_backup.py','combine.py', '.gitignore',
                'exp_custom_animations.py', 'exp_objectives.py',
                'exp_mobility_and_game_reactions.py', 'exp_reactions.py',
                'exp_view.py' 'exp_interactions.py', 'exp_game_reset.py',
                'exp_movement.py', 'exp_physics.py', 'exp_animations.py', 'exp_custom_ui.py',
                 
                 }   # Files to exclude

ignored_folders = {'__pycache__', '.git', 'exp_assets', 'Skins', 'Default_Armature', 'Exp_UI'}# Folders to exclude entirely (including .git)

def should_include_folder(folder_name: str) -> bool:
    """Exclude known ignored foTlders (.git, __pycache__)"""
    return folder_name not in ignored_folders

def should_include_file(file_name: str, file_path: str) -> bool:
    """
    Include a file only if:
    1) It ends with .py
    2) It's not in ignored_filesT
    3) It's not empty
    """
    # Skip if not a .py file or is explicitly ignored
    if not file_name.endswith('.py') or file_name in ignored_files:
        return False

    # Also skip if the file is empty (0 bytes)
    if os.path.getsize(file_path) == 0:
        return False

    return True

with open(output_file_path, 'w', encoding='utf-8') as outfile:
    # Walk through all files and folders in the package
    for root, dirs, files in os.walk(package_path):
        # Filter out unwanted directories (e.g. .git, __pycache__)
        dirs[:] = [d for d in dirs if should_include_folder(d)]

        # Determine relative path of the current folder (root)
        folder_path = os.path.relpath(root, package_path)

        # Write a folder header only if there will be something to show
        # We’ll check if any .py files remain after filtering.
        py_files = [
            f for f in files
            if should_include_file(f, os.path.join(root, f))
        ]
        # If no .py files remain in this folder, skip writing the header
        if not py_files:
            continue

        outfile.write(f"# Folder: {folder_path}\n")
        outfile.write("#" * 80 + "\n\n")

        for file_name in py_files:
            file_path = os.path.join(root, file_name)

            outfile.write(f"# File: {file_name}\n")
            outfile.write("#" * 40 + "\n")

            try:
                with open(file_path, 'r', encoding='utf-8') as infile:
                    outfile.write(infile.read())
                    outfile.write("\n\n")  # Add space between files
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

print(f"All files have been combined into {output_file_path}")
