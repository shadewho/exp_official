import os
# List of specific folders to loop through
folders_to_loop = ["Exp_Nodes"]

# Base directory (root directory where the addon and script are located)
base_directory = os.path.dirname(__file__)

# Output file path
output_file_path = r"C:\Users\spenc\Desktop\exploratory_Nodes\combined.txt"

def combine_python_files():
    """
    Combine all .py files in the addon root directory and specified folders into one .txt file.
    Each file is clearly labeled with its name and folder (if applicable).
    """
    try:
        # Create the output directory if it does not exist
        output_dir = os.path.dirname(output_file_path)
        os.makedirs(output_dir, exist_ok=True)

        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            # Process files in the root directory
            output_file.write("### Files in Root Directory ###\n")
            for file_name in os.listdir(base_directory):
                if file_name.endswith('.py') and file_name != "combine.py":
                    file_path = os.path.join(base_directory, file_name)
                    output_file.write(f"### {file_name} ###\n")
                    with open(file_path, 'r', encoding='utf-8') as input_file:
                        content = input_file.read()
                    output_file.write(content)
                    output_file.write("\n\n")  # Add spacing between files

            # Process files in the specified folders
            for folder_name in folders_to_loop:
                folder_path = os.path.join(base_directory, folder_name)

                if not os.path.exists(folder_path):
                    print(f"Folder does not exist: {folder_path}")
                    continue

                output_file.write(f"### Files in Folder: {folder_name} ###\n")
                for file_name in os.listdir(folder_path):
                    if file_name.endswith('.py'):
                        file_path = os.path.join(folder_path, file_name)
                        output_file.write(f"### {folder_name}/{file_name} ###\n")
                        with open(file_path, 'r', encoding='utf-8') as input_file:
                            content = input_file.read()
                        output_file.write(content)
                        output_file.write("\n\n")  # Add spacing between files

        print(f"All Python files have been combined into: {output_file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    combine_python_files()
