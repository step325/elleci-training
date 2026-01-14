
import zipfile
import os
import sys

def create_clean_zip(output_filename="Elleci_Kaggle_Clean.zip"):
    # List of files and directories to include
    include_paths = [
        "src",
        "data",
        "training",
        "inference_v2.py",
        "requirements.txt"
    ]

    print(f"Creating {output_filename}...")
    
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for item in include_paths:
            if os.path.isdir(item):
                for root, _, files in os.walk(item):
                    for file in files:
                        if "__pycache__" in root or file.endswith(".pyc"):
                            continue
                        
                        file_path = os.path.join(root, file)
                        # Create archive name with forward slashes
                        arcname = os.path.relpath(file_path, ".")
                        arcname = arcname.replace(os.sep, "/")
                        
                        print(f"Adding {file_path} as {arcname}")
                        zipf.write(file_path, arcname)
            else:
                if os.path.exists(item):
                    # Create archive name with forward slashes
                    arcname = item.replace(os.sep, "/")
                    print(f"Adding {item} as {arcname}")
                    zipf.write(item, arcname)
                else:
                    print(f"Warning: {item} not found")

    print(f"\nSuccess! Created {output_filename}")
    print("Upload this file to Kaggle Dataset (it uses forward slashes).")

if __name__ == "__main__":
    create_clean_zip()
