# project_structure.py
import os

IGNORE_DIRS = {"__pycache__"}  # folders to ignore even if not hidden

def print_project_structure(root_dir: str, prefix: str = ""):
    """
    Recursively prints folder structure with files,
    ignoring hidden files/folders and specified ignored directories.
    """
    entries = sorted(os.listdir(root_dir))
    # Filter out hidden files/folders and ignored directories
    entries = [
        e for e in entries
        if not e.startswith(".") and e not in IGNORE_DIRS
    ]
    entries_count = len(entries)

    for i, entry in enumerate(entries):
        path = os.path.join(root_dir, entry)
        connector = "└── " if i == entries_count - 1 else "├── "
        print(prefix + connector + entry)

        if os.path.isdir(path):
            extension = "    " if i == entries_count - 1 else "│   "
            print_project_structure(path, prefix + extension)


def save_structure_to_file(root_dir: str, output_file: str):
    """Save the folder structure to a text file."""
    with open(output_file, "w", encoding="utf-8") as f:
        def write_structure(dir_path, prefix=""):
            entries = sorted(os.listdir(dir_path))
            entries = [
                e for e in entries
                if not e.startswith(".") and e not in IGNORE_DIRS
            ]
            entries_count = len(entries)
            for i, entry in enumerate(entries):
                path = os.path.join(dir_path, entry)
                connector = "└── " if i == entries_count - 1 else "├── "
                f.write(prefix + connector + entry + "\n")
                if os.path.isdir(path):
                    extension = "    " if i == entries_count - 1 else "│   "
                    write_structure(path, prefix + extension)
        write_structure(root_dir)


if __name__ == "__main__":
    project_root = os.getcwd()
    print(f"Project Structure for: {project_root}\n")
    print_project_structure(project_root)

    save_file = os.path.join(project_root, "project_structure.txt")
    save_structure_to_file(project_root, save_file)
    print(f"\n✅ Project structure saved to: {save_file}")
