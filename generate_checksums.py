import hashlib
from pathlib import Path

def generate_sha256(file_path):
    """Calculates the SHA-256 hash of a file in chunks."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def should_skip(path):
    """Returns True if any part of the path starts with a dot."""
    return any(part.startswith('.') for part in path.parts)

def process_path(target, f_out):
    """Processes a single file or directory, skipping hidden files."""
    path = Path(target)
    
    # Skip if the target itself is hidden
    if should_skip(path):
        return

    if path.is_file():
        file_hash = generate_sha256(path)
        f_out.write(f"{file_hash}  {path}\n")
        print(f"Hashed file: {path}")
        
    elif path.is_dir():
        # rglob("*") finds all files; we filter them in the loop
        for sub_path in sorted(path.rglob("*")):
            # Skip hidden files and files inside hidden directories
            if sub_path.is_file() and not should_skip(sub_path):
                file_hash = generate_sha256(sub_path)
                f_out.write(f"{file_hash}  {sub_path}\n")
                print(f"Hashed: {sub_path}")

def main(path_list, output_file):
    with open(output_file, "w") as f_out:
        for target in path_list:
            process_path(target, f_out)

if __name__ == "__main__":
    # Add as many paths as you need here
    PATHS_TO_HASH = [
	"CHANGELOG",
	"data/insulin4rl",
        "LICENSE",
        "notebook_utils",
        "pixi.toml",
	"tutorial_notebook.ipynb"
    ]
    OUTPUT_FILENAME = "SHA256SUMS"
    
    main(PATHS_TO_HASH, OUTPUT_FILENAME)
