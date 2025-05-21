import os
import sys
import shutil

def delete_transformation(name):
    base_path = os.path.join("Transformations", name)

    if not os.path.exists(base_path):
        print(f"Transformation '{name}' does not exist.")
        return

    confirm = input(f"Are you sure you want to delete '{name}'? This cannot be undone. (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled.")
        return

    shutil.rmtree(base_path)
    print(f"Deleted transformation: {base_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python delete_transformation.py <TransformationName>")
        sys.exit(1)

    transformation_name = sys.argv[1]
    delete_transformation(transformation_name)
