import os
import sys

def create_transformation(name):
    base_path = os.path.join("Transformations", name)

    if os.path.exists(base_path):
        print(f"Transformation '{name}' already exists.")
        return

    os.makedirs(base_path)

    file_name = name[0].lower() + name[1:] + ".py"
    file_path = os.path.join(base_path, file_name)
    function_name = name[0].lower() + name[1:]

    with open(file_path, "w") as f:
        f.write("import numpy as np\n\n")
        f.write(f"def {function_name}():\n")
        f.write("    # TODO: implement transformation logic\n")
        f.write("    return None\n")

    print(f"Created transformation: {file_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python create_transformation.py <TransformationName>")
        sys.exit(1)

    transformation_name = sys.argv[1]
    create_transformation(transformation_name)
