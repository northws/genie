
import os

filepath = "/root/autodl-tmp/genie/evaluations/pipeline/pipeline.py"
backup_path = filepath + ".bak"

if not os.path.exists(backup_path):
    import shutil
    shutil.copyfile(filepath, backup_path)
    print(f"Backed up {filepath} to {backup_path}")

with open(filepath, "r") as f:
    lines = f.readlines()

# Line 43 is index 42
line_idx = 42
target_line = lines[line_idx]
print(f"Original line: {repr(target_line)}")

if "assert os.path.exists(coords_dir)" in target_line:
    new_lines = [
        '\t\tif not os.path.exists(coords_dir):\n',
        '\t\t\tcoords_dir = input_dir\n'
    ]
    lines[line_idx] = "".join(new_lines) # Replace the line with multiple lines (as a single string or insert)
    # Actually assignment to list index replaces one item. I want to insert potentially.
    # But here I replace one line "assert ..." with "if ... \n ...".
    # So I can just assign a single string that contains newline.
    
    # Wait, better to use list slicing to replace and insert
    lines[line_idx:line_idx+1] = new_lines

    with open(filepath, "w") as f:
        f.writelines(lines)
    print("Successfully patched pipeline.py")
else:
    print("Line 43 did not match expected content.")
