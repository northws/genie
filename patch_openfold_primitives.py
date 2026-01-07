
import os

filepath = "/root/miniconda3/lib/python3.12/site-packages/openfold/model/primitives.py"
backup_path = filepath + ".bak"

if not os.path.exists(backup_path):
    import shutil
    shutil.copyfile(filepath, backup_path)
    print(f"Backed up {filepath} to {backup_path}")

with open(filepath, "r") as f:
    lines = f.readlines()

line_idx = 23 # Line 24 in 1-based indexing
target_line = lines[line_idx]

if 'deepspeed_is_installed = importlib.util.find_spec("deepspeed") is not None' in target_line:
    lines[line_idx] = 'deepspeed_is_installed = False # Disabled via patch\n'
    with open(filepath, "w") as f:
        f.writelines(lines)
    print("Successfully patched primitives.py")
else:
    print(f"Line 24 mismatch: {repr(target_line)}")
