
import os

filepath = "/root/miniconda3/lib/python3.12/site-packages/esm/esmfold/v1/pretrained.py"
backup_path = filepath + ".bak"

if not os.path.exists(backup_path):
    import shutil
    shutil.copyfile(filepath, backup_path)
    print(f"Backed up {filepath} to {backup_path}")

with open(filepath, "r") as f:
    lines = f.readlines()

insertion_point = -1
for i, line in enumerate(lines):
    if 'model_state = model_data["model"]' in line:
        insertion_point = i + 1
        break

if insertion_point != -1:
    new_lines = [
        '\n',
        '    # Fix for OpenFold compatibility\n',
        '    for k in list(model_state.keys()):\n',
        '        if "linear_q_points.weight" in k:\n',
        '            model_state[k.replace("linear_q_points.weight", "linear_q_points.linear.weight")] = model_state.pop(k)\n',
        '        elif "linear_q_points.bias" in k:\n',
        '            model_state[k.replace("linear_q_points.bias", "linear_q_points.linear.bias")] = model_state.pop(k)\n',
        '        elif "linear_kv_points.weight" in k:\n',
        '            model_state[k.replace("linear_kv_points.weight", "linear_kv_points.linear.weight")] = model_state.pop(k)\n',
        '        elif "linear_kv_points.bias" in k:\n',
        '            model_state[k.replace("linear_kv_points.bias", "linear_kv_points.linear.bias")] = model_state.pop(k)\n'
    ]
    
    # Check if already patched to avoid duplication
    if "Fix for OpenFold compatibility" not in "".join(lines):
        lines[insertion_point:insertion_point] = new_lines
        with open(filepath, "w") as f:
            f.writelines(lines)
        print("Successfully patched pretrained.py")
    else:
        print("File already patched.")
else:
    print("Could not find insertion point.")
