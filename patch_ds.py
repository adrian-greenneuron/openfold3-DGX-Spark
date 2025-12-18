import os

target = "/usr/local/lib/python3.12/dist-packages/deepspeed/ops/op_builder/builder.py"
print(f"Patching {target}...")

if not os.path.exists(target):
    print(f"Error: {target} not found!")
    exit(1)

content = open(target).read()

# The logic to replace the architecture detection
# We want to replace 'num = cc[0] + cc[2]' with a version that handles '121' -> '120'
# and cleans up extra chars.

old_code = 'num = cc[0] + cc[2]'
patch_code = 'num = cc.replace("+PTX","").replace(".","");\n            if num == "121": num="120"'

if old_code not in content:
    print(f"Warning: '{old_code}' not found in builder.py. File content snippet:\n{content[:500]}...")
    # Depending on DeepSpeed version, exact string might differ.
    # But for 0.15.4 strict match should work.
    exit(1)

new_content = content.replace(old_code, patch_code)
with open(target, "w") as f:
    f.write(new_content)

print("SUCCESS: Patched DeepSpeed builder.py for Blackwell compatibility")
