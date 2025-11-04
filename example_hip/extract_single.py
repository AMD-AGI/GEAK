import json
import os

# Input JSON file
input_file = "FromRe_instructions_FULL.json"

# Output folder
output_dir = "FromRe_instructions"
os.makedirs(output_dir, exist_ok=True)

# Read the JSON array
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Ensure it's a list
if not isinstance(data, list):
    raise ValueError("Input JSON must be an array of objects.")

# Write each object inside its own list
for i, obj in enumerate(data, start=1):
    output_path = output_dir + f"_{i}.json"
    with open(output_path, "w", encoding="utf-8") as out_f:
        json.dump([obj], out_f, ensure_ascii=False, indent=4)

print(f"Split into {len(data)} files in '{output_dir}' folder.")
