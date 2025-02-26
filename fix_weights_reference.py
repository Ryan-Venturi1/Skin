import json
import os
import glob

# Check which weight files actually exist
model_dir = 'model'
weight_files = glob.glob(os.path.join(model_dir, '*.bin'))
print(f"Found weight files: {weight_files}")

# Detect pattern - do we have 3 shards or 5?
shard3_pattern = [f for f in weight_files if 'shard1of3' in f]
shard5_pattern = [f for f in weight_files if 'shard1of5' in f]

# Determine which pattern to use
use_pattern = 'of3' if shard3_pattern else 'of5'
print(f"Using pattern: {use_pattern}")

# Load model.json
with open(os.path.join(model_dir, 'model.json'), 'r') as f:
    model_json = json.load(f)

# Update weightsManifest to use the correct pattern
if 'weightsManifest' in model_json:
    shards = []
    
    if use_pattern == 'of3':
        shards = ['group1-shard1of3.bin', 'group1-shard2of3.bin', 'group1-shard3of3.bin']
    else:
        shards = ['group1-shard1of5.bin', 'group1-shard2of5.bin', 'group1-shard3of5.bin', 
                  'group1-shard4of5.bin', 'group1-shard5of5.bin']
    
    # Update paths
    model_json['weightsManifest'][0]['paths'] = shards
    print(f"Updated weightsManifest to use: {shards}")

    # Save updated model.json
    with open(os.path.join(model_dir, 'model.json'), 'w') as f:
        json.dump(model_json, f, indent=2)
    print("Updated model.json saved")
else:
    print("ERROR: No weightsManifest found in model.json")