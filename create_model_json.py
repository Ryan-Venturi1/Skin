import json
import os

# Create a minimal but valid model.json structure
model_json = {
    "format": "layers-model",
    "generatedBy": "manual-fix",
    "convertedBy": "manual-fix",
    "modelTopology": {
        "class_name": "Model",
        "config": {
            "name": "model",
            "layers": [
                {
                    "class_name": "InputLayer",
                    "config": {
                        "batch_input_shape": [None, 224, 224, 3],  # None in Python becomes null in JSON
                        "dtype": "float32",
                        "sparse": False,
                        "name": "input_1"
                    },
                    "name": "input_1",
                    "inbound_nodes": []
                }
                # Other layers will be preserved from existing file
            ]
        }
    },
    "weightsManifest": [
        {
            "paths": ["group1-shard1of1.bin"],
            "weights": []
        }
    ]
}

# Try to preserve weights info from existing model.json
try:
    with open('model/model.json', 'r') as f:
        existing_model = json.load(f)
    
    if 'weightsManifest' in existing_model:
        model_json['weightsManifest'] = existing_model['weightsManifest']
    
    # Try to get other layers if they exist
    try:
        if 'modelTopology' in existing_model and 'config' in existing_model['modelTopology']:
            existing_layers = existing_model['modelTopology']['config']['layers']
            if existing_layers and len(existing_layers) > 0:
                # Skip first layer if it's InputLayer
                if existing_layers[0]['class_name'] == 'InputLayer':
                    model_json['modelTopology']['config']['layers'].extend(existing_layers[1:])
                else:
                    model_json['modelTopology']['config']['layers'].extend(existing_layers)
    except Exception as e:
        print(f"Could not preserve existing layers: {str(e)}")
        print("But weights manifest should still work")
except Exception as e:
    print(f"Could not open existing model.json: {str(e)}")
    print("Creating minimal version")

# Backup original model.json if it exists
if os.path.exists('model/model.json'):
    try:
        os.rename('model/model.json', 'model/model.json.backup')
        print("Backed up original model.json to model.json.backup")
    except Exception as e:
        print(f"Could not backup original model.json: {str(e)}")

# Write new model.json
with open('model/model.json', 'w') as f:
    json.dump(model_json, f, indent=2)

print("Created new model.json with proper input shape")
print("You may need to restart your web server for changes to take effect")