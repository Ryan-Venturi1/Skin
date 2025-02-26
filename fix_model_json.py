import json
import os
import sys

def fix_model_json(model_json_path):
    """
    Fix the model.json file to ensure it has proper input shape information
    """
    print(f"Attempting to fix model.json at: {model_json_path}")
    
    # Load the existing model.json file
    try:
        with open(model_json_path, 'r') as f:
            model_json = json.load(f)
            print("Model.json loaded successfully")
    except Exception as e:
        print(f"Error loading model.json: {str(e)}")
        return False
    
    # Check if it's a TFJS model format
    is_tfjs_format = 'modelTopology' in model_json
    needs_fix = False
    
    if is_tfjs_format:
        print("Detected TensorFlow.js layers model format")
        model_config = model_json.get('modelTopology', {}).get('config', {})
        layers = model_config.get('layers', [])
    else:
        print("Detected Keras model format")
        model_config = model_json.get('config', {})
        layers = model_config.get('layers', [])
    
    # Check if layers exist and fix if needed
    if not layers:
        print("No layers found in model.json")
        needs_fix = True
        
        # Create a minimal set of layers with proper input
        if is_tfjs_format:
            model_json['modelTopology']['config']['layers'] = []
            layers = model_json['modelTopology']['config']['layers']
        else:
            model_json['config']['layers'] = []
            layers = model_json['config']['layers']
    
    # Check if first layer is InputLayer with proper shape
    if layers and len(layers) > 0:
        first_layer = layers[0]
        
        if first_layer.get('class_name') != 'InputLayer':
            print("First layer is not an InputLayer. Adding InputLayer...")
            input_layer = {
                "class_name": "InputLayer",
                "config": {
                    "batch_input_shape": [None, 224, 224, 3],
                    "dtype": "float32",
                    "sparse": False,
                    "name": "input_1"
                },
                "name": "input_1",
                "inbound_nodes": []
            }
            layers.insert(0, input_layer)
            needs_fix = True
            
            # Update inbound_nodes for second layer if needed
            if len(layers) > 1 and 'inbound_nodes' in layers[1]:
                if not layers[1]['inbound_nodes']:
                    layers[1]['inbound_nodes'] = [[["input_1", 0, 0, {}]]]
                    print("Updated inbound_nodes for second layer")
        else:
            # Check if InputLayer has batch_input_shape
            if 'config' in first_layer and 'batch_input_shape' not in first_layer['config']:
                print("InputLayer is missing batch_input_shape. Adding it...")
                first_layer['config']['batch_input_shape'] = [None, 224, 224, 3]
                first_layer['config']['input_shape'] = [224, 224, 3]
                needs_fix = True
            elif 'config' in first_layer and first_layer['config'].get('batch_input_shape') is None:
                print("InputLayer has null batch_input_shape. Fixing it...")
                first_layer['config']['batch_input_shape'] = [None, 224, 224, 3]
                first_layer['config']['input_shape'] = [224, 224, 3]
                needs_fix = True
    
    # Save the fixed model.json if needed
    if needs_fix:
        # Backup the original file
        backup_path = model_json_path + '.backup'
        try:
            with open(backup_path, 'w') as f:
                json.dump(model_json, f)
            print(f"Original model.json backed up to {backup_path}")
        except Exception as e:
            print(f"Warning: Failed to create backup: {str(e)}")
        
        # Save the fixed model.json
        try:
            with open(model_json_path, 'w') as f:
                json.dump(model_json, f, indent=2)
            print(f"Fixed model.json saved to {model_json_path}")
            return True
        except Exception as e:
            print(f"Error saving fixed model.json: {str(e)}")
            return False
    else:
        print("No fixes needed for model.json")
        return True

if __name__ == "__main__":
    # Get model.json path from command line or use default
    model_json_path = sys.argv[1] if len(sys.argv) > 1 else 'model/model.json'
    
    if not os.path.exists(model_json_path):
        print(f"Error: {model_json_path} does not exist")
        sys.exit(1)
    
    success = fix_model_json(model_json_path)
    
    if success:
        print("Model.json fix completed successfully")
        sys.exit(0)
    else:
        print("Failed to fix model.json")
        sys.exit(1)