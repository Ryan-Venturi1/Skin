import os
import random
import json
import tensorflow as tf
import subprocess
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

# Enhanced configuration with better settings
SEED = 42
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 64
EPOCHS = 10  # Increased epochs for better training
DATASET_DIR = 'isic_dataset'
VALIDATION_SPLIT = 0.15  # Reduced validation to have more training data
LEARNING_RATE = 1e-4
FINE_TUNE_LEARNING_RATE = 5e-5

# Set random seeds for reproducibility
random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

print("Setting up data directories...")

TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
VAL_DIR = os.path.join(DATASET_DIR, 'val')

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)
os.makedirs('model', exist_ok=True)

def split_data_into_train_val():
    """
    Creates train and validation directories with class subdirectories
    """
    print("Organizing data into train/validation splits...")
    
    # Get all diagnosis folders
    class_dirs = [d for d in os.listdir(DATASET_DIR) 
                  if os.path.isdir(os.path.join(DATASET_DIR, d)) 
                  and d not in ['train', 'val']]
    
    if not class_dirs:
        print("No class directories found! Please run dataset.py first.")
        return False
    
    # Track class counts for balancing
    class_counts = {}
    for class_dir in class_dirs:
        # Create class directory in train and val
        train_class_dir = os.path.join(TRAIN_DIR, class_dir)
        val_class_dir = os.path.join(VAL_DIR, class_dir)
        
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)
        
        # Get all image files
        source_dir = os.path.join(DATASET_DIR, class_dir)
        image_files = [f for f in os.listdir(source_dir) if f.endswith('.jpg')]
        
        class_counts[class_dir] = len(image_files)
        
        # Skip if no images
        if not image_files:
            print(f"Warning: No images found in {source_dir}")
            continue
            
        # Shuffle files
        random.shuffle(image_files)
        
        # Split into train and validation
        split_idx = int(len(image_files) * (1 - VALIDATION_SPLIT))
        train_files = image_files[:split_idx]
        val_files = image_files[split_idx:]
        
        print(f"{class_dir}: {len(train_files)} train, {len(val_files)} validation images")
        
        # Create symbolic links (or copy if symlinks not supported)
        for f in train_files:
            src = os.path.join(source_dir, f)
            dst = os.path.join(train_class_dir, f)
            if not os.path.exists(dst):
                try:
                    os.symlink(os.path.abspath(src), dst)
                except (OSError, NotImplementedError):
                    import shutil
                    shutil.copy2(src, dst)
        
        for f in val_files:
            src = os.path.join(source_dir, f)
            dst = os.path.join(val_class_dir, f)
            if not os.path.exists(dst):
                try:
                    os.symlink(os.path.abspath(src), dst)
                except (OSError, NotImplementedError):
                    import shutil
                    shutil.copy2(src, dst)
    
    # Print class distribution
    print("\nClass distribution:")
    for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cls}: {count} images")
    
    return True

# Enhanced data augmentation for better model generalization
def create_data_generators():
    """Create train and validation data generators with augmentation"""
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,  # Increased rotation
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.3,  # Increased zoom
        horizontal_flip=True,
        vertical_flip=True,
        shear_range=0.1,  # Added shear
        brightness_range=[0.8, 1.2],  # Added brightness variation
        fill_mode='nearest'
    )

    # Only rescaling for validation
    val_datagen = ImageDataGenerator(rescale=1./255)

    print("Loading training data...")
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    print("Loading validation data...")
    validation_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, validation_generator

def build_and_train_model(train_generator, validation_generator):
    """Build and train the model with a two-phase approach"""
    
    # Determine number of classes dynamically
    num_classes = len(train_generator.class_indices)
    print(f"Training with {num_classes} classes: {train_generator.class_indices}")
    
    # Save class names immediately
    class_indices = train_generator.class_indices
    class_names = {str(v): k for k, v in class_indices.items()}
    with open('model/class_names.json', 'w') as f:
        json.dump(class_names, f)
    
    print("Building EfficientNetB0 model...")
    # EXPLICITLY create input layer first - this is key for correct model.json format
    inputs = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3), name="input_layer")
    
    # Initialize the base model with pre-trained weights
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_tensor=inputs  # Use our explicit input tensor
    )
    
    # Freeze the base model initially
    base_model.trainable = False
    
    # Add custom classification layers
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)  # Increased dropout for better generalization
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Create the initial model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile the initial model
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Explicitly set input shape info
    input_shape = model.input_shape
    print(f"Model input shape: {input_shape}")
    
    # Callbacks for training
    callbacks = [
        ModelCheckpoint(
            'best_model_phase1.h5',
            save_best_only=True,
            monitor='val_accuracy',
            mode='max'
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6
        )
    ]
    
    # Phase 1: Train only the top layers
    print("Phase 1: Training top layers...")
    history_phase1 = model.fit(
        train_generator,
        epochs=10,  # Fewer epochs for initial phase
        validation_data=validation_generator,
        callbacks=callbacks
    )
    
    # Load the best weights from phase 1
    model.load_weights('best_model_phase1.h5')
    
    # Phase 2: Fine-tune the model by unfreezing some layers
    print("\nPhase 2: Fine-tuning the model...")
    
    # Unfreeze the top layers of the base model
    base_model.trainable = True
    
    # Freeze the bottom layers and unfreeze the top layers
    # Typically we unfreeze the last 20-30% of the layers
    for layer in base_model.layers[:-20]:  # Keep bottom layers frozen
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=FINE_TUNE_LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Updated callbacks for phase 2
    callbacks = [
        ModelCheckpoint(
            'best_model_phase2.h5',
            save_best_only=True,
            monitor='val_accuracy',
            mode='max'
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=8,  # More patience for fine-tuning
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=4,
            min_lr=1e-7
        )
    ]
    
    # Train with unfrozen layers
    history_phase2 = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=callbacks,
        initial_epoch=len(history_phase1.history['loss'])  # Continue from phase 1
    )
    
    # Load the best weights from phase 2
    model.load_weights('best_model_phase2.h5')
    
    return model

def save_model(model):
    """Save the model in multiple formats with explicit input shape handling"""
    
    print("Saving model in Keras format...")
    model.save('skin_model.keras')
    
    # Also save as HDF5 for backward compatibility
    print("Also saving model in HDF5 format...")
    model.save('skin_model.h5')
    
    # Manual creation of model.json with explicit input shape
    print("Creating explicit model.json with input shape...")
    model_json_str = model.to_json()
    model_dict = json.loads(model_json_str)
    
    # Ensure the first layer includes input shape info
    if "config" in model_dict and "layers" in model_dict["config"] and len(model_dict["config"]["layers"]) > 0:
        first_layer = model_dict["config"]["layers"][0]
        if first_layer["class_name"] == "InputLayer":
            print("Confirmed InputLayer exists with shape:", first_layer["config"].get("batch_input_shape"))
        else:
            print("Adding explicit InputLayer to model.json")
            input_layer = {
                "class_name": "InputLayer",
                "config": {
                    "batch_input_shape": [None, IMG_HEIGHT, IMG_WIDTH, 3],
                    "dtype": "float32",
                    "sparse": False,
                    "name": "input_layer"
                },
                "name": "input_layer",
                "inbound_nodes": []
            }
            model_dict["config"]["layers"].insert(0, input_layer)
    
    with open('model/model.json', 'w') as f:
        json.dump(model_dict, f)
    
    print("\nConverting model to TensorFlow.js format...")
    try:
        # Run conversion with essential flags
        result = subprocess.run(
            ["tensorflowjs_converter",
             "--input_format=keras",
             "--output_format=tfjs_layers_model",
             "skin_model.h5",
             "model/"],
            check=True, 
            capture_output=True, 
            text=True
        )
        print("Conversion successful!")
        
        # Open the converted model.json
        with open('model/model.json', 'r') as f:
            tfjs_model = json.load(f)
        
        # FIX: Update weightsManifest paths to include only file names (remove any directory info)
        if "weightsManifest" in tfjs_model:
            for manifest in tfjs_model["weightsManifest"]:
                manifest["paths"] = [os.path.basename(p) for p in manifest.get("paths", [])]
            print("Fixed weightsManifest paths to only include file names")
        
        # Fix input shape in the TFJS model if needed
        needs_fixing = False
        if 'modelTopology' in tfjs_model and 'config' in tfjs_model['modelTopology']:
            model_config = tfjs_model['modelTopology']['config']
            if 'layers' in model_config and len(model_config['layers']) > 0:
                first_layer = model_config['layers'][0]
                if first_layer['class_name'] != 'InputLayer':
                    print("Adding missing InputLayer to TFJS model.json")
                    input_layer = {
                        "class_name": "InputLayer",
                        "config": {
                            "batch_input_shape": [None, IMG_HEIGHT, IMG_WIDTH, 3],
                            "dtype": "float32",
                            "sparse": False,
                            "name": "input_layer"
                        },
                        "name": "input_layer",
                        "inbound_nodes": []
                    }
                    model_config['layers'].insert(0, input_layer)
                    needs_fixing = True
                elif 'config' in first_layer and 'batch_input_shape' not in first_layer['config']:
                    print("Adding missing batch_input_shape to InputLayer")
                    first_layer['config']['batch_input_shape'] = [None, IMG_HEIGHT, IMG_WIDTH, 3]
                    needs_fixing = True
            if needs_fixing:
                with open('model/model.json', 'w') as f:
                    json.dump(tfjs_model, f)
                print("Fixed TFJS model.json with proper input shape")
        
    except subprocess.CalledProcessError as e:
        print("Conversion failed with error:")
        print(e.stderr)
        # (Alternate conversion and manual fix code can go here if needed.)
    except FileNotFoundError:
        print("tensorflowjs_converter not found. Please install it with:")
        print("pip install tensorflowjs")
        # Fallback: Create a minimal model.json
        try:
            print("Creating minimal model.json for fallback...")
            model_dict = json.loads(model.to_json())
            tfjs_model = {
                "format": "layers-model",
                "generatedBy": "manual-fix",
                "convertedBy": "manual-fix",
                "modelTopology": model_dict,
                "weightsManifest": [
                    {
                        "paths": ["group1-shard1of1.bin"],
                        "weights": []
                    }
                ]
            }
            # Ensure the input layer is present and correct
            if 'config' in tfjs_model['modelTopology'] and 'layers' in tfjs_model['modelTopology']['config']:
                layers = tfjs_model['modelTopology']['config']['layers']
                if layers[0]['class_name'] == 'InputLayer':
                    layers[0]['config']['batch_input_shape'] = [None, IMG_HEIGHT, IMG_WIDTH, 3]
                else:
                    input_layer = {
                        "class_name": "InputLayer",
                        "config": {
                            "batch_input_shape": [None, IMG_HEIGHT, IMG_WIDTH, 3],
                            "dtype": "float32",
                            "sparse": False,
                            "name": "input_layer"
                        },
                        "name": "input_layer",
                        "inbound_nodes": []
                    }
                    layers.insert(0, input_layer)
            os.makedirs('model', exist_ok=True)
            with open('model/model.json', 'w') as f:
                json.dump(tfjs_model, f)
            print("Created minimal model.json for fallback")
        except Exception as e4:
            print(f"Failed to create minimal model.json: {str(e4)}")

if __name__ == "__main__":
    # First, ensure we have the train/val split
    if not os.listdir(TRAIN_DIR) or not os.listdir(VAL_DIR):
        success = split_data_into_train_val()
        if not success:
            print("Error creating train/val split. Exiting.")
            exit(1)
    else:
        print("Train/validation split already exists.")
    
    # Create data generators
    train_generator, validation_generator = create_data_generators()
    
    # Build and train the model
    model = build_and_train_model(train_generator, validation_generator)
    
    # Save the model (and convert to TFJS format)
    save_model(model)
    
    print("\nTraining and conversion complete!")