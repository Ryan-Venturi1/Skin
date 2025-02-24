import os
import random
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Configuration - easily adjustable parameters
SEED = 42
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 16
EPOCHS = 10
DATASET_DIR = 'isic_dataset'  # Directory created by the API downloader script
VALIDATION_SPLIT = 0.2  # 20% for validation

# Set random seeds for reproducibility
random.seed(SEED)
tf.random.set_seed(SEED)

print("Setting up data directories...")

# Directory to store train/validation splits
TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
VAL_DIR = os.path.join(DATASET_DIR, 'val')

# Create directories if they don't exist
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

# Function to split data into train/validation sets
def split_data_into_train_val():
    """
    Creates train and validation directories with class subdirectories
    """
    print("Organizing data into train/validation splits...")
    
    # Get all diagnosis folders
    class_dirs = [d for d in os.listdir(DATASET_DIR) 
                 if os.path.isdir(os.path.join(DATASET_DIR, d)) 
                 and d not in ['train', 'val']]
    
    for class_dir in class_dirs:
        # Create class directory in train and val
        train_class_dir = os.path.join(TRAIN_DIR, class_dir)
        val_class_dir = os.path.join(VAL_DIR, class_dir)
        
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)
        
        # Get all image files
        source_dir = os.path.join(DATASET_DIR, class_dir)
        image_files = [f for f in os.listdir(source_dir) if f.endswith('.jpg')]
        
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

# Check if train/val split already exists
if not os.listdir(TRAIN_DIR) or not os.listdir(VAL_DIR):
    split_data_into_train_val()
else:
    print("Train/validation split already exists.")

# Set up data generators with augmentation
print("Setting up data generators...")
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,  # Skin lesions can appear in any orientation
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

# Determine number of classes dynamically
num_classes = len(train_generator.class_indices)
print(f"Training with {num_classes} classes: {train_generator.class_indices}")

# Build model
print("Building MobileNetV2 model...")
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
)

# Freeze the base model
base_model.trainable = False

# Add custom classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Create callbacks
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    'best_model.h5',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=3,
    restore_best_weights=True
)

# Train the model
print(f"Starting training for {EPOCHS} epochs...")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=[checkpoint_callback, early_stopping]
)

# Save the trained model
print("Saving model...")
model.save('skin_analysis_model.h5')

# Print instructions for TensorFlow.js conversion
print("\nTo convert the model to TensorFlow.js format, run:")
print("tensorflowjs_converter --input_format=keras skin_analysis_model.h5 model/")
print("\nTraining complete!")