import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2 # Or EfficientNetB0, ResNet50
import matplotlib.pyplot as plt
import pathlib
import os

# --- 1. Define Parameters ---
IMG_SIZE = (224, 224) # Input size expected by the pre-trained model (check documentation!)
NUM_CLASSES = 3       # Number of shapes you want to classify (e.g., circle, square, triangle, etc.)
BATCH_SIZE = 32       # Training batch size
LEARNING_RATE = 0.001 # Initial learning rate for the new head
EPOCHS_HEAD = 10      # Number of epochs to train only the new head
EPOCHS_FINE_TUNE = 10 # Number of epochs for fine-tuning (optional)
FINE_TUNE_LR = 0.00001 # Very low learning rate for fine-tuning
AUTOTUNE = tf.data.AUTOTUNE
CATEGORIES = ['down', 'neutral', 'up']  # Define fixed order of categories
NUM_CATEGORIES = len(CATEGORIES)

# --- 2. Load the Pre-trained Base Model ---
# Load the model WITHOUT its classification head (include_top=False)
# Specify input_shape matching your image size (plus color channels)
# Use 'imagenet' weights
base_model = MobileNetV2(input_shape=IMG_SIZE + (3,),
                         include_top=False,
                         weights='imagenet')

# --- 3. Freeze the Base Model Layers ---
# Prevent the pre-trained weights from being updated initially
base_model.trainable = False

# --- 4. Add Your Custom Classification Head ---
# Start building on top of the base model's output
inputs = keras.Input(shape=IMG_SIZE + (3,))
model = base_model(inputs, training=False) # Pass inputs, ensure batch norm layers are in inference mode

# Add layers appropriate for classification
model = layers.GlobalAveragePooling2D()(model) # Average the spatial features
model = layers.Dropout(0.2)(model)             # Add dropout for regularization
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(model) # Final layer with units = num_shapes

# Create the new model
model = keras.Model(inputs, outputs)

# --- 5. Compile the Model ---
# Use an optimizer (Adam is common) and appropriate loss function
model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
              loss='categorical_crossentropy', # Use 'sparse_categorical_crossentropy' if labels are integers
              metrics=['acc'])

# model.summary() # Print model architecture

# --- 6. Prepare Your Data ---
# Load your shape images (e.g., using tf.keras.utils.image_dataset_from_directory)
# Ensure images are resized to IMG_SIZE
# IMPORTANT: Preprocess input images EXACTLY as the base model expects.
# Often involves scaling pixel values (e.g., to [-1, 1] for MobileNetV2 or [0, 1] then specific normalization for others)
# Check the documentation for tf.keras.applications.<model_name>.preprocess_input
# train_dataset = ... (load and preprocess training data)
# validation_dataset = ... (load and preprocess validation data)

# Example using image_dataset_from_directory (assumes preprocessing is handled separately or via a layer)
# train_dir = 'path/to/your/shape_data/train'
# val_dir = 'path/to/your/shape_data/validation'
# train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,
#                                                              shuffle=True,
#                                                              batch_size=BATCH_SIZE,
#                                                              image_size=IMG_SIZE)
# validation_dataset = tf.keras.utils.image_dataset_from_directory(val_dir,
#                                                                    shuffle=True,
#                                                                    batch_size=BATCH_SIZE,
#                                                                    image_size=IMG_SIZE)

# Add preprocessing as part of the dataset pipeline or model
# AUTOTUNE = tf.data.AUTOTUNE
# def preprocess_data(image, label):
#     image = tf.cast(image, tf.float32)
#     # Apply the specific preprocessing function for your chosen base model
#     image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
#     return image, label

# train_dataset = train_dataset.map(preprocess_data).cache().prefetch(buffer_size=AUTOTUNE)
# validation_dataset = validation_dataset.map(preprocess_data).cache().prefetch(buffer_size=AUTOTUNE)

# Load and preprocess images from a directory
def load_and_preprocess_image(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_png(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img

def process_path(file_path):
    # Extract category from path
    label = tf.strings.split(file_path, os.path.sep)[-2]
    # Convert category name to index using fixed CATEGORIES list
    label_id = tf.where(tf.constant(CATEGORIES) == label)[0]
    # Create one-hot vector with NUM_CATEGORIES
    label_onehot = tf.reshape(tf.one_hot(label_id, NUM_CATEGORIES), [NUM_CATEGORIES])
    img = load_and_preprocess_image(file_path)
    print(label_onehot)
    return img, label_onehot

def get_datasets(data_dir, validation_split=0.2):
    data_dir = pathlib.Path(data_dir)
    
    # Validate categories
    found_categories = [item.name for item in data_dir.glob('*') if item.is_dir()]
    if not all(cat in found_categories for cat in CATEGORIES):
        raise ValueError(f"Missing categories. Expected {CATEGORIES}, found {found_categories}")
    
    # Get all file paths and convert to strings
    all_files = [str(f) for f in data_dir.glob('*/*.png')]
    print(f"Found {len(all_files)} PNG files")
    print(f"Categories found: {found_categories}")
    
    # Shuffle all files
    all_files = tf.random.shuffle(all_files)
    
    # Calculate split
    val_size = int(len(all_files) * validation_split)
    train_files = all_files[val_size:]
    val_files = all_files[:val_size]
    
    print(f"Training images: {len(train_files)}")
    print(f"Validation images: {len(val_files)}")
    
    # Create datasets
    train_ds = tf.data.Dataset.from_tensor_slices(train_files)
    val_ds = tf.data.Dataset.from_tensor_slices(val_files)
    
    # Process both datasets
    train_ds = train_ds.shuffle(1000)
    train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.cache().batch(BATCH_SIZE).prefetch(AUTOTUNE)
    
    val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.cache().batch(BATCH_SIZE).prefetch(AUTOTUNE)
    
    return train_ds, val_ds

# Create datasets
print("Loading datasets from 'images' directory...")
train_dataset, validation_dataset = get_datasets('images')


# --- 7. Train the Head ---
print("Training the new classification head...")
with tf.device('/device:GPU:0'):
    history = model.fit(train_dataset,
                        epochs=EPOCHS_HEAD,
                        validation_data=validation_dataset)

# --- 8. (Optional but Recommended) Fine-tuning ---
print("Starting fine-tuning...")
# Unfreeze the base model (or parts of it)
base_model.trainable = True

# It's often better to only unfreeze the top layers/blocks of the base model first
# For example, freeze all layers up to a certain point:
fine_tune_at = 100 # Example layer index - choose carefully!
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Re-compile the model with a VERY LOW learning rate
model.compile(optimizer=keras.optimizers.Adam(learning_rate=FINE_TUNE_LR), # Much lower LR!
              loss='categorical_crossentropy',
              metrics=['acc'])

# model.summary() # Check which layers are trainable

# Continue training
with tf.device('/device:GPU:0'):
    history_fine = model.fit(train_dataset,
                            epochs=EPOCHS_HEAD + EPOCHS_FINE_TUNE, # Total epochs
                            initial_epoch=history.epoch[-1] + 1, # Start from where the head training left off
                            validation_data=validation_dataset)

# --- 9. Evaluate ---
loss, accuracy = model.evaluate(validation_dataset)
print(f"Final Validation Accuracy: {accuracy * 100:.2f}%")

## Head model accuracy and loss plots
# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))
plt.plot(history.history['acc'], label='Train Accuracy')
plt.plot(history.history['val_acc'], label='Validation Accuracy')
plt.title('Head model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

# Plot training & validation loss values
plt.figure(figsize=(12, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Head model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

## Fine-tuning model accuracy and loss plots
# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))
plt.plot(history_fine.history['acc'], label='Train Accuracy')
plt.plot(history_fine.history['val_acc'], label='Validation Accuracy')
plt.title('Fine tuned model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

# Plot training & validation loss values
plt.figure(figsize=(12, 4))
plt.plot(history_fine.history['loss'], label='Train Loss')
plt.plot(history_fine.history['val_loss'], label='Validation Loss')
plt.title('Fine tuned model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

plt.show()

