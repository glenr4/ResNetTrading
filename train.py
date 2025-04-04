import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from keras import models, layers, optimizers
import matplotlib.pyplot as plt
import os
import pathlib

AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 20
IMG_SIZE = 224
EPOCHS = 15

CATEGORIES = ['down', 'neutral', 'up']  # Define fixed order of categories
NUM_CATEGORIES = len(CATEGORIES)

def load_and_preprocess_image(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_png(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0
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
train_ds, val_ds = get_datasets('images')

# Load the model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Fine tuning of the model
base_model.trainable = True
fine_tune_at = 143 #  the beginning of the conv5 stage
# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable = False

# Create Sequential Model
model = models.Sequential()
model.add(base_model)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(NUM_CATEGORIES, activation='softmax'))

# Train the model
model.compile(optimizer=optimizers.RMSprop(learning_rate=1e-4), 
             loss='categorical_crossentropy', 
             metrics=['acc'])

with tf.device('/device:GPU:0'):
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds
    )

model.save('./model')

# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))
plt.plot(history.history['acc'], label='Train Accuracy')
plt.plot(history.history['val_acc'], label='Validation Accuracy')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

# Plot training & validation loss values
plt.figure(figsize=(12, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()