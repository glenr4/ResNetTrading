import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

# Define data directories
train_dir = './images'  # Replace with the path to your data directory

# Image dimensions
img_width, img_height = 224, 224  # ResNet50 input size

# Batch size and number of classes
batch_size = 32
num_classes = 3

epochs = 10  # Adjust as needed

# Data Preprocessing with channel conversion
def rgba_to_rgb(image):
    """Converts RGBA image to RGB, ignoring the alpha channel."""
    image = image[:, :, :3]  # Slice the image to remove the alpha channel
    return image

# TODO update this to point to images directories that are already split into train and test.

train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values
    validation_split=0.2, # create validation split.
    preprocessing_function=rgba_to_rgb # Add preprocessing function
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='rgb',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='rgb',
    subset='validation'
)

# Load ResNet50 (without top layers)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Fine tuning of the model
base_model.trainable = True
fine_tune_at = 143 #  the beginning of the conv5 stage
# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable = False

# Add custom classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

# Train the model

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=epochs
)

# Save the trained model
model.save('/model/resnet50_transfer_learning.h5')

print("Model training complete. Model saved as 'resnet50_transfer_learning.h5'")

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

# Example of how to load the model later.
# loaded_model = tf.keras.models.load_model('resnet50_transfer_learning.h5')

# to use the loaded model for prediction you will need to preprocess the incoming images in the same way you did the training images.