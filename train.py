# https://medium.com/@paravisionlab/supercharge-your-ai-resnet50-transfer-learning-unleashed-b7c0e40976c4
# conda install tensorflow

from keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from tensorflow.keras.applications import ResNet50
from keras import models
from keras import layers
from keras import optimizers
import tensorflow as tf
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Check shape
# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)

# print(x_train[0][:5, :5, :]) # prints the first 5x5 pixels of each channel of the first image.

# Normalise the pixels
x_train = x_train/255
x_test = x_test/255

# print(y_train.shape)
# print(y_train)

# One hot encode categories
# [6] becomes [0 0 0 0 0 1 0 0 0]
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
# print(y_train.shape)
# print(y_train[0])

# Split training into training and validation
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

# Load the model
# Excluding top: excludes final classification layers
base_model=ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# base_model.summary()

# Fine tuning of the model
# for i, layer in enumerate(base_model.layers):
#       print(i, layer.name)
base_model.trainable = True
fine_tune_at = 143 #  the beginning of the conv5 stage

#Create A Sequential Model
model = models.Sequential()
model.add(layers.UpSampling2D(size=(7,7))) # Convert images from 32x32 to 224x224 (ResNet minimum)
model.add(base_model)
model.add(layers.GlobalAveragePooling2D()) # summarise the spatial information in CNN feature maps
model.add(layers.Dense(10, activation='softmax')) # Classify into 10 classes with probability value

# Train the model
model.compile(optimizer=optimizers.RMSprop(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['acc'])
with tf.device('/device:GPU:0'):
      history = model.fit(x_train, y_train, epochs=5, batch_size=20, validation_data=(x_val, y_val))
model.save('./model')
model.evaluate(x_test, y_test)

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
plt.show(False)