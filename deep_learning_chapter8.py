from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
# import numpy as np
import os, shutil, pathlib


try:
    original_dir = pathlib.Path(os.path.join(os.path.dirname(__file__), 'dogs-vs-cats', 'train'))
    new_base_dir = pathlib.Path(os.path.join(os.path.dirname(__file__), 'dogs-vs-cats-small'))
except:
    original_dir = pathlib.Path(os.path.join('dogs-vs-cats', 'train'))
    new_base_dir = pathlib.Path('dogs-vs-cats-small')

def make_subset(subset_name, start_index, end_index):
    for category in ('cat', 'dog'):
        dir = new_base_dir / subset_name / category
        os.makedirs(dir)
        for fname in [f"{category}.{i}.jpg" for i in range(start_index, end_index)]:
            shutil.copyfile(src=original_dir / fname, dst=dir / fname)

# make_subset('train', 0, 1000)
# make_subset('validation', 1000, 1500)
# make_subset('test', 1500, 2500)


# (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
#
# train_images = train_images.astype('float32') / 255
# train_images = train_images.reshape((len(train_images), 28, 28, 1))
# test_images = test_images.astype('float32') / 255
# test_images = test_images.reshape((len(test_images), 28, 28, 1))
#
# inputs = keras.Input(shape=(28,28,1))
# x = layers.Conv2D(filters=32, kernel_size=3, activation='relu')(inputs)
# x = layers.MaxPooling2D(pool_size=2)(x)
# x = layers.Conv2D(filters=64, kernel_size=3, activation='relu')(x)
# x = layers.MaxPooling2D(pool_size=2)(x)
# x = layers.Conv2D(filters=128, kernel_size=3, activation='relu')(x)
# x = layers.Flatten()(x)
# outputs = layers.Dense(10, activation='softmax')(x)
# model = keras.Model(inputs, outputs)
#
# model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#
# history = model.fit(train_images, train_labels, epochs=5, batch_size=128, validation_split=0.1)
# # plt.plot(history.history['loss'], label='training loss')
# plt.plot(history.history['accuracy'], label='training accuracy')
# plt.plot(history.history['val_accuracy'], label='validation accuracy')
# plt.legend()
#
# print(model.evaluate(test_images, test_labels))

inputs = keras.Input(shape=(180,180,3))
x = layers.Rescaling(1./255)(inputs)
x = layers.Conv2D(filters=32, kernel_size=3, activation='relu')(inputs)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=64, kernel_size=3, activation='relu')(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=128, kernel_size=3, activation='relu')(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=256, kernel_size=3, activation='relu')(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=256, kernel_size=3, activation='relu')(x)
x = layers.Flatten()(x)
outputs = layers.Dense(1, activation='sigmoid')(x)
model = keras.Model(inputs, outputs)

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
