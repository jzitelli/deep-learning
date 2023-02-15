from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
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

data_augmentation = keras.Sequential([layers.RandomFlip('horizontal'),
                                      layers.RandomRotation(0.1),
                                      layers.RandomZoom(0.2)])
inputs = keras.Input(shape=(180,180,3))
x = data_augmentation(inputs)
x = layers.Rescaling(1./255)(x)
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
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)
model = keras.Model(inputs, outputs)

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

from tensorflow.keras.utils import image_dataset_from_directory

train_dataset = image_dataset_from_directory(new_base_dir / 'train',
                                             image_size=(180,180),
                                             batch_size=32)
validation_dataset = image_dataset_from_directory(new_base_dir / 'validation',
                                             image_size=(180,180),
                                             batch_size=32)
test_dataset = image_dataset_from_directory(new_base_dir / 'test',
                                             image_size=(180,180),
                                             batch_size=32)

# callbacks = [keras.callbacks.ModelCheckpoint(filepath='convnet_from_scratch.keras', save_best_only=True, monitor='val_loss')]
# callbacks = [keras.callbacks.ModelCheckpoint(filepath='convnet_from_scratch_with_augmentation.keras', save_best_only=True, monitor='val_loss')]
# history = model.fit(train_dataset, epochs=100, validation_data=validation_dataset, callbacks=callbacks)
#
# plt.figure()
# plt.plot(history.history['accuracy'], label='training accuracy')
# plt.plot(history.history['val_accuracy'], label='validation accuracy')
# plt.legend()
#
# plt.figure()
# plt.plot(history.history['loss'][1:], label='training loss')
# plt.plot(history.history['val_loss'][1:], label='validation loss')
# plt.legend()

# # test_model = keras.models.load_model('convnet_from_scratch.keras')
# test_model = keras.models.load_model('convnet_from_scratch_with_augmentation.keras')
# test_model = keras.models.load_model('fine_tuning.keras')
# test_loss, test_acc = model.evaluate(test_dataset)
# print(f"test accuracy: {test_acc:.3f}")

conv_base = keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(180,180,3))

def get_features_and_labels(dataset):
    all_features = []
    all_labels = []
    for images, labels in dataset:
        preprocessed_images = keras.applications.vgg16.preprocess_input(images)
        features = conv_base.predict(preprocessed_images)
        all_features.append(features)
        all_labels.append(labels)
    return np.concatenate(all_features), np.concatenate(all_labels)

train_features, train_labels = get_features_and_labels(train_dataset)
val_features, val_labels = get_features_and_labels(validation_dataset)
test_features, test_labels = get_features_and_labels(test_dataset)

inputs = keras.Input(shape=(5,5,512))
x = layers.Flatten()(inputs)
x = layers.Dense(256)(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)
model = keras.Model(inputs, outputs)

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
callbacks = [keras.callbacks.ModelCheckpoint(filepath='feature_extraction.keras', save_best_only=True, monitor='val_loss')]
history = model.fit(train_features, train_labels, epochs=20,
                    validation_data=(val_features, val_labels), callbacks=callbacks)

conv_base.trainable = False
inputs = keras.Input(shape=(180,180,3))
x = data_augmentation(inputs)
x = keras.applications.vgg16.preprocess_input(x)
x = conv_base(x)
x = layers.Flatten()(x)
x = layers.Dense(256)(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)
model = keras.Model(inputs, outputs)

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
callbacks = [keras.callbacks.ModelCheckpoint(filepath='feature_extraction_with_data_augmentation.keras',
                                             save_best_only=True, monitor='val_loss')]
history = model.fit(train_dataset, epochs=50,
                    validation_data=validation_dataset, callbacks=callbacks)

conv_base.trainable = True
for layer in conv_base.layers[:-4]:
    layer.trainable = False
model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.RMSprop(learning_rate=1e-5), metrics=['accuracy'])
callbacks = [keras.callbacks.ModelCheckpoint(filepath='fine_tuning.keras',
                                             save_best_only=True, monitor='val_loss')]
history = model.fit(train_dataset, epochs=30,
                    validation_data=validation_dataset, callbacks=callbacks)