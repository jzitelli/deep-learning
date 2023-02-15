import os
import matplotlib.pyplot as plt
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import random
from tensorflow import keras
from tensorflow.keras import layers

input_dir = 'images'
target_dir = os.path.join('annotations', 'trimaps')

input_img_paths = sorted([os.path.join(input_dir, fname) for fname in os.listdir(input_dir)
                          if fname.endswith('.jpg')])#[:3000]
target_paths = sorted([os.path.join(target_dir, fname) for fname in os.listdir(target_dir)
                       if fname.endswith('.png') and not fname.startswith('.')])#[:3000]

img_size = (100,100)
num_imgs = len(input_img_paths)

random.Random(1337).shuffle(input_img_paths)
random.Random(1337).shuffle(target_paths)

def path_to_input_image(path):
    return img_to_array(load_img(path, target_size=img_size))

def path_to_target(path):
    img = img_to_array(load_img(path, target_size=img_size, color_mode='grayscale'))
    img = img.astype('uint8') - 1
    return img

input_imgs = np.zeros((num_imgs,) + img_size + (3,), dtype='float32')
targets = np.zeros((num_imgs,) + img_size + (1,), dtype='uint8')
for i in range(num_imgs):
    if i % 100 == 0:
        print(f'i = {i}')
    input_imgs[i] = path_to_input_image(input_img_paths[i])
    targets[i] = path_to_target(target_paths[i])

num_val_samples = 1000
train_input_imgs = input_imgs[:-num_val_samples]
train_targets = targets[:-num_val_samples]
val_input_imgs = input_imgs[-num_val_samples:]
val_targets = targets[-num_val_samples:]

def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,))
    x = layers.Rescaling(1./255)(inputs)
    x = layers.Conv2D(64, 3, strides=2, activation='relu', padding='same')(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.Conv2D(128, 3, strides=2, activation='relu', padding='same')(x)
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    # x = layers.Conv2D(256, 3, strides=2, activation='relu', padding='same')(x)
    # x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
    # x = layers.Conv2DTranspose(256, 3, activation='relu', padding='same')(x)
    # x = layers.Conv2DTranspose(256, 3, activation='relu', padding='same', strides=2)(x)
    x = layers.Conv2DTranspose(128, 3, activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(128, 3, activation='relu', padding='same', strides=2)(x)
    x = layers.Conv2DTranspose(64, 3, activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(64, 3, activation='relu', padding='same', strides=2)(x)
    outputs = layers.Conv2D(num_classes, 3, activation='softmax', padding='same')(x)
    model = keras.Model(inputs, outputs)
    return model

model = get_model(img_size, num_classes=3)

model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
callbacks = [keras.callbacks.ModelCheckpoint('oxford_segmentation.keras', save_best_only=True)]
history = model.fit(train_input_imgs, train_targets,
                    epochs=50, callbacks=callbacks, batch_size=64,
                    validation_data=(val_input_imgs, val_targets))

plt.figure()
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.legend()

test_model = keras.models.load_model('oxford_segmentation.keras')

def compare_target(model, i):
    plt.figure(); plt.title('image')
    plt.imshow(val_input_imgs[i] / 255.0)
    plt.figure(); plt.title('predicted')
    #plt.imshow(model.predict(np.expand_dims(val_input_imgs[i], axis=0))[0] * 2.0)
    plt.imshow(model.predict(val_input_imgs[i].reshape((1,) + img_size + (3,)))[0] * 2.0)
    plt.figure(); plt.title('actual')
    plt.imshow(val_targets[i] / 255.0)

compare_target(test_model, 500)
