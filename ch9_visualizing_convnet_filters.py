import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

model = keras.applications.xception.Xception(weights='imagenet', include_top=False)
# layer = model.get_layer(name='block3_sepconv1')
# layer = model.get_layer(name='block6_sepconv1')
layer = model.get_layer(name='block8_sepconv1')
# layer = model.get_layer(name='block10_sepconv1')
feature_extractor = keras.Model(inputs=model.input, outputs=layer.output)

def compute_loss(image, filter_index):
    activation = feature_extractor(image)
    filter_activation = activation[:, 2:-2, 2:-2, filter_index]
    return tf.reduce_mean(filter_activation)

@tf.function
def gradient_ascent_step(image, filter_index, learning_rate):
    with tf.GradientTape() as tape:
        tape.watch(image)
        loss = compute_loss(image, filter_index)
    grads = tape.gradient(loss, image)
    grads = tf.math.l2_normalize(grads)
    image += learning_rate * grads
    return image

img_width, img_height = 300, 300

def generate_filter_pattern(filter_index):
    iterations = 50
    learning_rate = 5.0
    image = tf.random.uniform(minval=0.4, maxval=0.6, shape=(1, img_width, img_height, 3))
    for i in range(iterations):
        image = gradient_ascent_step(image, filter_index, learning_rate)
    return image[0].numpy()

def deprocess_image(image):
    image -= image.mean()
    image /= image.std()
    image *= 64
    image += 128
    image = np.clip(image, 0, 255).astype('uint8')
    #image = image[25:-25, 25:-25, :]
    return image


plt.figure(); plt.imshow(deprocess_image(generate_filter_pattern(0)))
plt.figure(); plt.imshow(deprocess_image(generate_filter_pattern(1)))
plt.figure(); plt.imshow(deprocess_image(generate_filter_pattern(2)))
plt.figure(); plt.imshow(deprocess_image(generate_filter_pattern(3)))
# plt.figure(); plt.imshow(deprocess_image(generate_filter_pattern(4)))
# plt.figure(); plt.imshow(deprocess_image(generate_filter_pattern(5)))
