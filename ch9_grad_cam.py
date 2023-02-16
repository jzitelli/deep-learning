import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
model = keras.applications.xception.Xception(weights='imagenet')
def get_img_array(img_path, target_size):
    img = keras.utils.load_img(img_path, target_size=target_size)
    array = keras.utils.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return keras.applications.xception.preprocess_input(array)

# img_path = keras.utils.get_file(fname='elephant.jpg', origin='https://img-datasets.s3.amazonaws.com/elephant.jpg')
# img_path = 'giraffe.jpg'
# img_path = 'kangaroo.jpg'
# img_path = os.path.join('dogs-vs-cats', 'train', 'cat.8.jpg')
# img_path = os.path.join('dogs-vs-cats', 'train', 'cat.83.jpg')
# img_path = os.path.join('dogs-vs-cats', 'train', 'dog.7.jpg')
img_path = os.path.join('dogs-vs-cats', 'train', 'dog.34.jpg')

img_array = get_img_array(img_path, target_size=(299, 299))
preds = model.predict(img_array)
print(keras.applications.xception.decode_predictions(preds, top=3))
# plt.imshow(img_array[0])

last_conv_layer = model.get_layer('block14_sepconv2_act')
last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)

classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
x = classifier_input
for layer_name in ['avg_pool', 'predictions']:
    x = model.get_layer(layer_name)(x)
classifier_model = keras.Model(classifier_input, x)

with tf.GradientTape() as tape:
    last_conv_layer_output = last_conv_layer_model(img_array)
    tape.watch(last_conv_layer_output)
    preds = classifier_model(last_conv_layer_output)
    top_pred_index = tf.argmax(preds[0])
    top_class_channel = preds[:, top_pred_index]
grads = tape.gradient(top_class_channel, last_conv_layer_output)

pooled_grads = tf.reduce_mean(grads, axis=(0,1,2)).numpy()
last_conv_layer_output = last_conv_layer_output.numpy()[0]
for i in range(pooled_grads.shape[-1]):
    last_conv_layer_output[:,:,i] *= pooled_grads[i]
heatmap = np.mean(last_conv_layer_output, axis=-1)
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)

plt.figure()
plt.imshow(keras.utils.img_to_array(keras.utils.load_img(img_path, target_size=(299,299))) / 255)
plt.imshow(heatmap, alpha=0.4, origin='upper', extent=[-0.5, 299-0.5, 299-0.5, -0.5])
