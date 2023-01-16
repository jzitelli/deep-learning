# -*- coding: utf-8 -*-
"""deep_learning_chapter3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1D0IOmAikodVjySL8lmf1qSDEMophaeVB
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

num_samples_per_class = 1000
negative_samples = np.random.multivariate_normal(
    mean=[0,3],
    cov=[[1, 0.5], [0.5, 1]], size=num_samples_per_class)
positive_samples = np.random.multivariate_normal(
    mean=[3,0],
    cov=[[1, 0.5], [0.5, 1]], size=num_samples_per_class)

plt.figure()
plt.scatter(negative_samples[:,0], negative_samples[:,1], alpha=0.1)
plt.scatter(positive_samples[:,0], positive_samples[:,1], alpha=0.1)
plt.show()

inputs = np.vstack([negative_samples,
                    positive_samples]).astype(np.float32)
targets = np.vstack([np.zeros((num_samples_per_class, 1)),
                     np.ones((num_samples_per_class, 1))]).astype(np.float32)

from tensorflow import keras
model = keras.Sequential([keras.layers.Dense(1)])
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.01),
              loss=keras.losses.MeanSquaredError(),
              metrics=[keras.metrics.BinaryAccuracy()])

perm = np.random.permutation(len(inputs))

history = model.fit(inputs[perm], targets[perm],
                    epochs=40,
                    batch_size=256)

plt.figure()
plt.plot(history.history['loss'], '-x')
plt.show()

W, b = model.get_weights()

# input_dim = 2
# output_dim = 1
# W = tf.Variable(initial_value=tf.random.uniform(shape=(input_dim, output_dim)))
# b = tf.Variable(initial_value=tf.zeros(shape=(output_dim,)))
# learning_rate = 0.1

# def model(inputs):
#   return tf.matmul(inputs, W) + b

# def square_loss(targets, predictions):
#   per_sample_losses = tf.square(targets - predictions)
#   return tf.reduce_mean(per_sample_losses)

# def training_step(inputs, targets):
#   with tf.GradientTape() as tape:
#     predictions = model(inputs)
#     loss = square_loss(targets, predictions)
#   grad_loss_wrt_W, grad_loss_wrt_b = tape.gradient(loss, [W, b])
#   W.assign_sub(grad_loss_wrt_W * learning_rate)
#   b.assign_sub(grad_loss_wrt_b * learning_rate)
#   return loss

# losses = [training_step(inputs, targets) for step in range(100)]

# plt.figure()
# plt.plot(losses, '-x')
# plt.show()

plt.figure()
plt.scatter(negative_samples[:,0], negative_samples[:,1], alpha=0.1)
plt.scatter(positive_samples[:,0], positive_samples[:,1], alpha=0.1)
plt.plot(x := np.linspace(-1, 4, 2),
         -W[0] / W[1] * x + (0.5 - b) / W[1])
plt.show()
