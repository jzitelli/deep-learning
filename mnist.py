from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.astype('float32') / 255
train_images = train_images.reshape((len(train_images), 28*28))
test_images = test_images.astype('float32') / 255
test_images = test_images.reshape((len(test_images), 28*28))

# plt.imshow(test_images[0].reshape((28,28)))

from tensorflow import keras
from tensorflow.keras import layers

def build_model():
    model = keras.Sequential([
        layers.Dense(512, activation='relu'),
        layers.Dense(10, activation='softmax'),
    ])
    model.compile(optimizer='rmsprop',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = build_model()
history = model.fit(train_images, train_labels,
                    epochs=10, batch_size=256, validation_split=0.2)

plt.figure()
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.legend()

plt.figure()
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.legend()

print(model.evaluate(test_images, test_labels))


# model = build_model()
# train_labels_shuffled = train_labels.copy()
# np.random.shuffle(train_labels_shuffled)
# history = model.fit(train_images, train_labels_shuffled,
#                     epochs=10, batch_size=128, validation_split=0.2)
# plt.figure()
# plt.plot(history.history['loss'], label='training loss')
# plt.plot(history.history['val_loss'], label='validation loss')
# plt.legend()
