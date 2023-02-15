from tensorflow.keras.datasets import reuters
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import time

num_words = 20000
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=num_words)
word_dict = {v: k for k, v in reuters.get_word_index().items()}

print(" ".join([word_dict.get(i-3,'?') for i in train_data[0]]))

def vectorize_sequences(sequences, dimension=num_words):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, np.array(sequence)] = 1
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

nval = 1000

# model = keras.Sequential([
#     layers.Dense(64, activation='relu'),
#     layers.Dense(64, activation='relu'),
#     layers.Dense(46, activation='softmax')
# ])
#
# y_train = to_categorical(train_labels)
# y_test = to_categorical(test_labels)
#
# model.compile(optimizer='rmsprop',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
#
# t0 = time.time()
# history = model.fit(x_train[nval:], y_train[nval:],
#                     epochs=20, batch_size=512,
#                     validation_data=(x_train[:nval], y_train[:nval]),
#                     verbose=1)
# print(time.time() - t0)
#
# plt.figure()
# plt.plot(history.history['loss'], label='training loss')
# plt.plot(history.history['val_loss'], label='validation loss')
# plt.legend()
# plt.show()


model = keras.Sequential([
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(46, activation='softmax')
])

y_train = np.array(train_labels)
y_test = np.array(test_labels)

model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

t0 = time.time()
history = model.fit(x_train[nval:], y_train[nval:],
                    epochs=20, batch_size=512,
                    validation_data=(x_train[:nval], y_train[:nval]),
                    verbose=1)
print(time.time() - t0)

plt.figure()
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.legend()
plt.show()
