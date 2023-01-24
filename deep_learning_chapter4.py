from tensorflow.keras.datasets import imdb
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
word_dict = {v: k for k, v in imdb.get_word_index().items()}

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, np.array(sequence)] = 1
        # for j in sequence:
        #     results[i,j] = 1
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')


model = keras.Sequential([
    layers.Dense(16, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

x_val = x_train[:10000]
x_train_partial = x_train[10000:]
y_val = y_train[:10000]
y_train_partial = y_train[10000:]

history = model.fit(x_train_partial,
                    y_train_partial,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

plt.figure()
plt.plot(history.history['loss'], '-x', label='training loss')
plt.plot(history.history['val_loss'], '-x', label='validation loss')
plt.legend()
plt.show()

plt.figure()
plt.plot(history.history['accuracy'], '-x', label='training accuracy')
plt.plot(history.history['val_accuracy'], '-x', label='validation accuracy')
plt.legend()
plt.show()

# model = keras.Sequential([
#     layers.Dense(16, activation='relu'),
#     layers.Dense(16, activation='relu'),
#     layers.Dense(1, activation='sigmoid')
# ])
# model.compile(optimizer='rmsprop',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
# model.fit(x_train,
#           y_train,
#           epochs=4,
#           batch_size=512)
# results = model.evaluate(x_test, y_test)
