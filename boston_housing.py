from tensorflow.keras.datasets import boston_housing
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std

def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


k = 4
nval = len(train_data) // k

model = build_model()
history = model.fit(train_data, train_targets,
                    epochs=100, batch_size=16)
                    # validation_data=(x_train[:nval], y_train[:nval])

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['mae'])