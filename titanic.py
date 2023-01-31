import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
# import seaborn as sns
import os

try:
    data_dir = os.path.join(os.path.dirname(__file__), 'titanic')
except:
    data_dir = 'titanic'
train_data = pd.read_csv(os.path.join(data_dir, 'train.csv'))

def extract_features(data):
    columns = ['Pclass', 'SibSp', 'Parch', 'Fare']
    return np.vstack([data[col].to_numpy('float32') for col in columns] +
                     [(train_data['Sex'] == 'male').to_numpy('float32')]).T.copy()

td = extract_features(train_data)
td -= td.mean(axis=0)
td /= td.std(axis=0)

model = keras.Sequential([
    layers.Dense(32, activation='relu'),
    layers.Dense(32, activation='relu'),
    # layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(td, train_data['Survived'].to_numpy('float32'),
                    epochs=100, batch_size=64, validation_split=0.4)

plt.figure()
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.legend()

plt.figure()
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.legend()
