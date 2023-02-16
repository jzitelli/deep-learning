import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

df = pd.read_csv('jena_climate_2009_2016.csv')

temperature = df['T (degC)'].to_numpy()
raw_data = df.iloc[:,1:].to_numpy()

# plt.plot(temperature)
# plt.plot(temperature[:1440])

num_train_samples = int(0.5*len(raw_data))
num_val_samples = int(0.25*len(raw_data))
num_test_samples = len(raw_data) - num_train_samples - num_val_samples

mean = raw_data[:num_train_samples].mean(axis=0)
raw_data -= mean
std = raw_data[:num_train_samples].std(axis=0)
raw_data /= std

sampling_rate = 6 # every hour
sequence_length = 120 # 5 days
delay = sampling_rate * (sequence_length + 24 - 1)
batch_size = 256

train_dataset = keras.utils.timeseries_dataset_from_array(
    raw_data[:-delay],
    targets=temperature[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=0, end_index=num_train_samples
)

val_dataset = keras.utils.timeseries_dataset_from_array(
    raw_data[:-delay],
    targets=temperature[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=num_train_samples,
    end_index=num_train_samples+num_val_samples
)

test_dataset = keras.utils.timeseries_dataset_from_array(
    raw_data[:-delay],
    targets=temperature[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=num_train_samples+num_val_samples
)


def evaluate_naive_method(dataset):
    total_abs_err = 0
    samples_seen = 0
    for samples, targets in dataset:
        preds = samples[:,-1,1] * std[1] + mean[1]
        total_abs_err += np.sum(np.abs(preds - targets))
        samples_seen += len(samples)
    return total_abs_err / samples_seen

# print(f'Validation MAE: {evaluate_naive_method(val_dataset)}')
# print(f'Test MAE: {evaluate_naive_method(test_dataset)}')

inputs = keras.Input(shape=(sequence_length, raw_data.shape[-1]))
x = layers.Flatten()(inputs)
x = layers.Dense(16, activation='relu')(x)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

callbacks = [keras.callbacks.ModelCheckpoint('jena_dense.keras', save_best_only=True)]
history = model.fit(train_dataset, epochs=10, validation_data=val_dataset, callbacks=callbacks)

plt.figure()
plt.plot(history.history['mae'], label='training MAE')
plt.plot(history.history['val_mae'], label='validation MAE')
plt.legend()

model = keras.models.load_model('jena_dense.keras')
print(f'model.evaluate(test_dataset): {model.evaluate(test_dataset)}')

plt.figure()
plt.plot(history.history['mae'][1:], label='training MAE')
plt.plot(history.history['val_mae'][1:], label='validation MAE')
plt.legend()

# LSTM model
inputs = keras.Input(shape=(sequence_length, raw_data.shape[-1]))
x = layers.LSTM(16)(inputs)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
callbacks = [keras.callbacks.ModelCheckpoint('jena_lstm.keras', save_best_only=True)]
history = model.fit(train_dataset, epochs=10, validation_data=val_dataset, callbacks=callbacks)

model = keras.models.load_model('jena_lstm.keras')
print(f'model.evaluate(test_dataset): {model.evaluate(test_dataset)}')
