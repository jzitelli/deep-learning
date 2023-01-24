import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os

try:
    data_dir = os.path.join(os.path.dirname(__file__), 'house-prices-advanced-regression-techniques')
except:
    data_dir = 'house-prices-advanced-regression-techniques'
train_data = pd.read_csv(os.path.join(data_dir, 'train.csv'))

td = np.vstack([
    # train_data.LotFrontage.to_numpy('float32'),
    train_data.LotArea.to_numpy('float32'),
    train_data.OverallQual.to_numpy('float32'),
    train_data.OverallCond.to_numpy('float32'),
    train_data.YearBuilt.to_numpy('float32'),
    train_data.YearRemodAdd.to_numpy('float32'),
    train_data.TotalBsmtSF.to_numpy('float32'),
    train_data['1stFlrSF'].to_numpy('float32'),
    train_data['2ndFlrSF'].to_numpy('float32'),
    train_data['LowQualFinSF'].to_numpy('float32'),
    train_data['GrLivArea'].to_numpy('float32'),
    # train_data['BsmtFullBath'].to_numpy('float32'),
    # train_data['BsmtHalfBath'].to_numpy('float32'),
    train_data['FullBath'].to_numpy('float32'),
    train_data['HalfBath'].to_numpy('float32'),
    train_data['BedroomAbvGr'].to_numpy('float32'),
    train_data['WoodDeckSF'].to_numpy('float32'),
    train_data['OpenPorchSF'].to_numpy('float32'),
    train_data['EnclosedPorch'].to_numpy('float32'),
    train_data['3SsnPorch'].to_numpy('float32'),
    train_data['ScreenPorch'].to_numpy('float32'),
    train_data['MiscVal'].to_numpy('float32'),
    train_data['TotRmsAbvGrd'].to_numpy('float32'),
    # train_data['ScreenPorch'].to_numpy('float32'),
]).T.copy()

mean = td.mean(axis=0)
td -= mean
std = td.std(axis=0)
td /= std

train_labels = train_data['SalePrice'].to_numpy('float32')

model = keras.Sequential([
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1)
])
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

history = model.fit(td.T, train_labels,
                    epochs=400, batch_size=256, validation_split=0.2)

plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.legend()

plt.figure()
plt.plot(history.history['mae'], label='training mae')
plt.plot(history.history['val_mae'], label='validation mae')
plt.legend()
