import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
import os

try:
    data_dir = os.path.join(os.path.dirname(__file__), 'house-prices-advanced-regression-techniques')
except:
    data_dir = 'house-prices-advanced-regression-techniques'
train_data = pd.read_csv(os.path.join(data_dir, 'train.csv'))
test_data = pd.read_csv(os.path.join(data_dir, 'test.csv'))

plt.hist(train_data.SalePrice, bins=50)

def extract_features(data):
    columns = ['LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
               #'TotalBsmtSF',
               '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea',
               #'BsmtFullBath', 'BsmtHalfBath',
               'FullBath', 'HalfBath', 'BedroomAbvGr', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch',
               '3SsnPorch', 'ScreenPorch', 'MiscVal', 'TotRmsAbvGrd',
               #'BsmtUnfSF', 'TotalBsmtSF'
               ]
    result = np.vstack([data[col].to_numpy('float32') for col in columns]).T.copy()
    result -= result.mean(axis=0)
    result /= result.std(axis=0)
    return result

td = extract_features(train_data)

model = keras.Sequential([
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(32, activation='relu'),
    # layers.Dense(16, activation='relu'),
    layers.Dense(1)
])
# model = keras.Sequential([
#     layers.Dense(64, activation='relu'),
#     layers.Dense(64, activation='relu'),
#     layers.Dense(1)
# ])

model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

history = model.fit(td, train_data['SalePrice'].to_numpy('float32'),
                    epochs=400, batch_size=256, validation_split=0.4)

plt.figure()
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.legend()

plt.figure()
plt.plot(history.history['mae'], label='training mae')
plt.plot(history.history['val_mae'], label='validation mae')
plt.legend()


td = extract_features(test_data)
predictions = model.predict(td)

predictions = pd.DataFrame({'Id': test_data.Id,
                            'SalePrice': predictions.flatten()})
print(predictions)
with open(os.path.join(data_dir, 'predictions.csv'), 'w') as f:
    f.write(predictions.to_csv(index=False, lineterminator='\n'))
