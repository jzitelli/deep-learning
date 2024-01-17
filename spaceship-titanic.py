import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
# import seaborn as sns
import os

try:
    data_dir = os.path.join(os.path.dirname(__file__), 'spaceship-titanic')
except:
    data_dir = 'spaceship-titanic'
train_data = pd.read_csv(os.path.join(data_dir, 'train.csv'))
test_data = pd.read_csv(os.path.join(data_dir, 'test.csv'))

def extract_features(data):
    data.CryoSleep = data.CryoSleep.fillna(False)
    data.Age = data.Age.fillna(data.Age.mean())
    data.VIP = data.VIP.fillna(False)
    # data.RoomService = data.RoomService.fillna(data.RoomService.mode())
    # data.FoodCourt = data.FoodCourt.fillna(data.FoodCourt.mode())
    data.ShoppingMall = data.ShoppingMall.fillna(0.0)
    # data.Spa = data.Spa.fillna(data.Spa.mode())
    data.HomePlanet = data.HomePlanet.fillna('NaN')
    columns = ['PassengerId', 'CryoSleep', 'Destination', 'Age',
    'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
    'Name']
    columns = [c for c in columns if not data[c].hasnans]
    # 'Cabin',
    result = np.vstack([data[col].to_numpy('float32') for col in columns] +
                       [#(data['Sex'] == 'male').to_numpy('float32'),
                        pd.Categorical(data['HomePlanet'].fillna('NaN')).codes]).T.copy()
    result -= result.mean(axis=0)
    result /= result.std(axis=0)
    return result

td = extract_features(train_data)

model = keras.Sequential([
    layers.Dense(16, activation='relu'),
    layers.Dense(16, activation='relu'),
    # layers.Dense(16, activation='relu'),
    # layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(td, train_data['Transported'].to_numpy('float32'),
                    epochs=20, batch_size=256, validation_split=0.3)

plt.figure()
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.legend()

plt.figure()
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.legend()

td = extract_features(test_data)

predictions = model.predict(td)
predictions = pd.DataFrame({'PassengerId': test_data.PassengerId,
                            'Survived': (predictions.flatten() >= 0.5).astype('int')})
print(predictions)
with open(os.path.join(data_dir, 'spaceship-titanic_predictions.csv'), 'w') as f:
    f.write(predictions.to_csv(index=False, lineterminator='\n'))
