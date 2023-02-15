import os
import matplotlib.pyplot as plt
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import random
from tensorflow import keras
from tensorflow.keras import layers

model = keras.applications.xception.Xception(weights='imagenet', include_top=False)