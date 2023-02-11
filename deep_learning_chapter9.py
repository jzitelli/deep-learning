import os

input_dir = 'images'
target_dir = os.path.join('annotations', 'trimaps')
input_img_paths = sorted([os.path.join(input_dir, fname) for fname in os.listdir(input_dir) if fname.endswith('.jpg')])
target_paths = sorted([os.path.join(target_dir, fname) for fname in os.listdir(target_dir)
                       if fname.endswith('.png') and not fname.startswith('.')])
import matplotlib.pyplot as plt
from tensorflow.keras.utils import load_img, img_to_array
