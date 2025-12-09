import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

def load_saved_model(model_path):
    return keras.models.load_model(model_path)

def preprocess_custom_image(image_path):
    from PIL import Image
    import cv2
    
    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img)