import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

def load_saved_model(model_path):
    return keras.models.load_model(model_path)