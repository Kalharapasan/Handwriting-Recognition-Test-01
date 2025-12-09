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
    
    if np.mean(img_array) > 127:
        img_array = 255 - img_array
    
    img_array = img_array.astype('float32') / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    
    return img_array

def plot_prediction_comparison(model, x_test, y_test, num_samples=5):
    predictions = model.predict(x_test[:num_samples])
    predicted_labels = np.argmax(predictions, axis=1)
    actual_labels = np.argmax(y_test[:num_samples], axis=1)
    
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    
    for i in range(num_samples):
        axes[0, i].imshow(x_test[i].reshape(28, 28), cmap='gray')
        axes[0, i].set_title(f'Actual: {actual_labels[i]}')
        axes[0, i].axis('off')
    