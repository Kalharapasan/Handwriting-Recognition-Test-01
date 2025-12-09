# Handwriting Recognition Application

A comprehensive machine learning application for recognizing handwritten digits using Convolutional Neural Networks (CNNs) trained on the MNIST dataset. Features an interactive drawing interface for real-time digit recognition.

## 🚀 Features

- **Advanced CNN Architecture**: 3-layer convolutional network with batch normalization and dropout
- **Interactive Drawing Canvas**: Real-time digit drawing and recognition interface
- **Real-time Prediction**: Instant prediction with confidence scores and probability distributions
- **Model Persistence**: Save and load trained models for future use
- **Comprehensive Evaluation**: Confusion matrix, classification reports, and per-class accuracy
- **Feature Visualization**: Visualize convolutional layer feature maps
- **Training Monitoring**: Live training curves with early stopping and learning rate scheduling
- **Data Preprocessing**: Automatic preprocessing pipeline for optimal model performance

## 📁 Project Structure

```
├── handwriting_recognition.ipynb  # Main Jupyter notebook with complete workflow
├── utils.py                       # Utility functions for model loading and image preprocessing
├── requirements.txt               # Python dependencies
├── README.md                      # This documentation
├── data/                          # Directory for storing datasets
├── models/                        # Directory for saving trained models
│   ├── handwriting_model.h5       # Saved model file
│   ├── final_handwriting_model.h5 # Final trained model
│   ├── training_history.npy       # Training history data
│   └── evaluation_results.npy     # Evaluation metrics
└── venv/                          # Python virtual environment
```

## 🛠 Installation

### Prerequisites
- Python 3.7+
- TensorFlow 2.15.0 compatible hardware (CPU/GPU)
- At least 4GB RAM recommended

### Setup Steps

1. **Clone or download the project files**

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   - **Windows**: `venv\Scripts\activate`
   - **Linux/Mac**: `source venv/bin/activate`

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Enable Long Paths (Windows only)**:
   ```cmd
   reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled /t REG_DWORD /d 1 /f
   ```

## 📦 Dependencies

- **TensorFlow 2.15.0**: Deep learning framework with Keras API
- **NumPy**: Numerical computing and array operations
- **Matplotlib**: Data visualization and plotting
- **OpenCV-Python**: Computer vision operations
- **Pillow (PIL)**: Image processing and manipulation
- **Scikit-learn**: Machine learning utilities and metrics
- **IPython Widgets**: Interactive UI components for Jupyter
- **Seaborn**: Statistical data visualization (for confusion matrix)

## 🎯 Usage

### Running the Application

1. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook handwriting_recognition.ipynb
   ```

2. **Execute cells in order**:
   - **Cell 1-2**: Import libraries and verify installation
   - **Cell 3-4**: Load and preprocess MNIST dataset
   - **Cell 5**: Visualize sample data
   - **Cell 6-8**: Create, compile, and display CNN model
   - **Cell 9-10**: Train the model with callbacks
   - **Cell 11-12**: Evaluate model performance
   - **Cell 13**: Visualize training history
   - **Cell 14-25**: Set up interactive drawing interface
   - **Cell 26-27**: Advanced evaluation and visualization

### Interactive Features

#### Drawing Canvas
- **Draw**: Click and drag to draw digits (0-9)
- **Clear**: Reset the canvas for new drawings
- **Predict**: Get real-time prediction with confidence scores
- **Save/Load Model**: Persist trained models

#### Real-time Prediction
```python
# The interface provides:
# - Visual feedback of your drawing
# - Predicted digit (0-9)
# - Confidence percentage
# - Probability distribution across all digits
```

## 🧠 Model Architecture

### CNN Architecture Details

```python
Model: Sequential([
    # Convolutional Block 1
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    # Convolutional Block 2
    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    # Convolutional Block 3
    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    Dropout(0.25),

    # Fully Connected Layers
    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
```

### Training Configuration

- **Optimizer**: Adam (default parameters)
- **Loss Function**: Categorical Cross-entropy
- **Batch Size**: 128
- **Epochs**: Up to 50 (with early stopping)
- **Callbacks**:
  - Early Stopping (patience=5)
  - ReduceLROnPlateau (factor=0.2, patience=3)

## 📊 Data & Performance

### Dataset: MNIST
- **Training samples**: 60,000 handwritten digits
- **Test samples**: 10,000 handwritten digits
- **Image dimensions**: 28×28 grayscale pixels
- **Classes**: 10 (digits 0-9)
- **Preprocessing**: Normalization (0-1), reshaping for CNN input

### Expected Performance
- **Training Accuracy**: ~99.5%
- **Validation Accuracy**: ~99.2%
- **Test Accuracy**: ~99.0%
- **Training Time**: ~10-15 minutes (depending on hardware)

### Per-Class Performance
The model typically achieves high accuracy across all digits:
- Best performance: Digits 1, 6, 7, 9
- Challenging digits: 4, 5, 8 (due to similar shapes)

## 🔧 Key Functions

### From `utils.py`:
```python
load_saved_model(model_path)           # Load trained model
preprocess_custom_image(image_path)    # Prepare custom images
plot_prediction_comparison(model, x_test, y_test, num_samples)  # Visualize predictions
```

### From the Notebook:
```python
load_and_preprocess_data()             # Load MNIST with preprocessing
create_cnn_model()                     # Build CNN architecture
evaluate_model(model, x_test, y_test) # Comprehensive evaluation
plot_training_history(history)         # Training curves
plot_confusion_matrix(y_true, y_pred) # Confusion matrix
visualize_feature_maps(model, image)  # CNN feature visualization
```

## 🎨 Advanced Features

### Feature Map Visualization
- Visualize what each convolutional layer "sees"
- Understand how the network processes different digit features
- Debug model behavior and feature extraction

### Confusion Matrix Analysis
- Identify which digits are commonly misclassified
- Understand model weaknesses for specific digit pairs
- Guide potential model improvements

### Interactive Drawing Interface
- HTML5 Canvas-based drawing
- Real-time prediction feedback
- Probability distribution visualization
- Model save/load functionality

## 🔍 Custom Image Prediction

### Using the Interactive Interface
1. Draw a digit in the canvas
2. Click "Predict" for instant results
3. View confidence scores and probability bars

### Programmatic Prediction
```python
from utils import preprocess_custom_image, load_saved_model

# Load your trained model
model = load_saved_model('models/final_handwriting_model.h5')

# Preprocess your image
processed_image = preprocess_custom_image('path/to/your/digit.png')

# Make prediction
prediction = model.predict(processed_image)
predicted_digit = np.argmax(prediction)
confidence = np.max(prediction)

print(f"Predicted digit: {predicted_digit}")
print(f"Confidence: {confidence:.2%}")
```

### Image Requirements
- **Format**: PNG, JPG, JPEG
- **Size**: Any size (automatically resized to 28×28)
- **Color**: Grayscale or RGB (converted to grayscale)
- **Background**: White or light background preferred
- **Digit**: Black/dark ink on light background

## 🐛 Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Ensure virtual environment is activated
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac

# Reinstall requirements
pip install -r requirements.txt
```

#### 2. Memory Issues
- Reduce batch size in training cell
- Use `model.fit(..., batch_size=64)` instead of 128
- Close other applications to free RAM

#### 3. Training Takes Too Long
- Enable GPU acceleration if available
- Reduce epochs or use early stopping
- Use smaller batch size

#### 4. Interactive Canvas Not Working
```python
# Restart Jupyter kernel
# Clear all outputs and re-run cells
# Ensure ipywidgets is properly installed
pip install ipywidgets
jupyter nbextension enable --py --sys-prefix widgetsnbextension
```

#### 5. Model Loading Errors
- Ensure model file exists in `models/` directory
- Check TensorFlow version compatibility
- Use `model.save('model.h5')` for saving

### Performance Optimization

#### For Better Accuracy
- Increase training epochs (remove early stopping)
- Add more convolutional layers
- Experiment with different architectures
- Use data augmentation

#### For Faster Training
- Use GPU acceleration
- Reduce model complexity
- Use smaller batch sizes
- Implement gradient accumulation

## 📈 Model Evaluation

### Comprehensive Metrics
The application provides:
- **Overall Accuracy**: Test set performance
- **Per-Class Accuracy**: Individual digit performance
- **Precision, Recall, F1-Score**: Detailed classification metrics
- **Confusion Matrix**: Visual error analysis
- **Training Curves**: Learning progress visualization

### Interpreting Results
```python
# Example output interpretation:
Test Accuracy: 0.9912
Precision: 0.9913
Recall: 0.9912
F1-Score: 0.9912

Per-class Accuracy:
Digit 0: 0.9959
Digit 1: 0.9974
...
```

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Make your changes
4. Test thoroughly with different digit samples
5. Ensure model performance doesn't degrade
6. Submit a pull request

### Code Style
- Follow PEP 8 Python style guidelines
- Add docstrings to new functions
- Include comments for complex logic
- Test interactive features manually

### Adding New Features
- Update this README with new functionality
- Add corresponding utility functions to `utils.py`
- Include example usage in the notebook
- Test with various input scenarios

## 📄 License

This project is open-source. Please check the license file for details.

## 🙏 Acknowledgments

- **MNIST Dataset**: Yann LeCun and Corinna Cortes
- **TensorFlow/Keras**: Google Brain Team
- **Open Source Community**: Contributors to all used libraries

## 🔗 References

- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras Documentation](https://keras.io/)
- [CNN Architecture Guide](https://cs231n.github.io/convolutional-networks/)

---

**Last Updated**: December 9, 2025
**Version**: 1.0.0
**Author**: Handwriting Recognition Project Team
