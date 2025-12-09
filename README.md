# Handwriting Recognition Application

A machine learning application for recognizing handwritten digits using Convolutional Neural Networks (CNNs) trained on the MNIST dataset.

## Features

- **MNIST Dataset Training**: Train a CNN model on the famous MNIST handwritten digits dataset
- **Real-time Prediction**: Make predictions on custom handwritten digit images
- **Interactive Visualization**: Visualize training data, model predictions, and performance metrics
- **Model Persistence**: Save and load trained models for future use
- **Data Preprocessing**: Automatic preprocessing of input images for optimal model performance

## Project Structure

```
├── handwriting_recognition.ipynb  # Main Jupyter notebook with complete workflow
├── utils.py                       # Utility functions for model loading and image preprocessing
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── data/                          # Directory for storing datasets
├── models/                        # Directory for saving trained models
└── venv/                          # Python virtual environment
```

## Installation

1. **Clone the repository** (if applicable) or ensure you have the project files

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Dependencies

- TensorFlow 2.15.0 (with Keras)
- NumPy
- Matplotlib
- OpenCV-Python
- Pillow (PIL)
- Scikit-learn
- IPython Widgets

## Usage

### Running the Application

1. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook handwriting_recognition.ipynb
   ```

2. **Follow the notebook workflow**:
   - **Data Loading**: Load and preprocess MNIST dataset
   - **Model Building**: Create and compile CNN architecture
   - **Training**: Train the model with appropriate hyperparameters
   - **Evaluation**: Assess model performance on test data
   - **Prediction**: Test model on custom handwritten digit images

### Key Functions

#### From `utils.py`:
- `load_saved_model(model_path)`: Load a previously trained model
- `preprocess_custom_image(image_path)`: Prepare custom images for prediction
- `plot_prediction_comparison(model, x_test, y_test, num_samples)`: Visualize predictions

#### From the notebook:
- `load_and_preprocess_data()`: Load MNIST data with preprocessing
- `visualize_sample_data()`: Display sample digits from the dataset

## Model Architecture

The application uses a Convolutional Neural Network with:
- Convolutional layers for feature extraction
- Max pooling for dimensionality reduction
- Dense layers for classification
- Dropout for regularization
- Softmax output for digit classification (0-9)

## Data

The model is trained on the MNIST dataset:
- **Training samples**: 60,000 handwritten digits
- **Test samples**: 10,000 handwritten digits
- **Image size**: 28x28 grayscale pixels
- **Classes**: 10 (digits 0-9)

## Custom Image Prediction

To predict on your own handwritten digits:

1. Save your digit image as a PNG/JPG file
2. Use the `preprocess_custom_image()` function to prepare the image
3. Load your trained model
4. Make predictions using `model.predict()`

## Performance

Typical performance metrics (may vary based on training):
- **Accuracy**: ~98-99% on test set
- **Loss**: Cross-entropy loss function
- **Optimizer**: Adam optimizer

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open-source. Please check the license file for details.

## Requirements

- Python 3.7+
- TensorFlow 2.15.0 compatible hardware (CPU/GPU)
- At least 4GB RAM recommended
- Sufficient storage for datasets and models
