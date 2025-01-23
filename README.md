# Quick, Draw! Classifier

This project implements a machine learning model to classify hand-drawn sketches using the Quick, Draw! dataset. It includes data preprocessing, model training, evaluation, and a GUI application for testing the model.

## Project Structure

```
quickdraw_project/
│
├── quickdraw_data/         # Raw .ndjson files from Quick, Draw! dataset
├── datasets/               # Processed datasets
├── models/                 # Saved models and visualizations
├── scripts/               
│   ├── preprocess_data.py  # Data preprocessing
│   ├── train_model.py      # Model training
│   ├── evaluate_model.py   # Model evaluation
│   └── gui_app.py          # GUI application
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## Setup and Installation

1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download Quick, Draw! dataset files (.ndjson format) and place them in the `quickdraw_data` folder.

## Usage

1. Preprocess the data:
   ```bash
   python3 scripts/preprocess_data.py
   ```

2. Train the model:
   ```bash
   python3 scripts/train_model.py
   ```

3. Evaluate the model:
   ```bash
   python3 scripts/evaluate_model.py
   ```

4. Run the GUI application:
   ```bash
   python3 scripts/gui_app.py
   ```

## Model Architecture

The model uses a CNN architecture with:
- Multiple convolutional layers with batch normalization
- Max pooling layers
- Dropout for regularization
- Dense layers for classification

## Performance

The model's performance metrics and visualizations will be saved in the `models` directory after evaluation:
- confusion_matrix.png: Visualization of model predictions
- prediction_samples.png: Sample predictions on validation data
- training_history.png: Training and validation metrics over time

## GUI Application

The GUI application provides a simple interface to:
- Draw sketches
- Get real-time predictions
- Clear the canvas
- View prediction confidence

## Dependencies

- TensorFlow
- OpenCV
- NumPy
- PyQt5
- scikit-learn
- matplotlib
- tqdm

## License

This project is licensed under the MIT License - see the LICENSE file for details.