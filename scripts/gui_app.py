import sys
import os
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                           QPushButton, QLabel, QFileDialog)
from PyQt5.QtGui import QPixmap, QPainter, QPen, QImage
from PyQt5.QtCore import Qt, QPoint
from tensorflow.keras.models import load_model
import cv2

class DrawingCanvas(QWidget):
    def __init__(self):
        super().__init__()
        self.drawing = False
        self.last_point = QPoint()
        self.image = QImage(280, 280, QImage.Format_RGB32)
        self.image.fill(Qt.white)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = event.pos()

    def mouseMoveEvent(self, event):
        if self.drawing:
            painter = QPainter(self.image)
            painter.setPen(QPen(Qt.black, 3, Qt.SolidLine, Qt.RoundCap))
            painter.drawLine(self.last_point, event.pos())
            self.last_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(0, 0, self.image)

    def clear(self):
        self.image.fill(Qt.white)
        self.update()

    def get_image(self):
        # Convert QImage to numpy array
        img = self.image.convertToFormat(QImage.Format_Grayscale8)
        width = img.width()
        height = img.height()
        ptr = img.bits()
        ptr.setsize(height * width)
        arr = np.array(ptr).reshape(height, width)

        # Resize to 28x28 and invert colors (white background to black, black lines to white)
        arr = cv2.resize(arr, (28, 28))
        arr = 255 - arr  # Invert colors
        return arr

class QuickDrawGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.load_model()

    def initUI(self):
        self.setWindowTitle('Quick, Draw! Classifier')
        self.setGeometry(100, 100, 400, 500)

        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Drawing canvas
        self.canvas = DrawingCanvas()
        layout.addWidget(self.canvas)

        # Clear button
        clear_btn = QPushButton('Clear', self)
        clear_btn.clicked.connect(self.canvas.clear)
        layout.addWidget(clear_btn)

        # Predict button
        predict_btn = QPushButton('Predict', self)
        predict_btn.clicked.connect(self.predict_drawing)
        layout.addWidget(predict_btn)

        # Prediction label
        self.pred_label = QLabel('Draw something and click Predict!', self)
        self.pred_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.pred_label)

    def load_model(self):
        try:
            self.model = load_model('models/best_model.h5')
            self.class_names = np.load('datasets/label_classes.npy')
        except Exception as e:
            print(f"Error loading model: {e}")
            self.pred_label.setText("Error: Model not found!")

    def predict_drawing(self):
        # Get the drawing and preprocess it
        img = self.canvas.get_image()

        # Normalize and reshape for model input
        img = img.reshape(1, 28, 28, 1).astype('float32') / 255.0

        # Make prediction
        try:
            predictions = self.model.predict(img)
            pred_idx = np.argmax(predictions[0])
            pred_class = self.class_names[pred_idx]
            confidence = predictions[0][pred_idx] * 100

            # Display prediction
            self.pred_label.setText(f'Prediction: {pred_class}\nConfidence: {confidence:.2f}%')
        except Exception as e:
            self.pred_label.setText(f"Error making prediction: {str(e)}")

def main():
    app = QApplication(sys.argv)
    gui = QuickDrawGUI()
    gui.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()

