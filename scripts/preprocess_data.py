import os
import json
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

class QuickDrawPreprocessor:
    def __init__(self, input_folder, output_folder):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.label_encoder = LabelEncoder()
        
    def parse_drawing(self, drawing, image_size=256):
        """Convert drawing strokes into a numpy array."""
        image = np.zeros((image_size, image_size), dtype=np.uint8)
        for stroke in drawing:
            points = np.array(stroke).T
            points = points.astype(np.int32)
            for i in range(len(points) - 1):
                cv2.line(image, tuple(points[i]), tuple(points[i + 1]), 255, 2)
        return cv2.resize(image, (28, 28))

    def process_ndjson(self, file_path):
        """Process a single .ndjson file."""
        images, labels = [], []
        with open(file_path, 'r') as f:
            data = [json.loads(line) for line in tqdm(f, desc=f"Reading {os.path.basename(file_path)}")]
            
        for item in tqdm(data, desc="Processing drawings"):
            if item.get('recognized', False):
                try:
                    img = self.parse_drawing(item['drawing'])
                    images.append(img)
                    labels.append(item['word'])
                except Exception as e:
                    print(f"Error processing drawing: {e}")
                    
        return np.array(images), np.array(labels)

    def process_all_files(self):
        """Process all .ndjson files in the input folder."""
        os.makedirs(self.output_folder, exist_ok=True)
        
        all_images, all_labels = [], []
        ndjson_files = [f for f in os.listdir(self.input_folder) if f.endswith('.ndjson')]
        
        for file_name in ndjson_files:
            file_path = os.path.join(self.input_folder, file_name)
            print(f"\nProcessing {file_name}")
            
            images, labels = self.process_ndjson(file_path)
            all_images.extend(images)
            all_labels.extend(labels)

        # Convert lists to numpy arrays
        all_images = np.array(all_images)
        all_labels = np.array(all_labels)
        
        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(all_labels)
        
        # Split into train and validation sets
        indices = np.random.permutation(len(all_images))
        split_idx = int(len(indices) * 0.8)  # 80% training, 20% validation
        
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        # Save training data
        np.savez_compressed(
            os.path.join(self.output_folder, 'train_data.npz'),
            images=all_images[train_indices],
            labels=encoded_labels[train_indices]
        )
        
        # Save validation data
        np.savez_compressed(
            os.path.join(self.output_folder, 'val_data.npz'),
            images=all_images[val_indices],
            labels=encoded_labels[val_indices]
        )
        
        # Save label encoder classes
        np.save(
            os.path.join(self.output_folder, 'label_classes.npy'),
            self.label_encoder.classes_
        )

def main():
    input_folder = "quickdraw_data"
    output_folder = "datasets"
    
    preprocessor = QuickDrawPreprocessor(input_folder, output_folder)
    preprocessor.process_all_files()
    print("Preprocessing complete!")

if __name__ == "__main__":
    main()