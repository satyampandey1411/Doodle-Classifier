# import os
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from sklearn.metrics import classification_report, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns

# class QuickDrawEvaluator:
#     def __init__(self, model_path, data_dir):
#         self.model = load_model(model_path)
#         self.data_dir = data_dir
#         self.load_data()
        
#     def load_data(self):
#         """Load validation data and class labels."""
#         val_data = np.load(os.path.join(self.data_dir, 'val_data.npz'))
#         self.x_val = val_data['images'].reshape(-1, 28, 28, 1).astype('float32') / 255.0
#         self.y_val = val_data['labels']
#         self.class_names = np.load(os.path.join(self.data_dir, 'label_classes.npy'))
        
#     def evaluate(self):
#         """Evaluate the model and generate metrics."""
#         # Get predictions
#         y_pred = self.model.predict(self.x_val)
#         y_pred_classes = np.argmax(y_pred, axis=1)
        
#         # Generate classification report
#         report = classification_report(
#             self.y_val, 
#             y_pred_classes,
#             target_names=self.class_names,
#             output_dict=True
#         )
        
#         # Generate confusion matrix
#         cm = confusion_matrix(self.y_val, y_pred_classes)
        
#         return report, cm
        
#     def plot_confusion_matrix(self, cm):
#         """Plot confusion matrix as a heatmap."""
#         plt.figure(figsize=(20, 20))
#         sns.heatmap(
#             cm,
#             xticklabels=self.class_names,
#             yticklabels=self.class_names,
#             annot=True,
#             fmt='d',
#             cmap='Blues'
#         )
#         plt.title('Confusion Matrix')
#         plt.xlabel('Predicted')
#         plt.ylabel('True')
#         plt.xticks(rotation=90)
#         plt.yticks(rotation=0)
#         plt.tight_layout()
#         plt.savefig('models/confusion_matrix.png')
#         plt.close()
        
#     def visualize_predictions(self, num_samples=25):
#         """Visualize random samples and their predictions."""
#         indices = np.random.choice(len(self.x_val), num_samples, replace=False)
#         samples = self.x_val[indices]
#         true_labels = self.y_val[indices]
#         predictions = self.model.predict(samples)
#         pred_labels = np.argmax(predictions, axis=1)
        
#         fig, axes = plt.subplots(5, 5, figsize=(15, 15))
#         for i, ax in enumerate(axes.flat):
#             ax.imshow(samples[i].reshape(28, 28), cmap='gray')
#             color = 'green' if true_labels[i] == pred_labels[i] else 'red'
#             ax.set_title(
#                 f'True: {self.class_names[true_labels[i]]}\nPred: {self.class_names[pred_labels[i]]}',
#                 color=color
#             )
#             ax.axis('off')
        
#         plt.tight_layout()
#         plt.savefig('models/prediction_samples.png')
#         plt.close()

# def main():
#     evaluator = QuickDrawEvaluator('models/best_model.h5', 'datasets')
#     report, cm = evaluator.evaluate()
    
#     # Print classification report
#     print("\nClassification Report:")
#     for class_name in evaluator.class_names:
#         metrics = report[class_name]
#         print(f"\n{class_name}:")
#         print(f"Precision: {metrics['precision']:.3f}")
#         print(f"Recall: {metrics['recall']:.3f}")
#         print(f"F1-score: {metrics['f1-score']:.3f}")
    
#     print(f"\nOverall Accuracy: {report['accuracy']:.3f}")
    
#     # Generate visualizations
#     evaluator.plot_confusion_matrix(cm)
#     evaluator.visualize_predictions()
#     print("\nEvaluation complete! Check the models directory for visualizations.")

# if __name__ == "__main__":
#     main()


import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

class QuickDrawEvaluator:
    def __init__(self, model_path, data_dir):
        self.data_dir = data_dir
        self.model_path = model_path
        self.load_model_and_data()
        
    def load_model_and_data(self):
        """Load the trained model and validation data."""
        try:
            print("Loading model...")
            self.model = load_model(self.model_path)
            
            print("Loading validation data...")
            val_data = np.load(os.path.join(self.data_dir, 'val_data.npz'))
            self.x_val = val_data['images'].reshape(-1, 28, 28, 1).astype('float32') / 255.0
            self.y_val = val_data['labels']
            self.class_names = np.load(os.path.join(self.data_dir, 'label_classes.npy'))
            
            print(f"Loaded {len(self.class_names)} classes: {', '.join(self.class_names)}")
            print(f"Validation set size: {len(self.x_val)} images")
            
        except Exception as e:
            raise Exception(f"Error loading model or data: {e}")
        
    def evaluate(self):
        """Evaluate the model and generate metrics."""
        try:
            print("\nEvaluating model...")
            # Get predictions
            y_pred = self.model.predict(self.x_val, verbose=1)
            y_pred_classes = np.argmax(y_pred, axis=1)
            
            # Generate classification report
            report = classification_report(
                self.y_val, 
                y_pred_classes,
                target_names=self.class_names,
                output_dict=True
            )
            
            # Generate confusion matrix
            cm = confusion_matrix(self.y_val, y_pred_classes)
            
            return report, cm
            
        except Exception as e:
            raise Exception(f"Error during evaluation: {e}")
        
    def plot_confusion_matrix(self, cm):
        """Plot confusion matrix as a heatmap."""
        try:
            plt.figure(figsize=(20, 20))
            sns.heatmap(
                cm,
                xticklabels=self.class_names,
                yticklabels=self.class_names,
                annot=True,
                fmt='d',
                cmap='Blues'
            )
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.xticks(rotation=90)
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig('models/confusion_matrix.png')
            plt.close()
            print("Saved confusion matrix plot to models/confusion_matrix.png")
            
        except Exception as e:
            print(f"Warning: Could not save confusion matrix plot: {e}")
        
    def visualize_predictions(self, num_samples=25):
        """Visualize random samples and their predictions."""
        try:
            indices = np.random.choice(len(self.x_val), num_samples, replace=False)
            samples = self.x_val[indices]
            true_labels = self.y_val[indices]
            predictions = self.model.predict(samples)
            pred_labels = np.argmax(predictions, axis=1)
            
            fig, axes = plt.subplots(5, 5, figsize=(15, 15))
            for i, ax in enumerate(axes.flat):
                ax.imshow(samples[i].reshape(28, 28), cmap='gray')
                color = 'green' if true_labels[i] == pred_labels[i] else 'red'
                ax.set_title(
                    f'True: {self.class_names[true_labels[i]]}\nPred: {self.class_names[pred_labels[i]]}',
                    color=color,
                    fontsize=8
                )
                ax.axis('off')
            
            plt.tight_layout()
            plt.savefig('models/prediction_samples.png')
            plt.close()
            print("Saved prediction samples plot to models/prediction_samples.png")
            
        except Exception as e:
            print(f"Warning: Could not save prediction samples plot: {e}")

def main():
    try:
        print("Starting model evaluation...")
        evaluator = QuickDrawEvaluator('models/best_model.h5', 'datasets')
        
        # Evaluate model
        report, cm = evaluator.evaluate()
        
        # Print classification report
        print("\nClassification Report:")
        for class_name in evaluator.class_names:
            metrics = report[class_name]
            print(f"\n{class_name}:")
            print(f"Precision: {metrics['precision']:.3f}")
            print(f"Recall: {metrics['recall']:.3f}")
            print(f"F1-score: {metrics['f1-score']:.3f}")
        
        print(f"\nOverall Accuracy: {report['accuracy']:.3f}")
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        evaluator.plot_confusion_matrix(cm)
        evaluator.visualize_predictions()
        
        print("\nEvaluation complete! Check the models directory for visualizations.")
        
    except Exception as e:
        print(f"Error during evaluation process: {str(e)}")
        raise

if __name__ == "__main__":
    main()