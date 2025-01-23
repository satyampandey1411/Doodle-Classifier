# import os
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras import layers, models, callbacks
# import matplotlib.pyplot as plt
# from tensorflow.keras.utils import Sequence

# # Disable GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# class DataGenerator(Sequence):
#     def __init__(self, data_path, batch_size=32, shuffle=True, max_samples=None):
#         data = np.load(data_path, allow_pickle=True)
        
#         # Limit the number of samples if specified
#         if max_samples is not None:
#             indices = np.random.choice(len(data['images']), max_samples, replace=False)
#             self.images = data['images'][indices]
#             self.labels = data['labels'][indices]
#         else:
#             self.images = data['images']
#             self.labels = data['labels']
            
#         self.batch_size = batch_size
#         self.shuffle = shuffle
#         self.indexes = np.arange(len(self.images))
#         if self.shuffle:
#             np.random.shuffle(self.indexes)

#     def __len__(self):
#         return int(np.floor(len(self.indexes) / self.batch_size))

#     def __getitem__(self, index):
#         indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
#         X = self.images[indexes]
#         y = self.labels[indexes]
#         X = X.reshape(-1, 28, 28, 1).astype('float32') / 255.0
#         return X, y

#     def on_epoch_end(self):
#         if self.shuffle:
#             np.random.shuffle(self.indexes)

# class QuickDrawTrainer:
#     def __init__(self, data_dir, model_dir):
#         self.data_dir = data_dir
#         self.model_dir = model_dir
#         os.makedirs(model_dir, exist_ok=True)

#     def get_data_generators(self, batch_size=32, max_samples=50000):
#         print(f"Loading data with max_samples={max_samples}...")
        
#         train_generator = DataGenerator(
#             os.path.join(self.data_dir, 'train_data.npz'),
#             batch_size=batch_size,
#             max_samples=max_samples
#         )
        
#         val_generator = DataGenerator(
#             os.path.join(self.data_dir, 'val_data.npz'),
#             batch_size=batch_size,
#             shuffle=False,
#             max_samples=max_samples//5  # 20% of training samples
#         )
        
#         # Get number of classes
#         sample_batch = next(iter(train_generator))[1]
#         self.num_classes = len(np.unique(sample_batch))
#         print(f"Number of classes: {self.num_classes}")
#         print(f"Training samples: {len(train_generator.images)}")
#         print(f"Validation samples: {len(val_generator.images)}")
        
#         return train_generator, val_generator

#     def create_model(self):
#         """Create an even more lightweight model"""
#         model = models.Sequential([
#             layers.Input(shape=(28, 28, 1)),
            
#             # Simplified architecture
#             layers.Conv2D(8, (3, 3), padding='same', activation='relu'),
#             layers.MaxPooling2D((2, 2)),
            
#             layers.Conv2D(16, (3, 3), padding='same', activation='relu'),
#             layers.MaxPooling2D((2, 2)),
            
#             layers.Flatten(),
#             layers.Dense(64, activation='relu'),
#             layers.Dropout(0.3),
#             layers.Dense(self.num_classes, activation='softmax')
#         ])
        
#         return model

#     def train(self, epochs=10, batch_size=32, max_samples=50000):
#         """Train the model with reduced epochs and samples"""
#         print("Initializing training...")
#         train_generator, val_generator = self.get_data_generators(batch_size, max_samples)
        
#         print("Creating model...")
#         model = self.create_model()
#         model.compile(
#             optimizer='adam',
#             loss='sparse_categorical_crossentropy',
#             metrics=['accuracy']
#         )
        
#         print("Setting up callbacks...")
#         callbacks_list = [
#             callbacks.EarlyStopping(
#                 monitor='val_loss',
#                 patience=3,
#                 restore_best_weights=True
#             ),
#             callbacks.ModelCheckpoint(
#                 filepath=os.path.join(self.model_dir, 'best_model.h5'),
#                 monitor='val_accuracy',
#                 save_best_only=True
#             )
#         ]
        
#         print("Starting training...")
#         history = model.fit(
#             train_generator,
#             validation_data=val_generator,
#             epochs=epochs,
#             callbacks=callbacks_list,
#             verbose=1
#         )
        
#         self.plot_training_history(history)
#         return model, history

#     def plot_training_history(self, history):
#         plt.figure(figsize=(12, 4))
        
#         plt.subplot(1, 2, 1)
#         plt.plot(history.history['accuracy'], label='Training')
#         plt.plot(history.history['val_accuracy'], label='Validation')
#         plt.title('Model Accuracy')
#         plt.xlabel('Epoch')
#         plt.ylabel('Accuracy')
#         plt.legend()
        
#         plt.subplot(1, 2, 2)
#         plt.plot(history.history['loss'], label='Training')
#         plt.plot(history.history['val_loss'], label='Validation')
#         plt.title('Model Loss')
#         plt.xlabel('Epoch')
#         plt.ylabel('Loss')
#         plt.legend()
        
#         plt.tight_layout()
#         plt.savefig(os.path.join(self.model_dir, 'training_history.png'))
#         plt.close()

# def main():
#     print("Starting the training process...")
    
#     # Training configuration
#     BATCH_SIZE = 32
#     MAX_SAMPLES = 50000  # Limit total samples for faster training
#     EPOCHS = 10
    
#     trainer = QuickDrawTrainer('datasets', 'models')
#     try:
#         print(f"Training model with batch_size={BATCH_SIZE}, max_samples={MAX_SAMPLES}")
#         model, history = trainer.train(
#             epochs=EPOCHS,
#             batch_size=BATCH_SIZE,
#             max_samples=MAX_SAMPLES
#         )
#         print("Training complete!")
#     except Exception as e:
#         print(f"Error during training: {str(e)}")

# if __name__ == "__main__":
#     main()

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
from tensorflow.keras.utils import Sequence

# Disable GPU if causing issues
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class DataGenerator(Sequence):
    def __init__(self, data_path, batch_size=32, shuffle=True, max_samples=None):
        try:
            data = np.load(data_path, allow_pickle=True)
            
            # Limit the number of samples if specified
            if max_samples is not None:
                indices = np.random.choice(len(data['images']), max_samples, replace=False)
                self.images = data['images'][indices]
                self.labels = data['labels'][indices]
            else:
                self.images = data['images']
                self.labels = data['labels']
                
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.indexes = np.arange(len(self.images))
            if self.shuffle:
                np.random.shuffle(self.indexes)
        except Exception as e:
            raise Exception(f"Error initializing DataGenerator: {e}")

    def __len__(self):
        return int(np.floor(len(self.indexes) / self.batch_size))

    def __getitem__(self, index):
        try:
            indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
            X = self.images[indexes]
            y = self.labels[indexes]
            X = X.reshape(-1, 28, 28, 1).astype('float32') / 255.0
            return X, y
        except Exception as e:
            raise Exception(f"Error getting batch {index}: {e}")

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

class QuickDrawTrainer:
    def __init__(self, data_dir, model_dir):
        self.data_dir = data_dir
        self.model_dir = model_dir
        try:
            os.makedirs(model_dir, exist_ok=True)
        except Exception as e:
            raise Exception(f"Error creating model directory: {e}")

    def get_data_generators(self, batch_size=32, max_samples=50000):
        print(f"Loading data with max_samples={max_samples}...")
        
        try:
            train_generator = DataGenerator(
                os.path.join(self.data_dir, 'train_data.npz'),
                batch_size=batch_size,
                max_samples=max_samples
            )
            
            val_generator = DataGenerator(
                os.path.join(self.data_dir, 'val_data.npz'),
                batch_size=batch_size,
                shuffle=False,
                max_samples=max_samples//5  # 20% of training samples
            )
            
            # Get number of classes
            sample_batch = next(iter(train_generator))[1]
            self.num_classes = len(np.unique(sample_batch))
            
            print(f"Number of classes: {self.num_classes}")
            print(f"Training samples: {len(train_generator.images)}")
            print(f"Validation samples: {len(val_generator.images)}")
            
            return train_generator, val_generator
        except Exception as e:
            raise Exception(f"Error creating data generators: {e}")

    def create_model(self):
        """Create a lightweight CNN model"""
        try:
            model = models.Sequential([
                layers.Input(shape=(28, 28, 1)),
                
                # First convolutional block
                layers.Conv2D(8, (3, 3), padding='same', activation='relu'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                
                # Second convolutional block
                layers.Conv2D(16, (3, 3), padding='same', activation='relu'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                
                # Dense layers
                layers.Flatten(),
                layers.Dense(64, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(self.num_classes, activation='softmax')
            ])
            
            return model
        except Exception as e:
            raise Exception(f"Error creating model: {e}")

    def train(self, epochs=10, batch_size=32, max_samples=50000):
        """Train the model with the specified parameters"""
        print("Initializing training...")
        
        try:
            # Get data generators
            train_generator, val_generator = self.get_data_generators(batch_size, max_samples)
            
            # Create and compile model
            print("Creating model...")
            model = self.create_model()
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Set up callbacks
            print("Setting up callbacks...")
            callbacks_list = [
                callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=3,
                    restore_best_weights=True
                ),
                callbacks.ModelCheckpoint(
                    filepath=os.path.join(self.model_dir, 'best_model.h5'),
                    monitor='val_accuracy',
                    save_best_only=True
                ),
                callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=2,
                    min_lr=1e-6
                )
            ]
            
            # Train the model
            print("Starting training...")
            history = model.fit(
                train_generator,
                validation_data=val_generator,
                epochs=epochs,
                callbacks=callbacks_list,
                verbose=1
            )
            
            # Save training history plot
            self.plot_training_history(history)
            
            # Save final model
            model.save(os.path.join(self.model_dir, 'final_model.h5'))
            
            return model, history
            
        except Exception as e:
            raise Exception(f"Error during training: {e}")

    def plot_training_history(self, history):
        """Plot and save training history"""
        try:
            # Create figure with subplots
            plt.figure(figsize=(12, 4))
            
            # Plot accuracy
            plt.subplot(1, 2, 1)
            plt.plot(history.history['accuracy'], label='Training')
            plt.plot(history.history['val_accuracy'], label='Validation')
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            
            # Plot loss
            plt.subplot(1, 2, 2)
            plt.plot(history.history['loss'], label='Training')
            plt.plot(history.history['val_loss'], label='Validation')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            # Save plot
            plt.tight_layout()
            plt.savefig(os.path.join(self.model_dir, 'training_history.png'))
            plt.close()
            
            # Save history to file
            history_dict = {
                'accuracy': history.history['accuracy'],
                'val_accuracy': history.history['val_accuracy'],
                'loss': history.history['loss'],
                'val_loss': history.history['val_loss']
            }
            np.save(os.path.join(self.model_dir, 'training_history.npy'), history_dict)
            
        except Exception as e:
            print(f"Warning: Could not save training history: {e}")

def main():
    print("Starting the training process...")
    
    # Training configuration
    BATCH_SIZE = 32
    MAX_SAMPLES = 50000  # Limit total samples for faster training
    EPOCHS = 10
    
    try:
        # Create trainer instance
        trainer = QuickDrawTrainer('datasets', 'models')
        
        # Train model
        print(f"Training model with batch_size={BATCH_SIZE}, max_samples={MAX_SAMPLES}")
        model, history = trainer.train(
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            max_samples=MAX_SAMPLES
        )
        
        # Print final metrics
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        print("\nTraining completed successfully!")
        print(f"Final training accuracy: {final_train_acc:.4f}")
        print(f"Final validation accuracy: {final_val_acc:.4f}")
        
    except Exception as e:
        print(f"Error during training process: {str(e)}")
        raise

if __name__ == "__main__":
    main()