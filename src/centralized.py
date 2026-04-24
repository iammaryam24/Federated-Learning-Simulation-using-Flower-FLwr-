"""
Centralized Training Implementation for Baseline Comparison
"""

import tensorflow as tf
import numpy as np
import logging
import time
import json
from model import FederatedModel
from data_loader import FederatedDataLoader

logger = logging.getLogger(__name__)

class CentralizedTrainer:
    """Train a model using traditional centralized approach"""
    
    def __init__(self, model=None, epochs=50, batch_size=32, learning_rate=0.001):
        """
        Initialize centralized trainer
        
        Args:
            model: Keras model to train
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.history = []
        self.model = model if model else self._create_model()
    
    def _create_model(self):
        """Create and compile the model"""
        model = FederatedModel.create_model()
        model = FederatedModel.compile_model(model, self.learning_rate)
        return model
    
    def train(self, train_dataset, val_dataset, test_dataset, verbose=1):
        """
        Train the model in centralized fashion
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            test_dataset: Test dataset
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        logger.info(f"Starting centralized training for {self.epochs} epochs")
        start_time = time.time()
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6
            ),
            tf.keras.callbacks.CSVLogger('results/centralized_training_log.csv')
        ]
        
        # Train the model
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.epochs,
            callbacks=callbacks,
            verbose=verbose
        )
        
        end_time = time.time()
        training_time = end_time - start_time
        
        logger.info(f"Centralized training completed in {training_time:.2f} seconds")
        
        # Evaluate on test set
        test_loss, test_accuracy = self.model.evaluate(test_dataset, verbose=0)
        logger.info(f"Test accuracy: {test_accuracy:.4f}, Test loss: {test_loss:.4f}")
        
        # Store history
        self.history = [
            {
                'epoch': i + 1,
                'accuracy': float(acc),
                'loss': float(loss),
                'val_accuracy': float(val_acc),
                'val_loss': float(val_loss)
            }
            for i, (acc, loss, val_acc, val_loss) in enumerate(zip(
                history.history['accuracy'],
                history.history['loss'],
                history.history['val_accuracy'],
                history.history['val_loss']
            ))
        ]
        
        results = {
            'training_time': training_time,
            'epochs': self.epochs,
            'final_accuracy': float(test_accuracy),
            'final_loss': float(test_loss),
            'history': self.history
        }
        
        return results, self.model
    
    def get_model_parameters(self):
        """Get model parameters"""
        return self.model.get_weights()
    
    def set_model_parameters(self, parameters):
        """Set model parameters"""
        self.model.set_weights(parameters)


def run_centralized_experiment(epochs=50, batch_size=32):
    """
    Run a complete centralized training experiment
    
    Args:
        epochs: Number of training epochs
        batch_size: Training batch size
        
    Returns:
        Training results
    """
    # Load data
    data_loader = FederatedDataLoader(batch_size=batch_size)
    x_train, y_train, x_test, y_test = data_loader.load_mnist_data()
    train_dataset, val_dataset, test_dataset = data_loader.get_centralized_data(
        x_train, y_train, x_test, y_test
    )
    
    # Create and train model
    trainer = CentralizedTrainer(epochs=epochs, batch_size=batch_size)
    results, model = trainer.train(train_dataset, val_dataset, test_dataset)
    
    # Save model
    model.save('results/centralized_model.h5')
    
    # Save results
    import json
    with open('results/centralized_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results