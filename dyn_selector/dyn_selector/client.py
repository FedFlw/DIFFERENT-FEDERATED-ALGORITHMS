from typing import Callable
from models import create_CNN_model, create_MLP_model
import tensorflow as tf
import flwr as fl
import numpy as np
import math

class FlwrClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train) -> None:
        super().__init__()
        self.model = model

        # Split train and validation sets (90% train, 10% validation)
        split_idx = math.floor(len(x_train) * 0.9)
        self.x_train, self.y_train = x_train[:split_idx], y_train[:split_idx]
        self.x_val, self.y_val = x_train[split_idx:], y_train[split_idx:]

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        # Retrieve configuration values or use defaults
        epochs = config.get("local_epochs", 2)  # Default to 2 epochs if not set
        batch_size = config.get("batch_size", 32)  # Default to batch size 32 if not set
        fraction_samples = config.get("fraction_samples", 1.0)  # Use all samples by default

        # Select subset of the training data based on fraction_samples
        num_samples = round(len(self.x_train) * fraction_samples)

        # Ensure a minimum number of samples to avoid zero samples issue
        if num_samples == 0:
            print(f"Client has too few samples with fraction_samples={fraction_samples}, setting to at least 1 sample.")
            num_samples = 1  # Ensure at least 1 sample

        idx = np.random.choice(len(self.x_train), num_samples, replace=False)
        x_train_selected = self.x_train[idx]
        y_train_selected = self.y_train[idx]

        # Print training details for debugging
        print(f"Client training on {len(x_train_selected)} samples, {epochs} epochs, batch size {batch_size}")

        # Ensure data is not empty
        if len(x_train_selected) == 0 or len(y_train_selected) == 0:
            raise ValueError("Client has no data to train on.")

        # Set the initial model weights
        self.model.set_weights(parameters)

        # Compile model with eager execution to track errors in detail
        self.model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"], run_eagerly=True)

        # Train the model
        history = self.model.fit(x_train_selected, y_train_selected, batch_size=batch_size, epochs=epochs, verbose=2)

        # Return updated model parameters and metrics
        return self.model.get_weights(), len(self.x_train), {"training_loss": history.history["loss"][-1]}

    def evaluate(self, parameters, config):
        # Set the current model weights for evaluation
        self.model.set_weights(parameters)

        # Evaluate the model on validation data
        loss, acc = self.model.evaluate(self.x_val, self.y_val, verbose=2)

        # Return evaluation loss, sample count, and accuracy
        return loss, len(self.x_val), {"accuracy": acc}


def gen_client_fn(fds) -> Callable[[str], fl.client.Client]:
    """Generate a Flower client function that creates a client instance using FederatedDataset."""
    
    def client_fn(cid: str) -> fl.client.Client:
        # Load dataset partition for this client using the FederatedDataset (fds)
        (x_train_cid, y_train_cid) = load_dataset(cid, is_cnn)

        if x_train_cid is None or y_train_cid is None:
            raise ValueError(f"Dataset for client {cid} not loaded correctly.")

        # Print dataset shape for debugging
        print(f"Client {cid} loaded data: x_train={x_train_cid.shape}, y_train={y_train_cid.shape}")

        # Create the model (you can choose CNN or MLP based on some logic or config if needed)
        model = create_MLP_model()  # Modify if CNN is required

        # Compile the model
        model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"], run_eagerly=True)

        # Return the client instance
        return FlwrClient(model, x_train_cid, y_train_cid)

    return client_fn



