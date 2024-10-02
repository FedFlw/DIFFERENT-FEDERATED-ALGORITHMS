from typing import Callable
from models import create_CNN_model, create_MLP_model
from dataset import load_dataset
import flwr as fl
import numpy as np
import math

class FlwrClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train) -> None:
        super().__init__()
        self.model = model
        # Split the training data, using 10% for validation
        split_idx = math.floor(len(x_train) * 0.9)
        self.x_train, self.y_train = x_train[:split_idx], y_train[:split_idx]
        self.x_val, self.y_val = x_train[split_idx:], y_train[split_idx:]

    def get_parameters(self, config):
        """Return the model weights."""
        return self.model.get_weights()

    def set_parameters(self, parameters):
        """Set the model weights received from the server."""
        self.model.set_weights(parameters)

    def fit(self, parameters, config):
        """Train the model using the client's local dataset."""
        # Set the model weights to the received global parameters
        self.set_parameters(parameters)

        # Retrieve training configuration
        epochs = config.get("local_epochs", 2)
        batch_size = config.get("batch_size", 32)

        print(f"Client training on {len(self.x_train)} samples for {epochs} epochs with batch size {batch_size}")

        # Train the model
        history = self.model.fit(self.x_train, self.y_train, batch_size=batch_size, epochs=epochs, verbose=2)

        # Return updated model weights and training metrics
        return self.get_parameters(config), len(self.x_train), {"training_loss": history.history["loss"][-1]}

    def evaluate(self, parameters, config):
        """Evaluate the model on the client's validation set."""
        self.set_parameters(parameters)

        if len(self.x_val) == 0:
            print(f"Client has 0 validation samples, skipping evaluation.")
            return float('inf'), 0, {"accuracy": None}

        loss, acc = self.model.evaluate(self.x_val, self.y_val, verbose=2)
        return loss, len(self.x_val), {"accuracy": acc}


def gen_client_fn(is_cnn: bool = False) -> Callable[[str], fl.client.Client]:
    """Create a function that generates clients with either CNN or MLP model."""
    def client_fn(cid: str) -> fl.client.Client:
        # Dynamically load the model based on the `is_cnn` flag
        if is_cnn:
            model = create_CNN_model()  # Use CNN model
        else:
            model = create_MLP_model()  # Use MLP model

        # Compile the model
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        # Load the client's dataset partition
        (x_train_cid, y_train_cid) = load_dataset(cid, is_cnn)

        # Create and return the client
        return FlwrClient(model, x_train_cid, y_train_cid)

    return client_fn
