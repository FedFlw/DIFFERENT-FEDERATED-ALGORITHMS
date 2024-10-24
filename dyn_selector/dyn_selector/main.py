import os
from typing import Dict, List, Optional, Tuple
from flwr.common.typing import Metrics
from utils import save_results_as_pickle
import numpy as np
from client_uo import gen_client_fn
from models import create_MLP_model
import hydra
from hydra.utils import instantiate
import flwr as fl
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the baseline."""
    # 1. Print parsed config
    print(OmegaConf.to_yaml(cfg))

    # 2. Prepare your dataset
    # (Dataset preparation code here)

    # 3. Define your clients
    client_fn = gen_client_fn((cfg.epochs_min, cfg.epochs_max), 
                              (cfg.fraction_samples_min, cfg.fraction_samples_max), 
                              (cfg.batch_size_min, cfg.batch_size_max), 
                              cfg.num_clients
                              )
    
    def get_fit_metrics_aggregation_fn():
        def fit_metrics_aggregation_fn(results: List[Tuple[int, Metrics]]) -> Metrics:
            # Initialize lists to store training losses
            training_losses = []
            for _, metrics in results:
                if "training_loss" in metrics:
                    training_loss = metrics["training_loss"]
                    training_losses.append(training_loss)

            # Calculate variance and average training loss
            variance_training_loss = np.var(training_losses)
            average_training_loss = np.mean(training_losses)

            aggregated_metrics = {
                "variance_training_loss": variance_training_loss,
                "average_training_loss": average_training_loss,
            }

            return aggregated_metrics

        return fit_metrics_aggregation_fn

    def get_evaluate_fn(model):
        """Return an evaluation function for server-side evaluation."""
        test_folder = cfg.dataset.dataset
        x_test = np.load(os.path.join(test_folder, "x_test.npy"))
        y_test = np.load(os.path.join(test_folder, "y_test.npy"))

        def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
            model.set_weights(parameters)
            loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
            return loss, {"accuracy": accuracy}

        return evaluate
    
    server_model = create_MLP_model()

    server_model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

    # 4. Define your strategy
    strategy = instantiate(
        cfg.strategy,
        evaluate_fn=get_evaluate_fn(server_model),
        fit_metrics_aggregation_fn=get_fit_metrics_aggregation_fn(),
    )

    # 5. Start Simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        client_resources={"num_cpus": 1},
        config=fl.server.ServerConfig(cfg.num_rounds),
        strategy= strategy,
        ray_init_args={"ignore_reinit_error": True, "include_dashboard": False},
    )

    # 6. Save your results
    print("................")
    print(f"History object: {history}")  # Print history object to ensure it's populated

    # Hydra automatically creates an output directory
    save_path = HydraConfig.get().runtime.output_dir
    print(f"Saving results to {save_path}")  # Print the save path

    # Save results as a Python pickle using the file_path
    save_results_as_pickle(history, file_path=save_path, extra_results={})


if __name__ == "__main__":
    main()
