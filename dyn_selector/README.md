# Heterogeneity Management for Edge-Based Federated Learning

**Authors:** XXXX-1, XXXX-2 and XXXX-3

**Abstract:** Federated machine learning, often referred to as Federated Learning (FL), is a convenient way to distribute machine learning on different nodes. FL saves bandwidth and keeps data privacy, but requires dedicated solutions to manage
heterogeneity —as for both managed data and selected peers to obtain convergent and robust models. This paper explores
various techniques for node selection, including random selection (as implemented in FedAvg), dynamic sampling, and the Power
of Choice along with its two variants. It also examines different workload optimizers such as the Static Optimizer, Uniform Optimizer, Round Time Optimizer, and Equal Computation Time Optimizer. In addition to implementing them in Flower, a flexible user-friendly FL framework, we carried out a careful evaluation to analyze and compare them. We evaluated the performance and impact of these algorithms compared to baseline techniques on MNIST data set and an MLP model and with different degrees of heterogeneity in the setting. The results obtained provide interesting insights on their effectiveness, convergence speed, and stability. The goal is also to encourage the community to extend the experiments and play with different strategies, features, and characteristics.



## About the experiments

****Dataset:**** MNIST from Keras

****Hardware Setup:**** These experiments were run on a desktop machine with 10 CPU threads. Any machine with 4 CPU cores or more would be able to run it in a reasonable amount of time. Note: the entire experiment runs on CPU-only mode.


## Experimental Setup

****Model:**** This directory implements the model:
* A Multi Layer Perceptron (MLP) used in the paper on MNIST. 
This is the model used by default.


****Dataset:**** The experiments include one dataset: MINST (MNIST). which is partitioned by default among 100 clients, creating imbalanced non-iid partitions using Latent Dirichlet Allocation (LDA) without resampling. All the clients have the same number of samples. Parameter `alpha` of the LDA can be set in the `base.yaml` or passed as argument, by default it is set to 2.

| Dataset | #classes | #partitions | partitioning method | partition settings |
| :------ | :---: | :---: | :---: | :---: |
| MNIST | 10 | 100 | Latent Dirichlet Allocation | All clients with same number of samples |



****Training Hyperparameters:**** 
| Hyperparameter | Description | Default Value |
| ---- | ----------- | ----- |
| `num_clients` | Number of total clients | 100 |
| `batch_size` | Batch size | 32 |
| `local_epochs` | Number of epochs during training | 4 |
| `decay_coefficient` | Decay coefficient used in dynamic sampling | 0.1 |
| `initial_sampling_rate` | Initial fraction of clients sampled | 0.2 |


## Environment Setup
By default, Poetry will use the Python version in your system. 
In some settings, you might want to specify a particular version of Python 
to use inside your Poetry environment. You can do so with `pyenv`. 
Check the documentation for the different ways of installing `pyenv`,
but one easy way is using the automatic installer:

```bash
curl https://pyenv.run | bash
```
You can then install any Python version with `pyenv install <python-version>`
(e.g. `pyenv install 3.10.6`) and set that version as the one to be used. 
```bash
# cd to your dyn_selector directory (i.e. where the `pyproject.toml` is)
pyenv install 3.10.6

pyenv local 3.10.6

# set that version for poetry
poetry env use 3.10.6
```
To build the Python environment as specified in the `pyproject.toml`, use the following commands:
```bash
# cd to your dyn_selector directory (i.e. where the `pyproject.toml` is)

# install the base Poetry environment
poetry install

# activate the environment
poetry shell
```

## Running the Experiments

First ensure you have activated your Poetry environment (execute `poetry shell` from this directory).

### Generate Clients' Dataset
First (and just the first time), the data partitions of clients must be generated.

To generate the partitions for the MNIST dataset (used in Figure 4 of the paper), run the following command:

```bash
# this will generate the datasets using the default settings in the `conf/base.yaml`
python -m dyn_selector.dataset_preparation
```

The generated datasets will be saved in the `mnist` folder.

If you want to modify the `alpha` parameter used to create the LDA partitions, you can override the parameter:

```bash
python -m dyn_selector.dataset_preparation alpha=<alpha>
```

### Running simulations and reproducing results
If you have not done it yet, [generate the clients' dataset](#generate-clients-dataset).


#### MLP on MNIST 

The default configuration for `dyn_selector.main` uses the dynamic sampling selector strategy with MLP on MNIST dataset. It can be run with the following:

```bash
python -m dyn_selector.main # this will run using the default settings in the `conf/config.yaml`
```

You can override settings directly from the command line in this way:

```bash
python -m dyn_selector.main num_rounds=100 # will set the number of rounds to 100
```

## Expected Results

This directory can reproduce the results for the experiments using dynamic sampling selector.

### Dynamic Sampling selector

```bash
# This will run the experiment using Dynamic Sampling strategy
python -m dyn_selector.main
```

### Plotting the results

The above commands would generate results by creating a directory under the following path `outputs/<date>/<hour-minutes-seconds>/<dataset_name>_${variant}_d${strategy.d}_CK${strategy.ck}`, containing a `results.pkl` file that you can plot by using the following command:

```bash
# This will plot a set of results in the same figure. 
python -m dyn_selector.plot_from_pickle --metrics-type="paper_metrics" <paths_to_results>
```
