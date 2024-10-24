# Heterogeneity Management for Edge-Based Federated Learning
HETEROGENEITY ANALYSIS IN FEDERATED LEARNING
**Abstract:** Federated machine learning, often referred to as Federated Learning (FL), is a convenient way to distribute machine learning on different nodes. FL saves bandwidth and keeps data privacy, but requires dedicated solutions to manage
heterogeneity —as for both managed data and selected peers to obtain convergent and robust models. This paper explores
various techniques for node selection, including random selection (as implemented in FedAvg), dynamic sampling, and the Power
of Choice along with its two variants. It also examines different workload optimizers such as the Static Optimizer, Uniform Optimizer, Round Time Optimizer, and Equal Computation Time Optimizer. In addition to implementing them in Flower, a flexible user-friendly FL framework, we carried out a careful evaluation to analyze and compare them. We evaluated the performance and impact of these algorithms compared to baseline techniques on MNIST data set and an MLP model and with different degrees of heterogeneity in the setting. The results obtained provide interesting insights on their effectiveness, convergence speed, and stability. The goal is also to encourage the community to extend the experiments and play with different strategies, features, and characteristics.


Directory Structure
dyn_selector directory: Contains the code for the experiments using dynamic selector from paper Ji et al. 2021.
power_of_choice directory: Contains the code for the experiments using the Power of Choice family of selectors, based on the paper by YJ Cho et al. 2022.
In each subdirectory, there is a README.md file explaining how to setup and replicate the experiments.
