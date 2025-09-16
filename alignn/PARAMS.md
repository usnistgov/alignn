# ALIGNN Training Parameters Documentation

This document provides detailed descriptions of parameters used for training ALIGNN (Atomistic Line Graph Neural Network) models.

## Basic Training Parameters

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `batch_size` | Number of samples per batch during training | 16 |
| `criterion` | Loss function used for training (e.g., 'l1', 'mse') | 'l1' |
| `epochs` | Total number of training epochs | 30 |
| `learning_rate` | Initial learning rate for optimizer | 0.001 |
| `optimizer` | Optimization algorithm ('adam', 'adamw', etc.) | 'adamw' |
| `scheduler` | Learning rate scheduler type ('onecycle', 'none', etc.) | 'onecycle' |
| `warmup_steps` | Number of warmup steps for learning rate | 2000 |
| `weight_decay` | L2 regularization factor | 1e-05 |
| `random_seed` | Seed for random number generators to ensure reproducibility | 123 |
| `progress` | Whether to display progress bars during training | True |

## Data Configuration

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `dataset` | Dataset name or type | 'user_data' |
| `filename` | Base filename for saved models and outputs | 'sample' |
| `target` | Property to predict (column name in dataset) | 'target' |
| `id_tag` | Identifier column in the dataset | 'jid' |
| `atom_features` | Type of atom features to use | 'atomic_number' |
| `n_train` | Number of training samples | 214 |
| `n_val` | Number of validation samples | 25 |
| `n_test` | Number of test samples | 25 |
| `train_ratio` | Fraction of data used for training | 0.9 |
| `val_ratio` | Fraction of data used for validation | 0.05 |
| `test_ratio` | Fraction of data used for testing | 0.05 |
| `keep_data_order` | Whether to maintain the original order of data | True |
| `use_canonize` | Whether to canonize molecular graphs | True |
| `standard_scalar_and_pca` | Apply standardization and PCA to input features | False |
| `target_multiplication_factor` | Factor to multiply target values | None |

## Graph Construction Parameters

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `cutoff` | Cutoff radius for constructing graphs (Å) | 5.0 |
| `cutoff_extra` | Secondary cutoff for special interactions (Å) | 3.0 |
| `max_neighbors` | Maximum number of neighbors per atom | 12 |
| `neighbor_strategy` | Method to construct the graph ('radius_graph', 'k_nearest') | 'radius_graph' |
| `compute_line_graph` | Whether to precompute line graphs | False |

## Hardware & Performance

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `device` | Computing device ('cpu', 'cuda', etc.) | 'cpu' |
| `distributed` | Whether to use distributed training | False |
| `data_parallel` | Whether to use DataParallel for multi-GPU training | False |
| `dtype` | Data type for tensors ('float32', 'float64', etc.) | 'float32' |
| `num_workers` | Number of data loading worker processes | 0 |
| `pin_memory` | Whether to use pinned memory for faster GPU transfer | False |
| `use_lmdb` | Use LMDB database for faster data loading | True |

## Output & Checkpointing

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `output_dir` | Directory for saving outputs | 'OutputDir' |
| `write_checkpoint` | Whether to save model checkpoints | True |
| `write_predictions` | Whether to save model predictions | True |
| `store_outputs` | Whether to store all outputs during training | False |
| `save_dataloader` | Whether to save the dataloader for future use | False |
| `log_tensorboard` | Whether to log metrics to TensorBoard | False |
| `n_early_stopping` | Number of epochs without improvement before early stopping | None |

## Classification Settings

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `classification_threshold` | Threshold for binary classification | None |
| `normalize_graph_level_loss` | Whether to normalize graph-level loss | False |

## Model Architecture Parameters

The `model` dictionary contains parameters specific to the ALIGNN architecture:

### Core Architecture

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `name` | Model architecture name | 'alignn_atomwise' |
| `gcn_layers` | Number of graph convolution layers | 2 |
| `alignn_layers` | Number of ALIGNN layers | 2 |
| `hidden_features` | Size of hidden feature vectors | 64 |
| `embedding_features` | Size of embedding feature vectors | 64 |
| `output_features` | Size of output feature vectors | 1 |
| `atom_input_features` | Number of atom input features | 1 |
| `edge_input_features` | Number of edge input features | 80 |
| `triplet_input_features` | Number of triplet input features | 40 |

### Atomistic Properties and Forces

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `calculate_gradient` | Whether to calculate force gradients | True |
| `add_reverse_forces` | Whether to add reverse force contributions | True |
| `gradwise_weight` | Weight for gradient (force) loss term | 1.0 |
| `graphwise_weight` | Weight for graph-level (energy) loss term | 1.0 |
| `stresswise_weight` | Weight for stress tensor loss term | 0.0 |
| `atomwise_weight` | Weight for atom-level property loss term | 0.0 |
| `grad_multiplier` | Multiplier for gradient values | -1 |
| `stress_multiplier` | Multiplier for stress values | 1.0 |
| `batch_stress` | Whether to calculate stress for batches | True |
| `force_mult_natoms` | Whether to multiply forces by number of atoms | False |
| `energy_mult_natoms` | Whether to multiply energy by number of atoms | True |

### Advanced Features

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `backtracking` | Whether to use backtracking in edge updates | True |
| `lighten_edges` | Whether to reduce edge feature dimensionality | True |
| `lg_on_fly` | Calculate line graphs on the fly (saves memory) | True |
| `link` | Activation function type for linking layers | 'identity' |
| `classification` | Whether the task is classification | False |
| `zero_inflated` | Whether to use zero-inflated loss | False |
| `use_cutoff_function` | Whether to use smooth cutoff functions | False |
| `use_penalty` | Whether to apply penalties | True |
| `penalty_factor` | Factor for penalty term | 0.1 |
| `penalty_threshold` | Threshold for applying penalty | 1.0 |
| `multiply_cutoff` | Whether to multiply features by cutoff function | False |
| `inner_cutoff` | Inner cutoff radius (Å) | 4.0 |
| `exponent` | Exponent for cutoff function | 5 |
| `include_pos_deriv` | Whether to include positional derivatives | False |
| `additional_output_features` | Number of additional output features | 0 |
| `additional_output_weight` | Weight for additional output loss | 0.0 |
| `atomwise_output_features` | Number of atom-level output features | 0 |
| `extra_features` | Number of extra features | 0 |

## Version Control

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `version` | Git commit hash or version identifier | 'af3ae5d1c5711ef9cad6cf930de78f30e6627382' |
