
# `ALIGNNAtomWiseConfig` – Model Configuration

This class defines the configuration schema for the `alignn_atomwise` model in the ALIGNN (Atomistic Line Graph Neural Network) framework. It controls architectural parameters, loss weights, graph construction rules, and auxiliary behaviors used in energy, force, and stress prediction tasks.

## 📌 Basic Info

| Parameter | Type | Default | Description |
|----------|------|---------|-------------|
| `name` | `Literal["alignn_atomwise"]` | — | Identifier for the configuration. Must be `"alignn_atomwise"` to match model type. |

## 🏗️ Model Architecture

| Parameter | Type | Default | Description |
|----------|------|---------|-------------|
| `alignn_layers` | `int` | 2 | Number of ALIGNN (line graph) layers used. |
| `gcn_layers` | `int` | 2 | Number of GCN (base graph) layers used. |
| `atom_input_features` | `int` | 1 | Input feature dimension for each atom (usually atomic number). |
| `edge_input_features` | `int` | 80 | Input features for each bond/edge (e.g., radial basis encoding). |
| `triplet_input_features` | `int` | 40 | Number of features used to encode bond angle triples. |
| `embedding_features` | `int` | 64 | Size of the embedding used throughout the network. |
| `hidden_features` | `int` | 64 | Number of features in hidden layers of the network. |
| `output_features` | `int` | 1 | Output size of the model (e.g., energy, scalar property). |

## 📈 Output Control and Loss Weighting

| Parameter | Type | Default | Description |
|----------|------|---------|-------------|
| `atomwise_output_features` | `int` | 0 | Output dimensions per atom (for per-atom properties). |
| `graphwise_weight` | `float` | 1.0 | Weight applied to the total energy prediction loss. |
| `gradwise_weight` | `float` | 1.0 | Weight applied to the force (gradient) loss. |
| `stresswise_weight` | `float` | 0.0 | Weight applied to the stress tensor loss. |
| `atomwise_weight` | `float` | 0.0 | Weight applied to per-atom target loss. |
| `calculate_gradient` | `bool` | `True` | Whether to calculate gradients of energy w.r.t. atomic positions (for force/stress). |

## ⚛️ Physics-Aware Settings

| Parameter | Type | Default | Description |
|----------|------|---------|-------------|
| `grad_multiplier` | `int` | -1 | Scales force gradients (useful in multitask settings). |
| `force_mult_natoms` | `bool` | `False` | Scale force loss by number of atoms. |
| `energy_mult_natoms` | `bool` | `True` | Scale energy loss by number of atoms (for energy/atom objective). |
| `include_pos_deriv` | `bool` | `False` | Include explicit derivatives with respect to atomic positions. |
| `stress_multiplier` | `float` | 1.0 | Multiplier for the stress loss term. |
| `add_reverse_forces` | `bool` | `True` | Add reverse vectors when computing forces (symmetry enforcement). |

## 🔁 Graph Construction Options

| Parameter | Type | Default | Description |
|----------|------|---------|-------------|
| `use_cutoff_function` | `bool` | `False` | Use a smooth cutoff function on edge distances. |
| `inner_cutoff` | `float` | `3.0` | Distance (in Å) for filtering bond-angle-bond triples (u–v–w). |
| `lighten_edges` | `bool` | `True` | Use the inner cutoff to sparsify line graph edges (for speed). |
| `backtracking` | `bool` | `True` | Whether to allow edges like A–B–A in line graph (usually False for angles). |
| `lg_on_fly` | `bool` | `True` | Construct the line graph dynamically at runtime (on-the-fly). |
| `batch_stress` | `bool` | `True` | Compute stress in batch mode across structures. |
| `multiply_cutoff` | `bool` | `False` | Apply cutoff scaling directly to edge features. |

## 🧪 Loss Penalty Options

| Parameter | Type | Default | Description |
|----------|------|---------|-------------|
| `use_penalty` | `bool` | `True` | Whether to apply penalty loss on large prediction errors. |
| `exponent` | `int` | 5 | Exponent used in the penalty loss function. |
| `penalty_factor` | `float` | 0.1 | Scale factor applied to penalty loss. |
| `penalty_threshold` | `float` | 1.0 | Threshold above which penalty loss is applied. |

## 🧩 Miscellaneous

| Parameter | Type | Default | Description |
|----------|------|---------|-------------|
| `extra_features` | `int` | 0 | Number of extra structure-level input features (global descriptors). |
| `zero_inflated` | `bool` | `False` | Use zero-inflated regression (for sparse outputs). |
| `classification` | `bool` | `False` | Perform classification rather than regression. |
| `additional_output_features` | `int` | 0 | Number of auxiliary output dimensions (for multitask learning). |
| `additional_output_weight` | `float` | 0.0 | Weight applied to auxiliary output loss. |

