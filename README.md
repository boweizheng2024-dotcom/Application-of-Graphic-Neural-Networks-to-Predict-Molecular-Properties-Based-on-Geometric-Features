# Application of Graph Neural Networks to Predict Molecular Properties Based on Geometric Features

This project uses three message passing neural network variants to predict the electric dipole moment of drug-like molecules from QM9 (Quantum Mechanics dataset 9).

## 1. Dataset

### QM9 and target

QM9 contains small organic molecules with up to nine heavy atoms. Each molecule is represented as a graph whose nodes are atoms and whose edges encode chemical bonds. To keep the practical experiment quick and make the three models easier to compare, it uses a small subset split equally into 1,000 training, 1,000 validation, and 1,000 test molecules (consecutive dataset slices). For stronger predictive performance, use substantially more of QM9 for training—for example, an 80/10/10 train/validation/test split. A scaffold-based split is especially useful when the goal is to assess generalization to molecular structures whose scaffolds were not seen during training.

### Graph representation and initial features

Initially, each atom has 11 features:

1. Five-way one-hot atom type: H, C, N, O, or F.
2. Atomic number.
3. Aromaticity flag.
4. Three-way one-hot hybridization: sp, sp², or sp³.
5. Number of attached hydrogens.

The dataset also retains each atom's 3D position, which we will use in the last two MPNN variants.

Each edge has a four-way one-hot encoding of bond type: single, double, triple, or aromatic. 

The graph transform adds every directed pair of distinct atoms, so each molecular graph is complete and has no self-loops. If a pair is not chemically bonded, its edge feature is the all-zero vector; existing chemical bonds retain their bond-type encoding.

## 2. Model Framework

### Basic MPNN

Let $h_i^{(l)}$ be the hidden state of atom $i$ at layer $l$, and $e_{ij}$ the edge features from atom $j$ to atom $i$. The basic MPNN first computes messages, sums them at each destination atom, and updates the node state with a residual connection:

$$
m_{ij}^{(l)}=\psi\left([h_i^{(l)}\,\|\,h_j^{(l)}\,\|\,e_{ij}]\right),
\qquad
\bar m_i^{(l)}=\sum_{j\ne i}m_{ij}^{(l)},
$$

$$
h_i^{(l+1)}=h_i^{(l)}+
\phi\left([h_i^{(l)}\,\|\,\bar m_i^{(l)}]\right).
$$

In this equation, $m_{ij}^{(l)}$ is the message sent from atom $j$ to atom $i$, and $\bar m_i^{(l)}$ is the sum of messages received by atom $i$. The update adds the MLP output to the previous node state through a residual connection.

Here $[\cdot\|\cdot]$ denotes concatenation. Both $\psi$ and $\phi$ are two-layer MLPs with Linear, BatchNorm, and ReLU blocks. Initial atom features are projected from 11 dimensions to the 64-dimensional hidden space.

After the final layer, node states are averaged within each molecule and passed to a linear prediction head:

$$
h_G=\frac{1}{|V|}\sum_{i\in V}h_i^{(L)},
\qquad
\hat y_G=W_{\mathrm{out}}h_G+b_{\mathrm{out}}.
$$

### Coordinate MPNN

Unlike the basic MPNN, this variant concatenates each atom's 3D coordinate $r_i\in\mathbb{R}^3$ to its 11-dimensional atom feature before the initial projection:

$$
h_i^{(0)}=W_{\mathrm{in}}[x_i\,\|\,r_i]+b_{\mathrm{in}}.
$$

It then uses the basic MPNN's message, aggregation, update, residual, and graph readout operations. Because it uses absolute Cartesian coordinates, its prediction is not invariant to rotations or translations of the same molecule.

### 3D MPNN

Unlike the basic MPNN, the 3D MPNN keeps the initial node input as the 11-dimensional atom feature $x_i$ (it does not concatenate absolute coordinates). Instead, it appends the pairwise Euclidean distance to each message:

$$
d_{ij}=\|r_i-r_j\|_2,
\qquad
m_{ij}^{(l)}=\psi\left([h_i^{(l)}\,\|\,h_j^{(l)}\,\|\,e_{ij}\,\|\,d_{ij}]\right).
$$

The rest of message aggregation, node updating, residual connections, and graph readout follows the basic MPNN.

**Intuition:** Rotating or translating the coordinate system changes the coordinates $r_i$, but it does not change the distance between any pair of atoms. Using distances therefore lets the model use 3D molecular geometry while keeping its scalar prediction invariant to global rotations and translations. This is useful because the predicted molecular property should not depend on how the same molecule is oriented or positioned in space.

## 3. Experimental Results

### Hyperparameter configuration

| Category | Configuration |
|---|---|
| Data and training | 1,000/1,000/1,000 train/validation/test molecules; batch size 32 (training shuffled); 100 epochs |
| Model architecture | 4 message-passing layers; hidden dimension 64; input dimension 11 (Basic/3D) or 14 (Coordinate); edge dimension 4; output dimension 1 |
| Message MLP | Input dimension 132 (Basic/Coordinate) or 133 (3D); Linear → BatchNorm → ReLU → Linear(64→64) → BatchNorm → ReLU |
| Update MLP | Concatenated node state and aggregated message; Linear(128→64) → BatchNorm → ReLU → Linear(64→64) → BatchNorm → ReLU |
| Aggregation, residual, readout | Sum aggregation; residual connection after each message-passing layer; global mean pooling followed by a linear prediction head |
| Optimizer and scheduler | Adam, learning rate 0.001; ReduceLROnPlateau (mode=min, factor=0.9, patience=5, minimum learning rate=0.00001) |
| Objective and metric | MSE loss on standardized targets; MAE reported in Debye |

Validation MAE is used by the learning-rate scheduler and to select the model-selection epoch. The test MAE shown below is the test error measured at the epoch with the best validation MAE; it is **not necessarily the result from the final (100th) epoch**. The notebook evaluates the test set whenever validation MAE improves and reports the test error from the selected epoch. Validation selects the epoch; it does not provide gradients for updating model weights.

| Model | Best validation MAE (Debye) | Test MAE (Debye) |
|---|---:|---:|
| Feature-only MPNN (MPNNModel) | 0.7573 | 0.6159 |
| Coordinate MPNN (CoordMPNNModel) | 0.7896 | 0.6284 |
| 3D MPNN (InvariantMPNNModel) | 0.5348 | 0.5048 |

On this saved run, the 3D MPNN has the lowest test MAE. The coordinate-input model is slightly worse than the feature-only baseline, while adding pairwise distances improves the test result.

**Reproducibility note:** These values are the outputs saved in the supplied notebook. They are from one indexed 3,000-molecule subset and one training run, so they can vary with initialization and environment. The notebook evaluates the test set whenever validation performance improves and reports the test MAE associated with the best validation epoch.
