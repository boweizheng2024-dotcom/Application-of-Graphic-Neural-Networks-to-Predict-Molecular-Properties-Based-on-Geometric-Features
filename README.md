# Application of Graph Neural Networks to Predict Molecular Properties Based on Geometric Features

This project uses three message passing neural network variants to predict the electric dipole moment of drug-like molecules from QM9 (Quantum Mechanics dataset 9).

## 1. Dataset

### QM9 and target

QM9 contains small organic molecules with up to nine heavy atoms. Each molecule is represented as a graph whose nodes are atoms and whose edges encode chemical bonds. 

To keep the practical experiment quick and make the three models easier to compare, it uses a small subset split equally into 1,000 training, 1,000 validation, and 1,000 test molecules. For stronger predictive performance, use substantially more of QM9 for training—for example, an 80%/10%/10% train/validation/test split.

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

Let $h_i^{(l)}$ be the feature of node $i$ at layer $l$, and $e_{ij}$ the edge features from node $j$ to node $i$. The node feature at layer $0$ is obtained by linearly projecting the initial node features:

$$
h_i^{0} = W_{in} h_i + b_{in},
$$

The basic MPNN first computes messages, sums them at each destination atom, and updates the node state with a residual connection:

$$
m_{ij}^{(l)}=\psi\left([h_i^{(l)}\|h_j^{(l)}\|e_{ij}]\right),
\qquad
h_i^{(l+1)}=h_i^{(l)}+
\phi\left([h_i^{(l)}\|\sum_{j\ne i}m_{ij}^{(l)}]\right).
$$


Here $[\cdot\|\cdot]$ denotes concatenation. Both $\psi$ and $\phi$ are two-layer MLPs with Linear layers, each followed by BatchNorm and ReLU. After the final layer $L$, node states are averaged within each molecule and passed to a linear prediction head:

$$
h_G=\frac{1}{|V|}\sum_{i\in V}h_i^{(L)},
\qquad
\hat y_G=W_{\mathrm{out}}h_G+b_{\mathrm{out}}.
$$

### Coordinate MPNN

Unlike the basic MPNN, the Coordinate MPNN concatenates each atom's 3D coordinate $r_i\in\mathbb{R}^3$ to its 11-dimensional node feature before the initial projection:

$$
h_i^{(0)}=W_{\mathrm{in}}[h_i|r_i]+b_{\mathrm{in}}.
$$

It then uses the basic MPNN's message, aggregation, update, residual, and graph readout operations.

### 3D MPNN

The 3D MPNN first linearly projects each node’s 11-dimensional initial feature into $h_i^{(0)}$, as in the basic MPNN, without concatenating its absolute coordinates. It then appends the pairwise Euclidean distance to each message:

$$
d_{ij}=\lVert r_i-r_j\rVert_2,
\qquad
m_{ij}^{(l)}=\psi\left([h_i^{(l)}|h_j^{(l)}|e_{ij}|d_{ij}]\right).
$$

The rest of message aggregation, node updating, residual connections, and graph readout follows the basic MPNN.

**Intuition:** Rotating or translating the coordinate system changes the coordinates $r_i$, but it does not change the distance between any pair of atoms. Using distances therefore lets the model use 3D molecular geometry while keeping its scalar prediction invariant to global rotations and translations. This is useful because the predicted molecular property should not depend on how the same molecule is oriented or positioned in space.

## 3. Experiment

### Hyperparameter configuration

| Category | Configuration |
|---|---|
| Data and training | 1,000/1,000/1,000 train/validation/test molecules; batch size 32; 100 epochs |
| Model architecture | 4 message passing layers; hidden dimension 64; input dimension 11 (Basic/3D) or 14 (Coordinate); edge dimension 4; output dimension 1 |
| Message MLP | Input dimension 132 (Basic/Coordinate) or 133 (3D); Linear → BatchNorm → ReLU → Linear(64→64) → BatchNorm → ReLU |
| Update MLP | Concatenated node state and aggregated message; Linear(128→64) → BatchNorm → ReLU → Linear(64→64) → BatchNorm → ReLU |
| Optimizer and scheduler | Adam, learning rate 0.001; ReduceLROnPlateau (mode=min, factor=0.9, patience=5, minimum learning rate=0.00001) |
| Loss function | Mean squared error (MSE) |
| Evaluation metric | Mean absolute error (MAE) |

### Experimental results

| Model | Test MAE |
|---|---:|
| Basic MPNN | 0.6159 |
| Coordinate MPNN  | 0.6284 |
| 3D MPNN  | 0.5048 |

The reported test MAE is measured using the model from the epoch with the lowest validation MAE.
