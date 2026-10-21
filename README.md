# Application of Graphic Neural Networks to Predict Molecular Properties Based on Geometric Features

This project compares three message-passing neural network variants for predicting a molecular property from QM9: a feature-only MPNN, a coordinate-input MPNN, and a distance-based invariant (3D) MPNN. The target used in the notebook is the QM9 electric dipole moment.

## 1. Dataset

### QM9 and target

QM9 contains small organic molecules with up to nine heavy atoms. Each molecule is represented as a graph whose nodes are atoms and whose edges encode chemical bonds. The notebook loads 130,831 molecules and selects target index 0, the electric dipole moment (in Debye).

For the experiments, the selected dataset subset is divided by index into:

| Split | Dataset indices | Samples |
|---|---:|---:|
| Training | 0–999 | 1,000 |
| Validation | 1,000–1,999 | 1,000 |
| Test | 2,000–2,999 | 1,000 |

The batch size is 32. The target values are standardized using the mean and standard deviation computed from the loaded target array. Training minimizes mean squared error on the standardized target; reported MAE is rescaled to the original target units (Debye).

### Graph representation and initial features

For a molecule with atom set $V$, the node feature matrix is $X\in\mathbb{R}^{|V|\times 11}$. Each atom has 11 features:

1. Five-way one-hot atom type: H, C, N, O, or F.
2. Atomic number.
3. Aromaticity flag.
4. Three-way one-hot hybridization: sp, sp², or sp³.
5. Number of attached hydrogens.

The notebook also retains each atom's atomic number as an auxiliary feature and its 3D position $r_i\in\mathbb{R}^3$.

The edge feature matrix is $E\in\mathbb{R}^{|E|\times 4}$, with a four-way one-hot encoding of bond type: single, double, triple, or aromatic. The graph transform adds every directed pair of distinct atoms, so each molecular graph is complete and has no self-loops. If a pair is not chemically bonded, its edge feature is the all-zero vector; existing chemical bonds retain their bond-type encoding. Thus the graph structure allows messages between every pair of atoms, while edge features distinguish bonded from non-bonded pairs.

## 2. Model Framework

### Shared message-passing and readout

All three models use 4 message-passing layers with 64 hidden dimensions, residual connections, and a graph-level mean readout. Let $h_i^{(l)}$ be atom $i$'s hidden state at layer $l$, and let $e_{ij}$ be the edge features from atom $j$ to atom $i$.

In the feature-only MPNN, the message function is an MLP over the source and destination states and edge features:

$$
m_{ij}^{(l)}=\psi\left([h_i^{(l)}\,\|\,h_j^{(l)}\,\|\,e_{ij}]\right).
$$

Messages are **summed** over incoming neighbors:

$$
\bar m_i^{(l)}=\sum_{j\ne i}m_{ij}^{(l)}.
$$

The update function is another MLP over the current node state and aggregated message. The layer output also has an outer residual connection:

$$
\tilde h_i^{(l+1)}=\phi\left([h_i^{(l)}\,\|\,\bar m_i^{(l)}]\right),
\qquad
h_i^{(l+1)}=h_i^{(l)}+\tilde h_i^{(l+1)}.
$$

Here $[\cdot\|\cdot]$ denotes concatenation. Both $\psi$ and $\phi$ are two-layer MLPs with Linear, BatchNorm, and ReLU blocks. Initial atom features are projected from 11 dimensions to the 64-dimensional hidden space.

After the final layer, node states are averaged within each molecule and passed to a linear prediction head:

$$
h_G=\frac{1}{|V|}\sum_{i\in V}h_i^{(L)},
\qquad
\hat y_G=W_{\mathrm{out}}h_G+b_{\mathrm{out}}.
$$

### Feature-only MPNN

The baseline uses atom and bond features only. It does not use 3D positions or any geometric quantity. Its message and update equations are the shared equations above.

### Coordinate MPNN

The coordinate-input model concatenates each atom's three Cartesian coordinates to its 11 atom features before the initial projection:

$$
h_i^{(0)}=W_{\mathrm{in}}[x_i\,\|\,r_i]+b_{\mathrm{in}}.
$$

It then uses the same MPNN message function, sum aggregation, residual update, and mean graph readout as the baseline. Coordinates are input features; this model does not update coordinates or make the output invariant to rotations and translations.

### Distance-based invariant MPNN (3D MPNN)

The notebook names this model **InvariantMPNNModel**. It keeps the same atom-feature input and shared update/readout design, but augments each edge message with the Euclidean distance between its endpoint atoms:

$$
d_{ij}=\|r_i-r_j\|_2,
$$

$$
m_{ij}^{(l)}=\psi\left([h_i^{(l)}\,\|\,h_j^{(l)}\,\|\,e_{ij}\,\|\,d_{ij}]\right).
$$

Distances are unchanged by a global translation or rotation, so the scalar graph prediction is invariant to these transformations. Unlike the coordinate-input model, this variant does not feed absolute coordinate components to the MLP.

## 3. Experimental Results

The notebook trains each model for 100 epochs using Adam (learning rate 0.001), MSE loss on standardized targets, and a ReduceLROnPlateau learning-rate scheduler. Validation MAE is used to track the best checkpoint/epoch. The evaluation metric is **mean absolute error (MAE)**, converted back to Debye.

| Model | Best validation MAE (Debye) | Test MAE (Debye) |
|---|---:|---:|
| Feature-only MPNN (MPNNModel) | 0.7573 | 0.6159 |
| Coordinate MPNN (CoordMPNNModel) | 0.7896 | 0.6284 |
| Distance-based invariant MPNN (InvariantMPNNModel) | 0.5348 | 0.5048 |

On this saved run, the distance-based invariant model has the lowest test MAE. The coordinate-input model is slightly worse than the feature-only baseline, while adding pairwise distances improves the test result.

**Reproducibility note:** These values are the outputs saved in the supplied notebook. They are from one indexed 3,000-molecule subset and one training run, so they can vary with initialization and environment. The notebook evaluates the test set whenever validation performance improves and reports the test MAE associated with the best validation epoch.
