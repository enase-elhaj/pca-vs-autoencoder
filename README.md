
## Linear vs Nonlinear Representation Learning: PCA vs Autoencoder on Fashion-MNIST

This project compares a classical linear method (Principal Component Analysis, PCA) with a nonlinear neural network (Autoencoder) for dimensionality reduction and image reconstruction using the Fashion-MNIST dataset.

The goal is to answer a practical and important question in modern ML:

When is simple, interpretable PCA “good enough”, and when does a neural autoencoder actually provide meaningful benefit?

This project emphasizes:

strong understanding of PCA, eigenvalues, eigenvectors, and SVD

implementation of a real undercomplete autoencoder

quantitative and visual comparison between the two approaches

## Project Overview

Dataset: Fashion-MNIST
60,000 training + 10,000 test 28×28 grayscale clothing images

Task: Compress images → reconstruct → measure information loss

Methods Compared:

PCA (linear, SVD-based)

Autoencoder (nonlinear neural network)

Each image is treated as a 784-dimensional vector (28×28 flattened).
We compress this into a smaller latent space, then reconstruct and evaluate performance.


## Repository Structure
```text
pca-vs-autoencoder/
├─ data/                     # Fashion-MNIST (downloaded automatically)
├─ notebooks/
│  ├─ 01_EDA.ipynb           # Dataset exploration & visualization
│  ├─ 02_PCA.ipynb           # PCA analysis & reconstruction
│  ├─ 03_Autoencoder.ipynb   # Autoencoder training & reconstruction
│  └─ 04_Comparison_and_Analysis.ipynb
│                            # Final PCA vs AE comparison
├─ src/                      # (reserved for reusable utilities / future refactor)
│  └─ README.md
├─ models/                   # Saved autoencoder models
├─ reports/                  # Plots / figures (optional)
├─ requirements.txt
└─ README.md
```
## Dataset & EDA (01_EDA.ipynb)

Loaded dataset via torchvision

Visualized random samples and class distribution

Flattened each image → 784-D feature vector

Computed and visualized the mean image

This frames the idea:
Fashion-MNIST lives in a high-dimensional pixel space, but likely in a much lower-dimensional structure.


## PCA: Linear Dimensionality Reduction (02_PCA.ipynb)

PCA was applied using sklearn.

Principal Component Analysis (PCA) was applied to the Fashion-MNIST training set, treating each 28×28 image as a 784-dimensional vector.

Key findings:

- **Explained variance vs components**
  - ~95% of the variance is captured by the first **182** principal components.
  - ~99% of the variance is captured by the first **445** principal components.
  - This shows that the effective dimensionality of the data is much lower than 784.

- **Reconstruction error vs number of components (MSE on 1,000 images)**  
  - k = 10  → MSE ≈ 0.0239  
  - k = 20  → MSE ≈ 0.0179  
  - k = 50  → MSE ≈ 0.0109  
  - k = 100 → MSE ≈ 0.0063  
  - k = 200 → MSE ≈ 0.0026  

- **Visual inspection**
  - With very few components (e.g., k = 10–20), reconstructions are blurry but still recognizable by class.
  - Around k = 50–100, reconstructions become much sharper and closer to the originals.
  - Increasing beyond ~200 components improves quality but with diminishing returns.

**Interpretation**

- PCA is able to compress 784-dimensional images down to roughly **100–200 dimensions** while preserving most visual information.
- This demonstrates that Fashion-MNIST images lie near a lower-dimensional subspace, and PCA can exploit that structure for compression and reconstruction.


## Autoencoder: Nonlinear Representation (03_Autoencoder.ipynb)

An undercomplete autoencoder was trained to learn a compressed 64-dimensional representation.

Architecture

Encoder: 784 → 256 → 64

Decoder: 64 → 256 → 784

Loss: MSE

Optimizer: Adam

Training: 20 epochs on 20,000 images

Training loss: steadily decreased from ~0.05 → ~0.0096

## PCA vs Autoencoder (k = 64 latent dimensions)

To ensure a fair comparison, both methods used 64-dimensional latent spaces

- PCA with **k = 64** principal components
- An undercomplete autoencoder with a **64-dimensional latent layer**
  - Encoder: 784 → 256 → 64
  - Decoder: 64 → 256 → 784
  - Trained with MSE loss and Adam optimizer for 20 epochs on 20,000 Fashion-MNIST images


Quantitative Comparison (1,000 test images)

PCA (k = 64): MSE ≈ 0.00910

Autoencoder (latent_dim = 64): MSE ≈ 0.00971

At the same compression level, PCA slightly outperforms this autoencoder in reconstruction accuracy.

**Interpretation**


PCA is theoretically the optimal linear method for minimizing reconstruction error (MSE), and Fashion-MNIST contains a significant amount of structure that can be captured linearly.

Although autoencoders are more flexible and capable of learning nonlinear manifolds, they do not automatically outperform PCA. To surpass PCA, the AE typically requires:

higher-capacity architectures (e.g., convolutional autoencoders),

longer or more carefully tuned training,

or task-specific objectives (e.g., denoising, robustness, generative modeling).


**Takeaway**

This comparison highlights an important practical lesson:

PCA remains a very strong, interpretable, and computationally efficient baseline for dimensionality reduction. Neural networks should be introduced when they provide clear additional value, not simply because they are newer or more complex.

## Visual Comparison

Both methods reconstruct images well

PCA often slightly sharper globally

Differences are small but measurable

## Key Insights

PCA remains a very strong baseline

Optimal linear method for minimizing reconstruction MSE

Fast, interpretable, computationally efficient

Autoencoders do not automatically “win”

Nonlinear models need:

more capacity (e.g., convolutional layers)

more training

or task-specific goals

Otherwise PCA can match or beat them

## Use PCA first.
Add neural networks when they clearly add value, not simply because they are newer or more complex.

## How to Run
git clone <repo-url>
cd pca-vs-autoencoder
python -m venv .venv

Activate venv:

Windows

.venv\Scripts\activate

Mac/Linux

source .venv/bin/activate

Install deps:

pip install -r requirements.txt

Run notebooks:

jupyter notebook

👤 Author

Enas Elhaj

Graduate Student — Applied Artificial Intelligence & Data Science
University of Denver | Ritchie School of Engineering & Computer Science

Former telecom & computer engineering lecturer with strong background in:

Python

Machine Learning & AI

Data Science

Databases

Software engineering

Currently building advanced AI and data-driven projects, exploring:

Predictive modeling

AI explainability / responsible AI

End-to-end ML systems

Business decision intelligence

Data-driven product strategy

📌 Passionate about
Turning data into actionable insights that solve real human + business problems.

📌 Contact / Profiles

LinkedIn: https://www.linkedin.com/in/enas-elhaj/

GitHub: https://github.com/enase-elhaj

Email: enas.elhaj@du.edu