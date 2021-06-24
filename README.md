# Sampling with Mirrored Stein Operators

Code for reproducing results in [https://arxiv.org/abs/2106.12506](https://arxiv.org/abs/2106.12506)

## Requirements

- R >= 4.0.4
- python >= 3.8
```
rpy2 >= 3.4.4
tensorflow >= 2.4.0
tensorflow_probability >= 0.12
numpy >= 1.19.5
scipy >= 1.6.3
matplotlib >= 3.4.1
pandas >= 1.2.4
seaborn >= 0.11.1
scikit-learn >= 0.24.2
tqdm
absl-py
```

## Experiments

### Approximation quality on the simplex

* Sparse Dirichlet: `dirichlet.ipynb`
* Quadratic: `quad.ipynb`

### Confidence intervals for post-selection inference

Note: The `R_HOME` variable must be set correctly before running the scripts.

- Simulation
    - 2D example: `selective_inf.ipynb`
    - Nominal coverage vs. actual coverage: `coverage.py`
    - Coverage vs. number of samples: `coverage_wrt_k.py`
    - Plotting: `plot_coverage.ipynb`
- HIV drug resistance: `hiv.ipynb`

### Large-scale posterior inference with non-Euclidean geometry

- Run script: `lr.py`
- Plotting: `plot_lr.ipynb`
