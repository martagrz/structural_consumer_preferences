# Neural Demand Estimation with Habit Formation and Rationality Constraints

This repository contains the code for the paper **"Neural Demand Estimation with Habit Formation and Rationality Constraints"**. It implements a neural econometric framework for estimating demand systems that are flexible, scalable, and economically disciplined. The model captures state dependence via habit formation and enforces integrability conditions (Slutsky symmetry, monotonicity) to enable valid welfare analysis.

## Overview

The codebase is organized into two main experimental pipelines:
1.  **Simulation Studies:** Validates the estimator's ability to recover structural parameters (elasticities, welfare) from known data-generating processes (DGPs) including CES, Quasilinear, and Habit Formation.
2.  **Empirical Application:** Applies the framework to the **Dominick's Finer Foods** scanner data (analgesics category) to estimate demand for differentiated products with habit formation.

## Installation

### Prerequisites
- Python 3.8+
- PyTorch
- NumPy, Pandas, Scikit-learn, Matplotlib

### Setup
Clone the repository and install dependencies:
```bash
git clone https://github.com/yourusername/structural_consumer_preferences.git
cd structural_consumer_preferences
pip install -r requirements.txt
```

## Usage

### 1. Simulation Experiments
Run the simulation pipeline to reproduce the DGP recovery results (Table 1) and other simulation figures.

```bash
# Run all simulation experiments
python run_neural_demand_simulation.py

# Run specific experiments (e.g., DGP recovery and CF endogeneity)
python run_neural_demand_simulation.py --exp 01 04

# Fast mode for testing
python run_neural_demand_simulation.py --fast
```

**Experiments:**
*   `01`: DGP Recovery (RMSE, elasticities, welfare across 5 DGPs)
*   `02`: Habit Advantage (Performance gain from habit state)
*   `03`: $\delta$ Identification (Profile likelihood for habit decay)
*   `04`: Control Function Endogeneity Correction

### 2. Dominick's Empirical Application
Run the empirical pipeline to estimate demand on the Dominick's dataset.

**Data Requirement:**
You need the Dominick's `wana.csv` (weekly movement) and `upcana.csv` (UPC catalogue) files. Place them in a `data/` directory or specify their paths.

```bash
# Run all Dominick's experiments
python run_neural_demand_dominicks.py --weekly data/wana.csv --upc data/upcana.csv

# Run specific experiments (e.g., Predictive Accuracy and Welfare)
python run_neural_demand_dominicks.py --weekly data/wana.csv --upc data/upcana.csv --exp 01 03
```

**Experiments:**
*   `01`: Predictive Accuracy (Out-of-sample RMSE/MAE)
*   `02`: Price Elasticities (Own- and cross-price matrices)
*   `03`: Welfare Analysis (Compensating Variation for price shocks)
*   `04`: Demand Curves (Visualizing estimated demand)
*   `07`: Full Model Sweep (Generates all main paper figures/tables)
*   `09`: Regularity Dashboard (Integrability diagnostics)

## Key Models

The core models are implemented in `src/models/`:

*   **`StaticND`**: Static neural demand system. Maps $(p, y) \to w$ via a softmax preference scorer.
*   **`HabitND`**: Habit-augmented demand system. Maps $(p, y, \bar{x}) \to w$, where $\bar{x}$ is the habit stock.
*   **`WindowND`**: Window-based neural demand model that conditions on lagged price/quantity features.
*   **`NeuralIRL_FE` / `MDPNeuralIRL_FE`**: Variants with store fixed effects (learned embeddings).
*   **Control Function (CF)**: All models support an `n_cf` argument to include control function residuals $\hat{v}$ for endogeneity correction.

## Project Structure

```
├── src/
│   └── models/             # PyTorch model definitions (StaticND, HabitND, WindowND, etc.)
├── experiments/
│   ├── neural_demand/
│   │   ├── simulation/     # Simulation experiment scripts (exp01_dgp_recovery.py, etc.)
│   │   └── dominicks/      # Dominick's experiment scripts (exp01_predictive_accuracy.py, etc.)
│   └── dominicks/          # Shared data loading and utility code for Dominick's
├── run_neural_demand_simulation.py  # Entry point for simulations
├── run_neural_demand_dominicks.py   # Entry point for Dominick's application
└── results/                # Output directory for tables and figures
```

## Citation

If you use this code, please cite the associated paper:

> Grzeskiewicz, M. (2026). "Neural Demand Estimation with Habit Formation and Rationality Constraints."
