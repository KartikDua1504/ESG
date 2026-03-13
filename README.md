# ESG Portfolio Optimization with Multi-Objective Evolutionary Algorithms

This project implements a sophisticated tri-objective portfolio optimization framework that balances **Financial Returns**, **Portfolio Risk (Variance)**, and **ESG (Environmental, Social, and Governance) Risk**. 

It leverages multi-objective evolutionary algorithms (NSGA-II) to discover the Pareto Front of optimal portfolios and incorporates preference handling via **SMAA-TOPSIS** to identify the best compromise solutions based on different investor profiles.

## Key Features

- **Tri-Objective Optimization**: Simultaneously optimizes for:
  1. **Maximize Expected Return**: Enhancing portfolio profitability.
  2. **Minimize Risk**: Reducing portfolio variance using a covariance matrix.
  3. **Minimize ESG Risk**: Integrating sustainability and social responsibility into the investment process.
- **Advanced Evolutionary Algorithm**: Uses **NSGA-II** (Non-dominated Sorting Genetic Algorithm II) to find the strict market boundaries (Pareto Front).
- **Preference Handling (SMAA-TOPSIS)**: Implements Stochastic Multicriteria Acceptability Analysis (SMAA) combined with TOPSIS to handle subjective investor preferences and uncertainty in weights.
- **Investor Profiles**: Includes pre-defined profiles such as:
  - `FIN-agg`: Aggressive Financial (Focus on Returns).
  - `FIN-con`: Conservative Financial (Focus on Risk reduction).
  - `ESG-awa`: ESG Aware (Balanced approach).
  - `ESG-mot`: ESG Motivated (Focus on sustainability).
- **Terminal-Based Visualization**: Rich interactive plots using `plotext` directly in your terminal, featuring Pareto projections, Monte Carlo weight distributions, and Rank Acceptability Indices (RAI).

## Installation

Ensuring you have Python installed, install the required dependencies:

```bash
pip install -r requirements.txt
```

The core dependencies are:
- `numpy`: Numerical computations.
- `pymoo`: Multi-objective optimization framework.
- `scipy`: Scientific utilities and distance metrics.
- `plotext`: Terminal-based plotting.

## Usage

The project contains two main execution scripts:

### 1. Basic Optimization (`esg_portfolio.py`)
Runs a standard tri-objective optimization on a set of 50 assets and identifies a single "Best Compromise" portfolio using SMAA-TOPSIS.

```bash
python esg_portfolio.py
```

### 2. Structured Simulation (`esg_portfolio-structured.py`)
A more complex simulation involving 100 assets with structured archetypes. It runs a full multi-profile analysis (4 different investor types) and displays results in a "slideshow" format within the terminal.

```bash
python esg_portfolio-structured.py
```

## Methodology

### Mathematical Formulation
The optimization problem is defined as:
- **Minimize** $F(x) = [ -f_1(x), f_2(x), f_3(x) ]$
- **Where**:
  - $f_1(x) = \mu^T \cdot x$ (Expected Return)
  - $f_2(x) = x^T \cdot \Sigma \cdot x$ (Portfolio Variance)
  - $f_3(x) = E^T \cdot x$ (ESG Risk)
- **Constraints**:
  - $\sum x_i = 1$ (Full investment)
  - $x_i \geq 0$ (No short selling)

### Decision Support
After the Pareto Front is generated, **SMAA-TOPSIS** is used to rank portfolios. By simulating thousands of possible weight combinations within the bounds of chosen investor profiles, the system calculates a **Rank Acceptability Index (RAI)**, indicating the probability of a portfolio being the top choice for that specific profile.

---
*Created for advanced portfolio analysis and ESG integration.*
