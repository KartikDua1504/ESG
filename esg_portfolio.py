import numpy as np
import pandas as pd
import plotext as plt
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.repair import Repair
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from scipy.spatial.distance import cdist

# ==========================================
# 1. DATA PREPARATION (Mock Data for N Assets)
# ==========================================
np.random.seed(42)
NUM_ASSETS = 50

# Generate synthetic data: Expected Returns, Covariance Matrix, and ESG Risk Scores
expected_returns = np.random.uniform(0.02, 0.15, NUM_ASSETS)
# Create a positive semi-definite covariance matrix
A = np.random.rand(NUM_ASSETS, NUM_ASSETS)
cov_matrix = np.dot(A, A.transpose()) * 0.05 
# ESG Risk Scores (Lower is better, per author's email)
esg_risks = np.random.uniform(10, 50, NUM_ASSETS) 

# ==========================================
# 2. OPTIMIZATION PROBLEM DEFINITION
# ==========================================
class PortfolioRepair(Repair):
    """ Ensures that portfolio weights always sum to 1 and are positive. """
    def _do(self, problem, X, **kwargs):
        X[X < 0] = 0
        return X / X.sum(axis=1, keepdims=True)

class ESGPortfolioProblem(ElementwiseProblem):
    def __init__(self, returns, cov, esg):
        self.returns = returns
        self.cov = cov
        self.esg = esg
        n_assets = len(returns)
        
        # 3 Objectives: Return, Risk, ESG Risk. 
        # Variables: N asset weights. Bounds: [0, 1]
        super().__init__(n_var=n_assets, n_obj=3, n_ieq_constr=0, 
                         xl=np.zeros(n_assets), xu=np.ones(n_assets))

    def _evaluate(self, x, out, *args, **kwargs):
        w = x
        
        # Obj 1: Maximize Return -> Minimize Negative Return
        f1 = -np.dot(w, self.returns)
        
        # Obj 2: Minimize Risk (Portfolio Variance)
        f2 = np.dot(w.T, np.dot(self.cov, w))
        
        # Obj 3: Minimize ESG Risk
        f3 = np.dot(w, self.esg)
        
        # The author specifically noted NO normalization happens here.
        # We pass raw objective values directly to the optimizer.
        out["F"] = [f1, f2, f3]

# ==========================================
# 3. RUNNING THE EVOLUTIONARY ALGORITHM
# ==========================================
print("Starting Multi-Objective Optimization...")
problem = ESGPortfolioProblem(expected_returns, cov_matrix, esg_risks)

# Using NSGA-II as the robust MOGA alternative to the proprietary ev-MOGA
algorithm = NSGA2(
    pop_size=200,
    repair=PortfolioRepair(), # Forces weights to sum to 1 dynamically
    eliminate_duplicates=True
)

termination = get_termination("n_gen", 400)

res = minimize(problem,
               algorithm,
               termination,
               seed=1,
               save_history=False,
               verbose=False)

# Extract Pareto Front (F) and corresponding Weights (X)
# Re-invert return values for analysis
pareto_front = res.F
pareto_front[:, 0] = -pareto_front[:, 0] 
pareto_weights = res.X

print(f"Discovered {len(pareto_front)} optimal portfolios on the Pareto front.")

# ==========================================
# 4. SMAA-TOPSIS IMPLEMENTATION
# ==========================================
print("Running SMAA-TOPSIS for preference handling...")

def run_smaa_topsis(pareto_front, n_iterations=10000):
    n_portfolios, n_obj = pareto_front.shape
    
    # Step A: Normalization for TOPSIS (Happens AFTER ev-MOGA, per email)
    # Benefit criteria: Return (idx 0). Cost criteria: Risk (idx 1), ESG Risk (idx 2)
    norm_front = np.zeros_like(pareto_front)
    norm_front[:, 0] = pareto_front[:, 0] / np.sqrt((pareto_front[:, 0]**2).sum())
    norm_front[:, 1] = (1 / pareto_front[:, 1]) / np.sqrt(((1 / pareto_front[:, 1])**2).sum())
    norm_front[:, 2] = (1 / pareto_front[:, 2]) / np.sqrt(((1 / pareto_front[:, 2])**2).sum())

    # Generate random weight vectors for SMAA (Monte Carlo simulation of preferences)
    # Dirichlet distribution ensures weights sum to 1
    w_space = np.random.dirichlet(np.ones(n_obj), size=n_iterations)
    
    rank_counters = np.zeros((n_portfolios, n_portfolios))

    # Calculate TOPSIS for each weight vector scenario
    for w in w_space:
        weighted_front = norm_front * w
        
        # Ideal and Anti-Ideal solutions
        ideal_best = np.max(weighted_front, axis=0)
        ideal_worst = np.min(weighted_front, axis=0)
        
        dist_best = cdist(weighted_front, [ideal_best]).flatten()
        dist_worst = cdist(weighted_front, [ideal_worst]).flatten()
        
        # Closeness coefficient
        closeness = dist_worst / (dist_best + dist_worst)
        
        # Rank portfolios based on closeness (higher is better)
        ranks = len(closeness) - np.argsort(np.argsort(closeness))
        
        # Tally the ranks for the acceptability index
        for port_idx, rank in enumerate(ranks):
            rank_counters[port_idx, rank - 1] += 1

    # Calculate Rank Acceptability Index (RAI)
    rai = rank_counters / n_iterations
    return rai

rai_matrix = run_smaa_topsis(pareto_front)

# Find the most universally acceptable portfolio (highest probability of ranking 1st)
best_portfolio_idx = np.argmax(rai_matrix[:, 0])
best_portfolio_stats = pareto_front[best_portfolio_idx]

print("\n--- BEST COMPROMISE PORTFOLIO (SMAA-TOPSIS) ---")
print(f"Expected Return: {best_portfolio_stats[0]*100:.2f}%")
print(f"Risk (Variance): {best_portfolio_stats[1]:.4f}")
print(f"ESG Risk Score:  {best_portfolio_stats[2]:.2f}")

# ==========================================
# 5. TERMINAL PLOTTING (Plotext)
# ==========================================
# Plotting Return vs ESG Risk (2D projection of 3D Pareto Front)
returns_plot = pareto_front[:, 0] * 100
esg_risk_plot = pareto_front[:, 2]

plt.scatter(esg_risk_plot, returns_plot, color='blue', marker='dot')
plt.scatter([best_portfolio_stats[2]], [best_portfolio_stats[0]*100], color='red', marker='x')

plt.title("Pareto Front: ESG Risk vs Return")
plt.xlabel("ESG Risk (Lower is Better)")
plt.ylabel("Return (%)")
plt.plotsize(80, 25)
plt.theme('dark') # Fits nicely with neovim terminal setups
plt.show()
