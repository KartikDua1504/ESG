import numpy as np
import plotext as plt
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.repair import Repair
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from scipy.spatial.distance import cdist
import sys

# ==========================================
# 0. MATHEMATICAL FORMULATION OUTPUT
# ==========================================
def print_objective_function():
    print("\n" + "="*65)
    print("   MATHEMATICAL FORMULATION OF THE OBJECTIVE FUNCTION")
    print("="*65)
    print(" Let 'x' be the vector of asset weights [x_1, x_2, ..., x_N]")
    print(" Let 'μ' be the vector of expected returns")
    print(" Let 'Σ' be the covariance matrix of asset returns")
    print(" Let 'E' be the vector of ESG risk scores (Lower is better)")
    print("\n The Tri-Objective Optimization Problem is defined as:")
    print("   Minimize F(x) = [ -f1(x), f2(x), f3(x) ]")
    print("\n Where:")
    print("   f1(x) = μ^T * x        (Maximize Expected Return)")
    print("   f2(x) = x^T * Σ * x    (Minimize Portfolio Variance)")
    print("   f3(x) = E^T * x        (Minimize Portfolio ESG Risk)")
    print("\n Subject to constraints:")
    print("   Σ x_i = 1              (Full investment, no leverage)")
    print("   x_i >= 0               (No short selling, bounded by 1)")
    print("="*65 + "\n")

# ==========================================
# 1. STRUCTURED MARKET DATA GENERATOR
# ==========================================
np.random.seed(10)
NUM_ASSETS = 100

archetypes = np.array([
    [0.02, 0.01, 15.0],  # Low Return, Low Vol, Low ESG Risk
    [0.18, 0.25, 45.0],  # High Return, High Vol, High ESG Risk
    [0.12, 0.15, 10.0],  # High Return, Med Vol, Low ESG Risk 
    [0.08, 0.08, 30.0]   # Med Return, Med Vol, Med ESG Risk
])

weights = np.random.dirichlet(np.ones(len(archetypes)), NUM_ASSETS)
asset_profiles = np.dot(weights, archetypes)

expected_returns = asset_profiles[:, 0] + np.random.normal(0, 0.005, NUM_ASSETS)
volatilities = asset_profiles[:, 1] + np.random.normal(0, 0.01, NUM_ASSETS)
volatilities = np.clip(volatilities, 0.01, 0.5)
esg_risks = asset_profiles[:, 2] + np.random.normal(0, 1.0, NUM_ASSETS)

corr_matrix = np.random.uniform(0.2, 0.6, (NUM_ASSETS, NUM_ASSETS))
np.fill_diagonal(corr_matrix, 1.0)
corr_matrix = (corr_matrix + corr_matrix.T) / 2
cov_matrix = np.outer(volatilities, volatilities) * corr_matrix

# ==========================================
# 2. OPTIMIZATION PROBLEM 
# ==========================================
class PortfolioRepair(Repair):
    def _do(self, problem, X, **kwargs):
        X[X < 0] = 0
        sum_X = X.sum(axis=1, keepdims=True)
        return np.where(sum_X > 0, X / sum_X, 1.0/X.shape[1])

class ESGPortfolioProblem(ElementwiseProblem):
    def __init__(self, returns, cov, esg):
        self.returns = returns
        self.cov = cov
        self.esg = esg
        super().__init__(n_var=len(returns), n_obj=3, xl=np.zeros(len(returns)), xu=np.ones(len(returns)))

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = -np.dot(x, self.returns)          
        f2 = np.dot(x.T, np.dot(self.cov, x))  
        f3 = np.dot(x, self.esg)               
        out["F"] = [f1, f2, f3]

# ==========================================
# 3. EVOLUTIONARY ALGORITHM
# ==========================================
print_objective_function()
print("[1/3] Evolving Pareto Front (Finding the strict market boundaries)...")
problem = ESGPortfolioProblem(expected_returns, cov_matrix, esg_risks)
algorithm = NSGA2(pop_size=600, repair=PortfolioRepair(), eliminate_duplicates=True)
res = minimize(problem, algorithm, get_termination("n_gen", 800), seed=1, verbose=False)

pareto_front = res.F
pareto_front[:, 0] = -pareto_front[:, 0] # Re-invert returns

# ==========================================
# 4. ROBUST SMAA-TOPSIS WITH PROFILES & RAI
# ==========================================
def generate_profile_weights(n, bounds):
    weights = []
    while len(weights) < n:
        w = np.random.dirichlet([1, 1, 1])
        if (bounds[0][0] <= w[0] <= bounds[0][1] and
            bounds[1][0] <= w[1] <= bounds[1][1] and
            bounds[2][0] <= w[2] <= bounds[2][1]):
            weights.append(w)
    return np.array(weights)

def run_smaa_topsis_with_rai(pareto_front, weight_space):
    norm_front = pareto_front / np.sqrt((pareto_front**2).sum(axis=0))
    n_portfolios = pareto_front.shape[0]
    rank_counters = np.zeros((n_portfolios, n_portfolios))
    
    for w in weight_space:
        weighted_front = norm_front * w
        ideal_best = np.array([np.max(weighted_front[:,0]), np.min(weighted_front[:,1]), np.min(weighted_front[:,2])])
        ideal_worst = np.array([np.min(weighted_front[:,0]), np.max(weighted_front[:,1]), np.max(weighted_front[:,2])])
        
        dist_best = cdist(weighted_front, [ideal_best]).flatten()
        dist_worst = cdist(weighted_front, [ideal_worst]).flatten()
        
        closeness = dist_worst / (dist_best + dist_worst)
        
        ranks = n_portfolios - 1 - np.argsort(np.argsort(closeness))
        for p_idx, r in enumerate(ranks):
            rank_counters[p_idx, r] += 1

    rai_matrix = rank_counters / len(weight_space)
    best_idx = np.argmax(rai_matrix[:, 0]) 
    return best_idx, rai_matrix[best_idx]

profiles = {
    "FIN-agg": [(0.6, 0.9), (0.05, 0.25), (0.05, 0.25)],
    "FIN-con": [(0.1, 0.3), (0.6, 0.9), (0.05, 0.2)],   
    "ESG-awa": [(0.3, 0.5), (0.2, 0.4), (0.3, 0.5)],    
    "ESG-mot": [(0.1, 0.3), (0.1, 0.3), (0.6, 0.9)]     
}

print("[2/3] Running SMAA-TOPSIS Monte Carlo Simulations...")
best_portfolios_idx = {}
best_portfolios_rai = {}
profile_weights = {}

for name, bounds in profiles.items():
    w_space = generate_profile_weights(2000, bounds)
    profile_weights[name] = w_space
    
    best_idx, best_rai = run_smaa_topsis_with_rai(pareto_front, w_space)
    best_portfolios_idx[name] = best_idx
    best_portfolios_rai[name] = best_rai
    print(f"      -> {name} complete.")

# ==========================================
# 5. TERMINAL PLOTTING (SLIDESHOW MODE)
# ==========================================
plt.theme('dark')

def pause():
    input("\n[Press ENTER to view the next set of graphs...]\n")

# --- SLIDE 1: PARETO FRONT PROJECTIONS ---
plt.clf()
plt.subplots(1, 3)
v_ret, v_var, v_esg = pareto_front[:, 0] * 100, pareto_front[:, 1], pareto_front[:, 2]
markers = {"FIN-agg": "x", "FIN-con": "+", "ESG-awa": "*", "ESG-mot": "d"}
colors = {"FIN-agg": "red", "FIN-con": "yellow", "ESG-awa": "green", "ESG-mot": "magenta"}

plt.subplot(1, 1)
plt.scatter(v_var, v_ret, color='blue', marker='braille')
for name, idx in best_portfolios_idx.items(): plt.scatter([v_var[idx]], [v_ret[idx]], color=colors[name], marker=markers[name], label=name)
plt.title("Variance vs Return")

plt.subplot(1, 2)
plt.scatter(v_esg, v_ret, color='cyan', marker='braille')
for name, idx in best_portfolios_idx.items(): plt.scatter([v_esg[idx]], [v_ret[idx]], color=colors[name], marker=markers[name])
plt.title("ESG Risk vs Return")

plt.subplot(1, 3)
plt.scatter(v_var, v_esg, color='magenta', marker='braille')
for name, idx in best_portfolios_idx.items(): plt.scatter([v_var[idx]], [v_esg[idx]], color=colors[name], marker=markers[name])
plt.title("Variance vs ESG Risk")

plt.plotsize(100, 30)
print("\n--- GRAPH SET 1: PARETO PROJECTIONS (Fig 6/7/8) ---")
plt.show()
pause()

# --- SLIDE 2: MONTE CARLO WEIGHT ITERATIONS ---
plt.clf()
plt.subplots(2, 2)
positions = {"FIN-agg": (1, 1), "FIN-con": (1, 2), "ESG-awa": (2, 1), "ESG-mot": (2, 2)}

for name, pos in positions.items():
    plt.subplot(*pos)
    w_500 = profile_weights[name][:500] 
    plt.plot(w_500[:, 0], label="Return Wgt", color="green")
    plt.plot(w_500[:, 1], label="Var Wgt", color="red")
    plt.plot(w_500[:, 2], label="ESG Wgt", color="blue")
    plt.title(f"{name} Weights (500 iter)")
    plt.ylim(0, 1)

plt.plotsize(100, 35)
print("\n--- GRAPH SET 2: MONTE CARLO WEIGHTS (Fig 3) ---")
plt.show()
pause()

# --- SLIDE 3: RANK ACCEPTABILITY INDEX (RAI) HISTOGRAMS ---
plt.clf()
plt.subplots(2, 2)

for name, pos in positions.items():
    plt.subplot(*pos)
    rai_top_30 = best_portfolios_rai[name][:30] * 100 
    ranks = [str(i+1) for i in range(30)]
    plt.bar(ranks, rai_top_30, color=colors[name])
    plt.title(f"{name} Best Portfolio RAI (%)")

plt.plotsize(100, 35)
print("\n--- GRAPH SET 3: RANK ACCEPTABILITY HISTOGRAMS ---")
plt.show()
print("\nAnalysis Complete! Exiting.")
