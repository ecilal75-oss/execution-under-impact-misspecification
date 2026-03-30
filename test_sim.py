import sys
sys.path.append('src')
from riccati import OWParams
from simulation import simulate_optimal_strategy, compute_internalization

# Test martingale flow
params = OWParams(theta=0.0, sigma=0.1)
results = simulate_optimal_strategy(params, n_paths=500)
intern = compute_internalization(results)

print(f'Mean cost: {results["mean_cost"]:.4f}')
print(f'Internalization: {intern*100:.1f}%')
print('Simulation OK')