import csv
import os
import numpy as np
import matplotlib.pyplot as plt

# --- Quantum Wavefunction (Bohr's Layer) ---
def psi_gaussian(x, mu=0, sigma=1):
	"""
	Gaussian wavefunction ψ(x), normalized.
	"""
	normalization = (1 / (sigma * np.sqrt(2 * np.pi))) ** 0.5
	return normalization * np.exp(- (x - mu)**2 / (4 * sigma**2))

# --- Deterministic Hidden Variable (Einstein's Layer) ---
def hidden_variable_rule(x, lambda_val):
	"""
	Einstein-style deterministic rule:
	Particle always chooses a position x where (x * λ) is an integer.
	"""
	return (x * lambda_val) % 1 < 0.05  # 5% tolerance for float error

# --- Measurement Simulation ---
def hybrid_measurement(mu=0, sigma=1, lambda_val=3, num_samples=1000):
	x_vals = np.linspace(-5, 5, num_samples)
	psi_vals = psi_gaussian(x_vals, mu, sigma)
	prob_density = np.abs(psi_vals) ** 2

	# Normalize probability density
	prob_density /= np.sum(prob_density)

	# Sample from the probability distribution (Bohr)
	x_measured = np.random.choice(x_vals, p=prob_density)

	# Check if it aligns with hidden variable rule (Einstein)
	if hidden_variable_rule(x_measured, lambda_val):
		accepted = True
	else:
		accepted = False

	return x_measured, accepted, prob_density, x_vals

# Create a directory to save the images if it doesn't exist
output_dir = 'measurement_plots'
os.makedirs(output_dir, exist_ok=True)

# CSV file for storing results
csv_filename = 'measurement_results.csv'

# Initialize CSV file with headers
if not os.path.exists(csv_filename):
	with open(csv_filename, mode='w', newline='') as file:
		writer = csv.writer(file)
		writer.writerow(['Iteration', 'Measured Position', 'Accepted by Hidden Variable Rule'])

for i in range(1000):
	# --- Run Simulation ---
	x, accepted, prob_density, x_vals = hybrid_measurement()
	
	# --- Output ---
	print(f"Measured position: {x:.4f}")
	print(f"Accepted by hidden variable rule? {'Yes' if accepted else 'No'}")
	
	# --- Save Results to CSV ---
	with open(csv_filename, mode='a', newline='') as file:
		writer = csv.writer(file)
		writer.writerow([i+1, f"{x:.4f}", 'Yes' if accepted else 'No'])

	# --- Plotting ---
	plt.plot(x_vals, prob_density, label='|ψ(x)|²')
	plt.axvline(x, color='r', linestyle='--', label='Measured x')
	if accepted:
		plt.scatter([x], [0], color='green', label='Accepted by λ')
	else:
		plt.scatter([x], [0], color='gray', label='Rejected by λ')
	plt.title("Einstein-Bohr Hybrid Measurement")
	plt.xlabel("Position x")
	plt.ylabel("Probability Density")
	plt.legend()
	plt.grid(True)

	# Save the plot to the specified directory
	plot_filename = os.path.join(output_dir, f"plot_{i+1:03d}.png")
	plt.savefig(plot_filename)
	plt.close()