import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from goph420_lab01.integration import integrate_gauss

# Define normal probability density function
def normal_pdf(x, mean=1.5, std_dev=0.5):
    return norm.pdf(x, mean, std_dev)

# Compute exceedance probability using Gauss-Legendre quadrature
def compute_exceedance_probability(threshold, mean=1.5, std_dev=0.5, npts=3):
    upper_limit = mean + 6 * std_dev  # Approximate ∞ using 6σ rule
    lims = [threshold, upper_limit]
    prob = integrate_gauss(lambda x: normal_pdf(x, mean, std_dev), lims, npts)
    return prob if not np.isnan(prob) else 0.0  # Handle nan cases

# Compute probability of true value being in a given range
def compute_range_probability(lower, upper, mean, std_dev, npts=3):
    lims = [lower, upper]
    prob = integrate_gauss(lambda x: normal_pdf(x, mean, std_dev), lims, npts)
    return prob if not np.isnan(prob) else 0.0

# Main execution
if __name__ == "__main__":
    # Define problem parameters
    magnitude_threshold = 4.0
    distance_mean = 10.28
    distance_error = 0.05
    distance_lower = 10.25
    distance_upper = 10.35

    # Compute probabilities
    prob_exceedance = compute_exceedance_probability(magnitude_threshold)
    prob_range = compute_range_probability(distance_lower, distance_upper, distance_mean, distance_error)

    print(f"Probability of a seismic event with magnitude > {magnitude_threshold}: {prob_exceedance:.6f}")
    print(f"Probability of true distance between {distance_lower}m and {distance_upper}m: {prob_range:.6f}")

    # Convergence analysis
    integration_points = np.arange(1, 6)
    exceedance_probs = [compute_exceedance_probability(magnitude_threshold, npts=n) for n in integration_points]
    range_probs = [compute_range_probability(distance_lower, distance_upper, distance_mean, distance_error, npts=n) for n in integration_points]

    # Ensure figure saving directory exists
    script_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(script_dir, "../figures")
    os.makedirs(figures_dir, exist_ok=True)

    # Plot convergence
    plt.figure()
    plt.loglog(integration_points, exceedance_probs, 'o-', label="Exceedance Probability")
    plt.loglog(integration_points, range_probs, 's-', label="Range Probability")
    plt.xlabel("Number of Integration Points")
    plt.ylabel("Probability Estimate")
    plt.legend()
    plt.title("Convergence of Probability Estimation")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.savefig(os.path.join(figures_dir, "probability_convergence.png"))
    plt.show()
