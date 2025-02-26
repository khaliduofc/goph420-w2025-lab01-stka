import numpy as np
import matplotlib.pyplot as plt
import os
from goph420_lab01.integration import integrate_newton


# Function to load seismic velocity data
def load_seismic_data(filename):
    return np.loadtxt(filename)


# Set file path for the seismic data
data_path = os.path.join(os.path.dirname(__file__), "s_wave_data.txt")
data = load_seismic_data(data_path)

# Extract time and velocity values
time = data[:, 0]  
velocity = data[:, 1]  

# Dynamically determine T (last time where velocity > 0.5% max velocity)
threshold = 0.005 * np.max(np.abs(velocity))
valid_indices = np.where(np.abs(velocity) > threshold)[0]
T = time[valid_indices[-1]] if valid_indices.size > 0 else 10.0

# Filter time and velocity based on T
time_sampled = time[time <= T]
velocity_sampled = velocity[time <= T]

# Ensure Simpson's rule has an odd number of points
time_sampled_simpson = time_sampled.copy()
velocity_sampled_simpson = velocity_sampled.copy()
if len(time_sampled_simpson) % 2 == 0:
    time_sampled_simpson = time_sampled_simpson[:-1]  # Remove last point
    velocity_sampled_simpson = velocity_sampled_simpson[:-1]

# Compute integrals using Trapezoidal and Simpson’s rule
trap_integral = integrate_newton(time_sampled, velocity_sampled, "trap")
simp_integral = integrate_newton(time_sampled_simpson, velocity_sampled_simpson, "simp")

print(f"Trapezoidal Rule Integral: {trap_integral}")
print(f"Simpson’s Rule Integral: {simp_integral}")

# Convergence analysis: Ensure uniform spacing
num_intervals = np.logspace(1, 3, 10, dtype=int)  # Log-spaced intervals
trap_results = []
simp_results = []
errors_trap = []
errors_simp = []

for n in num_intervals:
    time_uniform = np.linspace(time_sampled[0], time_sampled[-1], n)
    velocity_uniform = np.interp(time_uniform, time_sampled, velocity_sampled)

    trap_val = integrate_newton(time_uniform, velocity_uniform, "trap")
    trap_results.append(trap_val)
    errors_trap.append(abs(trap_val - trap_integral) / trap_integral)  # Relative error

    if n % 2 == 1:
        simp_val = integrate_newton(time_uniform, velocity_uniform, "simp")
    else:
        n += 1  # Ensure odd number of points
        time_uniform = np.linspace(time_sampled[0], time_sampled[-1], n)
        velocity_uniform = np.interp(time_uniform, time_sampled, velocity_sampled)
        simp_val = integrate_newton(time_uniform, velocity_uniform, "simp")

    simp_results.append(simp_val)
    errors_simp.append(abs(simp_val - simp_integral) / simp_integral)

# Plot convergence curves
plt.figure()
plt.loglog(num_intervals, errors_trap, 'o-', label="Trapezoidal Rule Error")
plt.loglog(num_intervals, errors_simp, 's-', label="Simpson's Rule Error")
plt.xlabel("Number of Intervals")
plt.ylabel("Relative Error")
plt.title("Convergence of Integration Estimates")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

# Save the figure
figures_dir = os.path.join(os.path.dirname(__file__), "../figures")
os.makedirs(figures_dir, exist_ok=True)
plt.savefig(os.path.join(figures_dir, "integration_convergence.png"))
plt.show()

# Plot raw seismic data
plt.figure()
plt.plot(time, velocity, 'o-', label="Velocity Data")
plt.axvline(T, color='r', linestyle='--', label=f"T = {T:.2f}")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.title("Seismic Velocity Data")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(figures_dir, "seismic_data_plot.png"))
plt.show()

if __name__ == "__main__":
    print("Seismic data processing complete.")
