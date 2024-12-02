# Simulation settings
delta_t = 0.001  # Time step in years
T = 4            # Total simulation time in years
time = np.arange(0, T, delta_t)  # Time vector
num_steps = len(time)

# Parameters for distributions
lambda_min, lambda_max = 0.2, 0.8
D_min, D_max = 1e7, 1e11
beta_0_min, beta_0_max = 0.15, 0.75
f_min, f_max = 2, 32

# Fixed parameters
R_bar = 1
C_0 = 1
g = 2.77
alpha=1

# Number of simulations
num_simulations = 1000  # Adjust based on your computational resources

# Define the multipliers
multipliers_2 = [3, 10, 30]    # For the second CDF

# Initialize counts over time for each multiplier
counts_over_time_2 = np.zeros((len(multipliers_2), num_steps - 1))

# Loop over simulations
for sim in range(num_simulations):
    # Sample parameters from log-uniform distributions
    lambda_sample = np.exp(np.random.uniform(np.log(lambda_min), np.log(lambda_max)))
    D_sample = np.exp(np.random.uniform(np.log(D_min), np.log(D_max)))
    beta_0_sample = np.exp(np.random.uniform(np.log(beta_0_min), np.log(beta_0_max)))
    f_sample = np.exp(np.random.uniform(np.log(f_min), np.log(f_max)))

    # Compute derived parameters
    S_bar = (g * R_bar ** (-lambda_sample * alpha) * C_0 ** (-lambda_sample * (1-alpha))) ** (-1 / beta_0_sample)
    R0 = f_sample ** (1 / (lambda_sample * alpha)) * R_bar
    upsilon = R_bar * (f_sample ** (1 / (lambda_sample * alpha)) - 1) / S_bar

    # Initialize arrays
    C = np.zeros(num_steps)
    Researchers = np.zeros(num_steps)
    S_valuesA = np.zeros(num_steps)
    beta_SA = np.zeros(num_steps)
    g_S_valuesA = np.zeros(num_steps - 1)
    S_values = np.zeros(num_steps)
    beta_S = np.zeros(num_steps)
    g_S_values = np.zeros(num_steps - 1)
    S_values_Exp = np.zeros(num_steps)

    # Initial conditions
    C[0] = C_0
    S_valuesA[0] = S_bar
    beta_SA[0] = beta_0_sample
    S_values[0] = S_bar
    beta_S[0] = beta_0_sample
    S_values_Exp[0] = S_bar
    Researchers[0] = f_sample ** (1/(lambda_sample * alpha)) * R_bar

    # Simulation loop
    for t in range(1, num_steps):
      # Non-accelerated case
      beta_S[t] = beta_0_sample * (1 - ((S_values[t - 1] / S_bar - 1) / (D_sample - 1))) ** (-1)
      C[t] = C[t-1] * (1+ delta_t * g * beta_0_sample / (lambda_sample * (1-alpha)))
      S_values[t] = S_values[t - 1] + delta_t * (R_bar ** (lambda_sample * alpha))* (C[t-1] ** (lambda_sample * (1-alpha))) * (S_values[t - 1] ** (1- beta_S[t-1]))

      # Accelerated case
      beta_SA[t] = beta_0_sample * (1 - ((S_valuesA[t - 1] / S_bar - 1) / (D_sample - 1))) ** (-1)
      S_valuesA[t] = S_valuesA[t - 1] + delta_t * Researchers[t-1] ** (lambda_sample * alpha) * C[t-1] ** (lambda_sample * (1-alpha)) * S_valuesA[t - 1] ** (1 - beta_SA[t])
      Researchers[t] = (R_bar + upsilon * S_valuesA[t])

      # Exponential case
      S_values_Exp[t] = S_values_Exp[t - 1] * (1 + delta_t * g)

    # Calculate growth rates
    S_valuesA_netend = S_valuesA[:-1]
    g_S_valuesA = np.diff(S_valuesA) / (S_valuesA_netend * delta_t)
    S_values_netend = S_values[:-1]
    g_S_values = np.diff(S_values) / (S_values_netend * delta_t)

    # Collect counts over time for the first case
    #for m_idx, multiplier in enumerate(multipliers):
     #   counts_over_time[m_idx] += g_S_valuesA > multiplier * 2.77

    # Collect counts over time for the second case
    for m_idx, multiplier in enumerate(multipliers_2):
        counts_over_time_2[m_idx] += g_S_valuesA > multiplier * g_S_values

# Calculate the fractions (CDF) over time
fractions_over_time_2 = counts_over_time_2 / num_simulations

# Apply smoothing to the fractions_over_time using a moving average filter
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='same')

# Define the window size for smoothing (adjust as needed)
window_size = 70  # For example, using a window size of 70 time steps

# Apply smoothing to each multiplier's data (second case)
fractions_smoothed_2 = np.zeros_like(fractions_over_time_2)
for m_idx in range(len(multipliers_2)):
    fractions_smoothed_2[m_idx] = moving_average(fractions_over_time_2[m_idx], window_size)

# Create a time mask to exclude data before a certain time
time_mask = time[:-1] >= 0.03  # Exclude data before 0.05 years


# Plot the smoothed CDFs (Second Case)
plt.figure(figsize=(10, 6))
for m_idx, multiplier in enumerate(multipliers_2):
    plt.plot(time[:-1][time_mask], fractions_smoothed_2[m_idx][time_mask], label=f'{multiplier}x')
plt.xlabel('Years')
plt.ylabel('Fraction where g_S_valuesA > multiplier × g_S_values')
plt.title('Fraction of Simulations where Accelerated Case growth exceeds Base case growth over time')
plt.legend()
plt.grid(True)
plt.show()
