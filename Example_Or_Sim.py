import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title('Simulation of Accelerated Growth Model')

# Simulation Mode Selector
simulation_mode = st.sidebar.selectbox(
    "Select Simulation Mode",
    ("Multiple Simulations", "Single Simulation")
)

if simulation_mode == "Multiple Simulations":
    # === Multiple Simulations Code ===

    # Simulation settings
    delta_t = st.sidebar.number_input('Time step in years (delta_t)', min_value=0.0001, max_value=1.0, value=0.001, step=0.0001)
    T = st.sidebar.number_input('Total simulation time in years (T)', min_value=0.1, max_value=10.0, value=4.0, step=0.1)
    time = np.arange(0, T, delta_t)
    num_steps = len(time)

    # Parameters for distributions
    lambda_min = st.sidebar.number_input('Minimum Lambda (λ_min)', min_value=0.01, max_value=1.0, value=0.2, step=0.01)
    lambda_max = st.sidebar.number_input('Maximum Lambda (λ_max)', min_value=lambda_min, max_value=1.0, value=0.8, step=0.01)
    D_min = st.sidebar.number_input('Minimum D (D_min)', min_value=1e6, max_value=1e12, value=1e7, step=1e6, format="%.0e")
    D_max = st.sidebar.number_input('Maximum D (D_max)', min_value=D_min, max_value=1e12, value=1e11, step=1e6, format="%.0e")
    beta_0_min = st.sidebar.number_input('Minimum Beta_0 (β₀_min)', min_value=0.01, max_value=1.0, value=0.15, step=0.01)
    beta_0_max = st.sidebar.number_input('Maximum Beta_0 (β₀_max)', min_value=beta_0_min, max_value=1.0, value=0.75, step=0.01)
    f_min = st.sidebar.number_input('Minimum f (f_min)', min_value=1.0, max_value=100.0, value=2.0, step=0.1)
    f_max = st.sidebar.number_input('Maximum f (f_max)', min_value=f_min, max_value=100.0, value=32.0, step=0.1)

    # Fixed parameters
    R_bar = 1
    C_0 = 1
    g = st.sidebar.number_input('Growth rate (g)', min_value=0.1, max_value=10.0, value=2.77, step=0.01)
    alpha = st.sidebar.number_input('Alpha (α)', min_value=0.0, max_value=1.0, value=1.0, step=0.01)

    # Number of simulations
    num_simulations = st.sidebar.number_input('Number of simulations', min_value=1, max_value=10000, value=1000, step=1)

    # Multipliers
    multipliers_2 = st.sidebar.multiselect(
        'Select multipliers for the second CDF (multipliers_2)',
        options=[3, 10, 30, 50, 100],
        default=[3, 10, 30]
    )

    # Run Simulation Button
    run_simulation = st.sidebar.button('Run Simulation')

    if run_simulation:
        counts_over_time_2 = np.zeros((len(multipliers_2), num_steps - 1))
        progress_bar = st.progress(0)
        status_text = st.empty()

        for sim in range(int(num_simulations)):
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
            S_values = np.zeros(num_steps)
            beta_S = np.zeros(num_steps)
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
                S_valuesA[t] = S_valuesA[t - 1] + delta_t * Researchers[t-1] ** (lambda_sample * alpha) * C[t] ** (lambda_sample * (1-alpha)) * S_valuesA[t - 1] ** (1 - beta_SA[t])
                Researchers[t] = (R_bar + upsilon * S_valuesA[t])

                # Exponential case
                S_values_Exp[t] = S_values_Exp[t - 1] * (1 + delta_t * g)

            # Calculate growth rates
            S_valuesA_netend = S_valuesA[:-1]
            g_S_valuesA = np.diff(S_valuesA) / (S_valuesA_netend * delta_t)
            S_values_netend = S_values[:-1]
            g_S_values = np.diff(S_values) / (S_values_netend * delta_t)

            # Collect counts over time for the second case
            for m_idx, multiplier in enumerate(multipliers_2):
                counts_over_time_2[m_idx] += g_S_valuesA > multiplier * g_S_values

            # Update progress bar
            if sim % max(int(num_simulations / 100), 1) == 0:
                progress_bar.progress(sim / num_simulations)
                status_text.text(f'Running simulation {sim}/{int(num_simulations)}')

        progress_bar.empty()
        status_text.text('Simulation completed.')

        # Calculate the fractions (CDF) over time
        fractions_over_time_2 = counts_over_time_2 / num_simulations

        # Apply smoothing to the fractions_over_time using a moving average filter
        def moving_average(data, window_size):
            return np.convolve(data, np.ones(window_size) / window_size, mode='same')

        # Define the window size for smoothing (adjust as needed)
        window_size = st.sidebar.slider('Smoothing window size', min_value=1, max_value=200, value=70, step=1)

        # Apply smoothing to each multiplier's data (second case)
        fractions_smoothed_2 = np.zeros_like(fractions_over_time_2)
        for m_idx in range(len(multipliers_2)):
            fractions_smoothed_2[m_idx] = moving_average(fractions_over_time_2[m_idx], window_size)

        # Create a time mask to exclude data before a certain time
        exclude_time = st.sidebar.number_input('Exclude data before (years)', min_value=0.0, max_value=float(T), value=0.03, step=0.01)
        time_mask = time[:-1] >= exclude_time

        # Plot the smoothed CDFs (Second Case)
        fig, ax = plt.subplots(figsize=(10, 6))
        for m_idx, multiplier in enumerate(multipliers_2):
            ax.plot(time[:-1][time_mask], fractions_smoothed_2[m_idx][time_mask], label=f'{multiplier}x')
        ax.set_xlabel('Years')
        ax.set_ylabel('Fraction where g_S_valuesA > multiplier × g_S_values')
        ax.set_title('Fraction of Simulations where Accelerated Case growth exceeds Base case growth over time')
        ax.legend()
        ax.grid(True)

        # Display the plot in Streamlit
        st.pyplot(fig)

elif simulation_mode == "Single Simulation":
    # === Single Simulation Code ===

    # Simulation settings
    delta_t = st.sidebar.number_input('Time step in years (delta_t)', min_value=0.0001, max_value=1.0, value=0.01, step=0.0001)
    T = st.sidebar.number_input('Total simulation time in years (T)', min_value=0.1, max_value=20.0, value=8.0, step=0.1)
    time = np.arange(0, T, delta_t)
    num_steps = len(time)

    # Fixed parameters
    R_bar = st.sidebar.number_input('R_bar', min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    C_0 = st.sidebar.number_input('C_0', min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    g = st.sidebar.number_input('Growth rate (g)', min_value=0.1, max_value=10.0, value=2.77, step=0.01)
    alpha = st.sidebar.number_input('Alpha (α)', min_value=0.0, max_value=1.0, value=0.9, step=0.01)

    # Parameters for the simulation
    lambda_sample = st.sidebar.number_input('Lambda (λ)', min_value=0.01, max_value=1.0, value=0.7, step=0.01)
    beta_0_sample = st.sidebar.number_input('Beta_0 (β₀)', min_value=0.01, max_value=1.0, value=0.7, step=0.01)
    D_sample = st.sidebar.number_input('D', min_value=1e6, max_value=1e12, value=1e7, step=1e6, format="%.0e")
    f_sample = st.sidebar.number_input('f', min_value=1.0, max_value=100.0, value=8.0, step=0.1)

    # Run Simulation Button
    run_simulation = st.sidebar.button('Run Simulation')

    if run_simulation:
        # Compute derived parameters
        S_bar = (g * R_bar ** (-lambda_sample * alpha) * C_0 ** (-lambda_sample * (1 - alpha))) ** (-1 / beta_0_sample)
        R0 = f_sample ** (1 / (lambda_sample * alpha)) * R_bar
        upsilon = R_bar * (f_sample ** (1 / (lambda_sample * alpha)) - 1) / S_bar

        # Initialize arrays
        C = np.zeros(num_steps)
        Researchers = np.zeros(num_steps)
        S_valuesA = np.zeros(num_steps)
        beta_SA = np.zeros(num_steps)
        S_values = np.zeros(num_steps)
        beta_S = np.zeros(num_steps)
        S_values_Exp = np.zeros(num_steps)

        # Initial conditions
        C[0] = C_0
        S_valuesA[0] = S_bar
        beta_SA[0] = beta_0_sample
        S_values[0] = S_bar
        beta_S[0] = beta_0_sample
        S_values_Exp[0] = S_bar
        Researchers[0] = f_sample ** (1 / (lambda_sample * alpha)) * R_bar

        # Simulation loop
        for t in range(1, num_steps):
            # Non-accelerated case
            beta_S[t] = beta_0_sample * (1 - ((S_values[t - 1] / S_bar - 1) / (D_sample - 1))) ** (-1)
            C[t] = C[t - 1] * (1 + delta_t * g * beta_0_sample / (lambda_sample * (1 - alpha)))
            S_values[t] = S_values[t - 1] + delta_t * (R_bar ** (lambda_sample * alpha)) * (C[t - 1] ** (lambda_sample * (1 - alpha))) * (S_values[t - 1] ** (1 - beta_S[t - 1]))

            # Accelerated case
            beta_SA[t] = beta_0_sample * (1 - ((S_valuesA[t - 1] / S_bar - 1) / (D_sample - 1))) ** (-1)
            S_valuesA[t] = S_valuesA[t - 1] + delta_t * Researchers[t - 1] ** (lambda_sample * alpha) * C[t] ** (lambda_sample * (1 - alpha)) * S_valuesA[t - 1] ** (1 - beta_SA[t])
            Researchers[t] = (R_bar + upsilon * S_valuesA[t])

            # Exponential case
            S_values_Exp[t] = S_values_Exp[t - 1] * (1 + delta_t * g)

        # Calculate Growth Rates
        S_valuesA_netend = S_valuesA[:-1]
        g_S_valuesA = np.diff(S_valuesA) / (S_valuesA_netend * delta_t)
        S_values_netend = S_values[:-1]
        g_S_values = np.diff(S_values) / (S_values_netend * delta_t)
        C_netend = C[:-1]
        g_C_values = np.diff(C) / (C_netend * delta_t)

        # First figure: Log plot of S_values over time
        fig1, axs = plt.subplots(2, 1, figsize=(10, 8))

        # Subplot 1: S_values over time (Log Scale)
        axs[0].semilogy(time, S_valuesA, '-', label='Accelerate')
        axs[0].semilogy(time, S_values, '-', label='Base')
        axs[0].semilogy(time, S_values_Exp, '-', label='Exp')
        axs[0].set_xlabel('Time')
        axs[0].set_ylabel('S(t)')
        axs[0].set_title('Simulation of S over Time (Log Scale)')
        axs[0].grid(True)
        axs[0].legend()

        # Subplot 2: beta_S over time
        axs[1].plot(time, beta_SA, '-', label='Accelerate')
        axs[1].plot(time, beta_S, '-', label='Base')
        axs[1].set_xlabel('Time')
        axs[1].set_ylabel(r'$\beta_S(t)$')
        axs[1].set_title('Beta_S over Time')
        axs[1].grid(True)
        axs[1].legend()

        # Display the first figure
        st.pyplot(fig1)

        # Second figure: Growth Rate Comparison
        fig2, ax = plt.subplots(figsize=(10, 6))
        ax.plot(time[:-1], g_S_valuesA, '-', label='Accelerate')
        ax.plot(time[:-1], g_S_values, '-', label='Base')
        ax.plot(time[:-1], g * np.ones(len(time[:-1])), 'r--', label='Exponential')
        ax.set_xlabel('Time')
        ax.set_ylabel('g')
        ax.set_title('Growth Rate Comparison')
        ax.grid(True)
        ax.legend()

        # Display the second figure
        st.pyplot(fig2)
