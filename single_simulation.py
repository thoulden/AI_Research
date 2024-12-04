import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def run():
    # === Single Simulation Code ===
    # Run Simulation Button
    run_simulation = st.sidebar.button('Run Simulation')
    
    # Simulation settings
    delta_t = st.sidebar.number_input('Time step in years (delta_t)', min_value=0.0001, max_value=1.0, value=0.01, step=0.0001)
    T = st.sidebar.number_input('Total simulation time in years (T)', min_value=0.1, max_value=20.0, value=8.0, step=0.1)
    time = np.arange(0, T, delta_t)
    num_steps = len(time)

    # Fixed parameters
    R_bar = st.sidebar.number_input('R_bar', min_value=0.0, max_value=100000.0, value=1.0, step=0.1)
    C_0 = st.sidebar.number_input('C_0', min_value=0.0, max_value=100000.0, value=1.0, step=0.1)
    g = st.sidebar.number_input('Growth rate (g)', min_value=0.1, max_value=10.0, value=2.77, step=0.01)
    alpha = st.sidebar.number_input('Labor share of research (α)', min_value=0.0, max_value=1.0, value=0.5, step=0.01)

    # Parameters for the simulation
    lambda_sample = st.sidebar.number_input('Parallelizability (λ)', min_value=0.01, max_value=1.0, value=0.7, step=0.01)
    beta_0_sample = st.sidebar.number_input('Diminishing Returns Initial (β₀)', min_value=0.01, max_value=1.0, value=0.7, step=0.01)
    D_sample = st.sidebar.number_input('Ceiling (D)', min_value=1e6, max_value=1e12, value=1e7, step=1e6, format="%.0e")
    f_sample = st.sidebar.number_input('Acceleration term (f)', min_value=1.0, max_value=100.0, value=8.0, step=0.1)

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
        ceiling = np.full(num_steps, S_bar*D_sample) 
        
        # Simulation loop
        for t in range(1, num_steps):
            # Non-accelerated case
            beta_S[t] = beta_0_sample * (1 - ((S_values[t - 1] / S_bar - 1) / (D_sample - 1))) ** (-1)
            C[t] = C[t - 1] * (1 + delta_t * g * beta_0_sample / (lambda_sample * (1 - alpha)))
            S_values[t] = S_values[t - 1] + delta_t * R_bar ** (lambda_sample * alpha) * C[t - 1] ** (lambda_sample * (1 - alpha)) * (S_values[t - 1] ** (1 - beta_S[t - 1]))

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
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.semilogy(time, S_valuesA, '-', label='Accelerate') # Accelerate line
        ax1.semilogy(time, S_values, '-', label='Base') # Base case line
        ax1.semilogy(time, S_values_Exp, 'r--', label='Exp') # Exponential line
        ax1.semilogy(time, ceiling, 'black', linewidth=0.5)  # Ceiling line
        label_position = (time[2], ceiling[2])  # Adjust to your preferred position
        ax1.text(label_position[0], label_position[1], 'Ceiling', fontsize=8, color='black')

        ax1.set_xlabel('Time')
        ax1.set_ylabel('S(t)')
        ax1.set_title('Software level over Time (Log Scale)')
        ax1.grid(True)
        ax1.legend()
        st.pyplot(fig1)

        # Second figure: Beta values
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(time, beta_SA, '-', label='Accelerate')
        ax2.plot(time, beta_S, '-', label='Base')
        ax2.set_xlabel('Time')
        ax2.set_ylabel(r'$\beta_S(t)$')
        ax2.set_title('Diminishing Returns over Time')
        ax2.grid(True)
        ax2.legend()
        st.pyplot(fig2)

        # Third figure: Growth Rate Comparison
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        ax3.plot(time[:-1], g_S_valuesA, '-', label='Accelerate')
        ax3.plot(time[:-1], g_S_values, '-', label='Base')
        ax3.plot(time[:-1], g * np.ones(len(time[:-1])), 'r--', label='Exponential')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('g')
        ax3.set_title('Growth Rate of Software -- Comparison')
        ax3.grid(True)
        ax3.legend()
        st.pyplot(fig3)

