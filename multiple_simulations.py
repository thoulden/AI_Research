import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def run():
    # === Multiple Simulations Code ===

    # Run Simulation Button
    run_simulation = st.sidebar.button('Run Simulation')
    
    # Simulation settings
    st.sidebar.subheader("Sampling Options")
    display_distributions = st.sidebar.checkbox('Display empirical distributions', key='display_distributions')
    
    # Add checkbox for enabling correlated sampling
    enable_correlation = st.sidebar.checkbox(
        r'Enable correlated sampling of $\beta_0$ and $f$', 
        key='enable_correlation'
    )

    # Option to compute growth
    compute_growth = st.sidebar.checkbox('Compute Growth', value=True)

    # Conditional input for noise standard deviation
    if enable_correlation:
        noise_std = st.sidebar.number_input(
            'Noise Standard Deviation (σ)',
            min_value=0.0,
            max_value=2.0,
            value=0.4,
            step=0.1,
            format="%.2f",
            help='Determines the variability around the linear relationship in log-space. Higher values introduce more randomness.',
            key='noise_std'
        )
    else:
        noise_std = 0.0

    # Time settings
    delta_t = st.sidebar.number_input('Time step in years', min_value=0.0001, max_value=1.0, value=0.01, step=0.0001)
    T = st.sidebar.number_input('Total simulation time in years (T)', min_value=0.1, max_value=10.0, value=4.0, step=0.1)
    num_steps = int(T / delta_t) + 1
    time = np.linspace(0, T, num=num_steps, endpoint=True)
    
    # Display Options
    st.sidebar.subheader("Display Options")
    # Removed smoothing logic

    # Parameter distributions
    lambda_min = st.sidebar.number_input('Minimum Lambda (λ_min)', min_value=0.01, max_value=1.0, value=0.2, step=0.01)
    lambda_max = st.sidebar.number_input('Maximum Lambda (λ_max)', min_value=lambda_min, max_value=1.0, value=0.8, step=0.01)
    D_min = st.sidebar.number_input('Minimum D (D_min)', min_value=1e6, max_value=1e12, value=1e7, step=1e6, format="%.0e")
    D_max = st.sidebar.number_input('Maximum D (D_max)', min_value=D_min, max_value=1e12, value=1e11, step=1e6, format="%.0e")
    beta_0_min = st.sidebar.number_input('Minimum Beta_0 (β₀_min)', min_value=0.01, max_value=1.0, value=0.15, step=0.01)
    beta_0_max = st.sidebar.number_input('Maximum Beta_0 (β₀_max)', min_value=beta_0_min, max_value=1.0, value=0.75, step=0.01)
    f_min = st.sidebar.number_input('Minimum f (f_min)', min_value=1.0, max_value=100.0, value=2.0, step=0.1)
    f_max = st.sidebar.number_input('Maximum f (f_max)', min_value=f_min, max_value=100.0, value=32.0, step=0.1)

    # Fixed parameters
    R_bar = st.sidebar.number_input('R_bar', min_value=0.0, max_value=100000.0, value=1.0, step=0.1)
    C_0 = st.sidebar.number_input('C_0', min_value=0.0, max_value=100000.0, value=1.0, step=0.1)
    g = st.sidebar.number_input('Growth rate (g)', min_value=0.1, max_value=10.0, value=2.77, step=0.01)
    alpha = st.sidebar.number_input('Alpha (α)', min_value=0.01, max_value=0.99, value=.5, step=0.01)

    # Number of simulations
    num_simulations = st.sidebar.number_input('Number of simulations', min_value=1, max_value=10000, value=1000, step=1)

    # Multipliers for the second scenario
    multipliers_2 = st.sidebar.multiselect(
        'Select multipliers for the second CDF (multipliers_2)',
        options=[1, 3, 10, 30, 50, 100],
        default=[3, 10, 30]
    )

    # Define intervals (in years) and their labels
    intervals = [1/12, 4/12, 1, 3]  # 1 month, 4 months, 1 year, 3 years
    interval_labels = ['1 month', '4 months', '1 year', '3 years']

    # Prepare arrays to count how many simulations meet the condition for each interval and multiplier
    interval_counts = np.zeros((len(multipliers_2), len(intervals)))

    # Initialize lists to store parameters (optional, for distributions)
    lambda_samples = []
    D_samples = []
    beta_0_samples = []
    f_samples = []
    
    if run_simulation:
        # counts_over_time_2: tracks fraction over time
        counts_over_time_2 = np.zeros((len(multipliers_2), num_steps - 1))
        progress_bar = st.progress(0)
        status_text = st.empty()

        # If correlation is enabled, pre-compute correlation coefficients
        if enable_correlation:
            log_beta0_min = np.log(beta_0_min)
            log_beta0_max = np.log(beta_0_max)
            log_f_min = np.log(f_min)
            log_f_max = np.log(f_max)
            b = (log_beta0_max - log_beta0_min) / (log_f_max - log_f_min)
            a = log_beta0_min - b * log_f_min

        for sim in range(int(num_simulations)):
            # Sample parameters
            if enable_correlation:
                # Sample f
                log_f = np.random.uniform(np.log(f_min), np.log(f_max))
                f_sample = np.exp(log_f)
                # Sample beta_0 with correlation
                log_beta0 = a + b * log_f + np.random.normal(0, noise_std)
                beta_0_sample = np.exp(log_beta0)
                beta_0_sample = np.clip(beta_0_sample, beta_0_min, beta_0_max)
            else:
                f_sample = np.exp(np.random.uniform(np.log(f_min), np.log(f_max)))
                beta_0_sample = np.exp(np.random.uniform(np.log(beta_0_min), np.log(beta_0_max)))

            lambda_sample = np.exp(np.random.uniform(np.log(lambda_min), np.log(lambda_max)))
            D_sample = np.exp(np.random.uniform(np.log(D_min), np.log(D_max)))

            # Store samples
            lambda_samples.append(lambda_sample)
            D_samples.append(D_sample)
            beta_0_samples.append(beta_0_sample)
            f_samples.append(f_sample)

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
                if compute_growth:
                    C[t] = C[t - 1] * (1 + delta_t * g * beta_0_sample / (lambda_sample * (1 - alpha)))
                else:
                    C[t] = C_0
                
                # Non-accelerated case
                beta_S[t] = beta_0_sample * (1 - ((S_values[t - 1] / S_bar - 1) / (D_sample - 1))) ** (-1)
                S_values[t] = S_values[t - 1] + delta_t * (R_bar ** (lambda_sample * alpha)) * (C[t-1] ** (lambda_sample * (1-alpha))) * (S_values[t - 1] ** (1- beta_S[t-1]))

                # Accelerated case
                beta_SA[t] = beta_0_sample * (1 - ((S_valuesA[t - 1] / S_bar - 1) / (D_sample - 1))) ** (-1)
                S_valuesA[t] = S_valuesA[t - 1] + delta_t * (Researchers[t-1] ** (lambda_sample * alpha)) * (C[t] ** (lambda_sample * (1-alpha))) * S_valuesA[t - 1] ** (1 - beta_SA[t])
                Researchers[t] = (R_bar + upsilon * S_valuesA[t])

                # Exponential case
                S_values_Exp[t] = S_values_Exp[t - 1] * (1 + delta_t * g)

            # Calculate growth rates for Accelerated case
            S_valuesA_netend = S_valuesA[:-1]
            g_S_valuesA = np.diff(S_valuesA) / (S_valuesA_netend * delta_t)

            # Compare g_S_valuesA to multiplier * g
            for m_idx, multiplier in enumerate(multipliers_2):
                # Counts over time
                counts_over_time_2[m_idx] += g_S_valuesA > (multiplier * g)

            # Check for continuous periods
            # We want to find if there's ANY continuous segment of length interval_steps
            # where g_S_valuesA > multiplier*g at all steps in that segment.
            for m_idx, multiplier in enumerate(multipliers_2):
                for i_idx, interval in enumerate(intervals):
                    # Convert interval in years to number of steps
                    interval_steps = max(int(interval / delta_t), 1)
                    
                    condition_met = False
                    max_start = (len(g_S_valuesA) - interval_steps)
                    if max_start < 0:
                        # Interval longer than simulation duration
                        continue
                    
                    for start_idx in range(max_start + 1):
                        a_window = g_S_valuesA[start_idx:start_idx+interval_steps]
                        # Check if all values exceed threshold in this window
                        if np.all(a_window > multiplier * g):
                            condition_met = True
                            break
                    
                    if condition_met:
                        interval_counts[m_idx, i_idx] += 1

            # Update progress bar
            if sim % max(int(num_simulations / 100), 1) == 0:
                progress_bar.progress(sim / num_simulations)
                status_text.text(f'Running simulation {sim}/{int(num_simulations)}')

        progress_bar.empty()
        status_text.text('Simulation completed.')

        # Compute fractions for the table
        fractions_table = interval_counts / num_simulations * 100
        column_labels = [f"{m}x" for m in multipliers_2]
        df_table = pd.DataFrame(fractions_table.T, columns=column_labels, index=interval_labels)

        st.markdown("### Probability Table (Any Continuous Period)")
        st.table(df_table.style.format("{:.0f}%"))

        # Calculate fractions over time
        fractions_over_time_2 = counts_over_time_2 / num_simulations

        # Plot fractions over time (no smoothing)
        fig, ax = plt.subplots(figsize=(10, 6))
        for m_idx, multiplier in enumerate(multipliers_2):
            ax.plot(time[:-1], fractions_over_time_2[m_idx], label=f'{multiplier}x')
        ax.set_xlabel('Years')
        ax.set_ylabel('Fraction where g_S_valuesA > multiplier × g')
        ax.set_title('Cumulative Fraction of Simulations Over Time')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # Display distributions if requested
        if display_distributions:
            st.markdown("##### Empirical Distributions of Sampled Parameters")
            fig_hist, axs = plt.subplots(2, 2, figsize=(12, 10))
            bin_edges_lambda = np.logspace(np.log10(lambda_min), np.log10(lambda_max), num=20)
            bin_edges_D = np.logspace(np.log10(D_min), np.log10(D_max), num=20)
            bin_edges_beta_0 = np.logspace(np.log10(beta_0_min), np.log10(beta_0_max), num=20)
            bin_edges_f = np.logspace(np.log10(f_min), np.log10(f_max), num=20)

            axs[0, 0].hist(lambda_samples, bins=bin_edges_lambda, edgecolor='black')
            axs[0, 0].set_title('Distribution of Parallelizability (λ)')
            axs[0, 0].set_xlabel('Lambda (λ)')
            axs[0, 0].set_ylabel('Frequency')

            axs[0, 1].hist(D_samples, bins=bin_edges_D, edgecolor='black')
            axs[0, 1].set_xscale('log')
            axs[0, 1].set_title(r'Distribution of Ceiling Term ($D$, log scale)')
            axs[0, 1].set_xlabel('D')
            axs[0, 1].set_ylabel('Frequency')

            axs[1, 0].hist(beta_0_samples, bins=bin_edges_beta_0, edgecolor='black')
            axs[1, 0].set_title('Distribution of Initial Diminishing Returns (β₀)')
            axs[1, 0].set_xlabel('β₀')
            axs[1, 0].set_ylabel('Frequency')

            axs[1, 1].hist(f_samples, bins=bin_edges_f, edgecolor='black')
            axs[1, 1].set_xscale('log')
            axs[1, 1].set_title(r'Distribution of Speed Up ($f$, log scale)')
            axs[1, 1].set_xlabel('f')
            axs[1, 1].set_ylabel('Frequency')

            plt.tight_layout()
            st.pyplot(fig_hist)

            if enable_correlation:
                st.markdown(r"##### Correlation between $\beta_0$ and f")
                fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6))
                ax_scatter.scatter(f_samples, beta_0_samples, alpha=0.5, edgecolor='k', linewidth=0.5)
                ax_scatter.set_xlabel('f')
                ax_scatter.set_ylabel(r'$\beta_0$')
                ax_scatter.set_title(r'Scatter Plot of $\beta_0$ vs. f')
                ax_scatter.grid(True)
                st.pyplot(fig_scatter)
