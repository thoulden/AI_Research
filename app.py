import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the other modules
import math_appendix
import multiple_simulations
import single_simulation

# Navigation link at the top right
col1, col2 = st.columns([8, 1])
with col1:
    st.write("")  # Empty placeholder to adjust alignment
with col2:
    st.markdown("[Math Appendix](?page=math_appendix)")

## parameters to put in table
def get_parameters_table():
    data = [
        {
            'Parameter': 'f',
            'Description': 'Initial shock to brainpower dedicated to AI research',
            'Low Estimate': 2,
            'Median Estimate': 8,
            'High Estimate': 32
        },
        {
            'Parameter': r'$\lambda$',
            'Description': 'Parallelizability of research',
            'Low Estimate': 0.2,
            'Median Estimate': 0.5,
            'High Estimate': 0.8
        },
        {
            'Parameter': r'$\beta$',
            'Description': 'Diminishing returns in discovering new ideas',
            'Low Estimate': 0.15,
            'Median Estimate': 0.45,
            'High Estimate': 0.75
        },
        {
            'Parameter': r'$\alpha$',
            'Description': 'Contribution of cognitive labor vs. compute to AI R&D',
            'Low Estimate': 0.0,
            'Median Estimate': 0.5,
            'High Estimate': 1.0
        },
        {
            'Parameter': 'g',
            'Description': 'Growth rate of software when GPT-6 is deployed',
            'Low Estimate': 2.0,
            'Median Estimate': 2.77,
            'High Estimate': 3.0
        },
        {
            'Parameter': r'$S_{\text{ceiling}}$',
            'Description': 'Ceiling on the level of software',
            'Low Estimate': '1e7',
            'Median Estimate': '1e8',
            'High Estimate': '1e9'
        },
    ]
    df = pd.DataFrame(data)
    return df


params = st.experimental_get_query_params()
if params.get('page') == ['math_appendix']:
    # Display Math Appendix
    math_appendix.display()
else:
    # Main Page Content
    st.title('Simulation of Accelerated Growth Model')

    st.markdown(r"""
    This tool analyzes the impact of automating AI R&D. The key parameters for this tool are: size of initial shock to the amount of brainpower dedicated to AI research ($f$), the parallelizability of research ($\lambda$), the level of diminishing returns for discovering new ideas ($\beta$), the share contribution of cognitive labor vs compute to AI R&D ($\alpha$), the growth rate of software when GPT-6 is deployed ($g$), and the ceiling on the level of software.

    This tool offers two options. You can run a bunch of simulations with uncertainty over key parameters—the output of this function will be a plot which shows the fraction of simulations where the growth rate of software exceeds the observed exponential rate by some amount over some number of years. Alternatively, you can run a single simulation under specific parameter values to illustrate the path of AI progress. Under this second option, you will also see the change in the level of diminishing research productivity over time and the growth rates compared to (a) constant exponential progress and (b) the projected path of software without deploying AI to research.

    As above, I allow the stock of compute to be growing so that, at the time of GPT-6, growth of the software level looks exponential. Now, I also allow AI to be deployed to research.
    """)

    # After your main introduction markdown
    st.markdown("### Model Parameters and Estimates")

    # Get the parameters table
    parameters_df = get_parameters_table()

    # Simulation Mode Selector
    st.sidebar.title("Simulation Options")
    simulation_mode = st.sidebar.selectbox(
        "Select Simulation Mode",
        ("Multiple Simulations", "Single Simulation")
    )

    if simulation_mode == "Multiple Simulations":
        multiple_simulations.run()
    elif simulation_mode == "Single Simulation":
        single_simulation.run()
