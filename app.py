import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

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
