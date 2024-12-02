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
def get_parameters_table_markdown():
    table_markdown = r'''
| Parameter            | Description                                                         | Low Estimate  | Median Estimate | High Estimate |
|----------------------|---------------------------------------------------------------------|---------------|-----------------|---------------|
| $f$                  | Initial shock to brainpower dedicated to AI research                | 2             | 8               | 32            |
| $\lambda$            | Parallelizability of research                                       | 0.2           | 0.5             | 0.8           |
| $\beta$              | Diminishing returns in discovering new ideas                        | 0.15          | 0.45            | 0.75          |
| $\alpha$             | Contribution of cognitive labor vs. compute to AI R&D               | 0.0           | 0.5             | 1.0           |
| $g$                  | Growth rate of software when GPT-6 is deployed                      | 2.0           | 2.77            | 3.0           |
| $S_{\text{ceiling}}$ | Ceiling on the level of software                                    | $1 \times 10^{7}$ | $1 \times 10^{8}$ | $1 \times 10^{9}$ |
    '''
    return table_markdown



params = st.experimental_get_query_params()
if params.get('page') == ['math_appendix']:
    # Display Math Appendix
    math_appendix.display()
else:
    # Main Page Content
    st.title('Simulation of Accelerated Growth Model')

    st.markdown(r"""
    This tool analyzes the impact of automating AI R&D. The key parameters for this tool are:

    - **$f$**: Size of initial shock to the amount of brainpower dedicated to AI research.
    - **$\lambda$**: Parallelizability of research.
    - **$\beta$**: Level of diminishing returns for discovering new ideas.
    - **$\alpha$**: Share contribution of cognitive labor vs compute to AI R&D.
    - **$g$**: Growth rate of software when GPT-6 is deployed.
    - **$S_{\text{ceiling}}$**: Ceiling on the level of software.

    This tool offers two options. You can run a bunch of simulations with uncertainty over key parameters—the output of this function will be a plot showing the fraction of simulations where the growth rate of software exceeds the observed exponential rate by some amount over some number of years. Alternatively, you can run a single simulation under specific parameter values to illustrate the path of AI progress. Under this second option, you will also see the change in the level of diminishing research productivity over time and the growth rates compared to (a) constant exponential progress and (b) the projected path of software without deploying AI to research.

    As above, we allow the stock of compute to grow so that, at the time of GPT-6, growth of the software level looks exponential. Now, we also allow AI to be deployed to research.
     """)

    st.markdown("### Model Parameters and Estimates")

    # Get the parameters table in Markdown format
    parameters_table_md = get_parameters_table_markdown()

    # Display the table
    st.markdown(parameters_table_md)


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
