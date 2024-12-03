import streamlit as st

def display():
    st.markdown(r"""
    ## Math Appendix

    The numerical experiment we consider begins when GPT-6 is released, and assume the law of motion for software: 
    """)
    st.latex(r"""
    \dot{S}_t = \bar{R}^{\lambda \alpha} {C}_t^{\lambda(1-\alpha)} S_t^{1-\beta(S_t)}
    """)
    st.markdown(r"""
    where $S$ is the software level, $\bar{R}$ is stock of human AI researchers at the time GPT-6 is released (we assume this is a fixed quantity over time), $C$ is the compute available for research on AI, $\lambda$ is the degree of parallelizability in research, and $\alpha \in (0,1)$ dictates the contributions of researchers vs. compute to progress in AI. Importantly, the degree of diminishing returns, $\beta$ is a function of $S$. We assume the functional form for $\beta$ is such that every time the software level closes half the gap between its level and some software ceiling, $S_{\text{ceiling}}$, $\beta$ doubles. This functional form is given by   
    """)
    st.latex(r"""
    \beta(S_t) = \beta_0 \left(1 - \frac{\frac{S}{\bar{S}} - 1}{\frac{S_{\text{ceiling}}}{\bar{S}} - 1}\right)^{-1}
    """)
    st.markdown(r"""
    where $\beta_0$ is the starting level of diminishing returns to research. 
        
    In the base case (where AI isn't deployed to AI research after GPT-6) we assume that software level is doubling every three months, this corresponds to an annual growth rate of 2.77. We want to calibrate the model so, under assumptions on $\bar{R}$ and $C_0$, the growth rate of $S$ matches observed growth rates. Dividing the laws of motion by software levels to get growth rates we have 
    """)
    st.latex(r"""
    g_{S,0} = \bar{R}^{\lambda \alpha} {C}_0^{\lambda(1-\alpha)} S_0^{-\beta(S_0)} \implies S_0 = \left[2.77 \times \left(\bar{R}^{\lambda \alpha} {C}_0^{\lambda(1-\alpha)}\right)^{-1}\right]^{\frac{-1}{\beta_0}}
    """)
    st.markdown(r"""
    Next, we construct a case such that progress in AI would remain exponential (if it weren't for increasing diminishing returns to research); this case requires growth in compute. Specifically, given the form for the growth rate of $S$, for $g_S$ to remain constant we require
    """)
    st.latex(r"""
    g_C = \frac{\beta_0}{\lambda (1-\alpha) }g_{S,0}
    """)
    st.markdown(r"""
    Equations thus far are sufficient to simulate the path of $S$ over time. Simulating, one can see that this path looks generally exponential for some time until we get quite close to the software ceiling, in which case progress trails off. 

    Now I turn to the case where we allow deployment of AI to accelerate AI research. Specifically, I assume that research accelerates by a factor $f$ after deployment of AI to research. To fix intuition, we can think of this as an increasing of the stock of researchers by $f^{\frac{1}{\lambda \alpha}}$ at $t=0$.

    Next, we need to determine how AI progress contributes to the increased stock of AI researchers. I assume 
    """)
    st.latex(r"""
    R_t = \bar{R} + \upsilon S_t
    """)
    st.markdown(r"""
    where $\upsilon$ scales software into human researcher equivalents. To calibrate $\upsilon$, we again look to the initial conditions of the model. Namely, so that the stock of available researchers after reaching GPT-6 is $f^{\frac{1}{\lambda \alpha}} \times \bar{R}$, we require
    """)
    st.latex(r"""
    f^{\frac{1}{\lambda \alpha}} \bar{R} = \bar{R} + \upsilon S_0 \implies \upsilon = \bar{R} \times \left(f^{\frac{1}{\lambda \alpha}} - 1\right) \times \left[2.77 \times \left(\bar{R}^\alpha C_0^{1-\alpha}\right)^{-\lambda}\right]^{\frac{1}{\beta_0}}
    """)
    st.markdown(r"""
    We will assume that growth in compute follows the same path as the base case where AI is not deployed to research.

    A useful property of this environment is that the ratio $g_{S,\text{accelerated}}/g_{S,\text{base}}$ is unaffected by choices of $\bar{R}$ and $C_0$ -- this ratio is what is being studied in the multiple simulations case. [to prove]
    """)
    # Include a link to go back to the main page
    st.markdown("[Go back](?page=main)")
