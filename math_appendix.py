import streamlit as st

def display():
    st.markdown(r""" ## Math Appendix""")
    
    st.markdown("""##### Semi-Endogenous Growth Environment""")
    st.markdown(r"""
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

    The ratio we are ultimately interested in $g_{S,\text{accelerated}}/g_{S,\text{base}}$ -- an given this environment the choices of $\bar{R}$ and $C_0$ can have an effect on this ratio -- these are also the variables which seem to be difficult to calibrate meanigfully. Forutunately, changing these varibales has an (almost) undetectable effect on the ratio we are studying. 
    """)

    st.markdown("""##### Correlated Sampling""")
    st.markdown(r"""When running mutliple simulations I allow for users to select for $\beta_0$ and $f$ to be positively correlated. Specifically, I use ther relationship"
    """)
    st.latex(r"""
    \log(\beta_0) = \text{Intercept} + \text{Slope} \times \log(f) + \epsilon \quad \epsilon \sim N(0,\sigma^2)
    """)
    st.markdown(r"""
    Ignoring $\epsilon$ for a second, to calibrate the slope we just need to ensure that choices of $f$ are scaled so that each choice in the range of possible $f$ can 'pick out' a possible $\beta_0$ value. To do this, set 
    """)
    st.latex(r"""
    \text{Slope} = \frac{\log(\beta_{0,max})-\log(\beta_{0,min})}{\log(f_{max}) -\log(f_{max})}
    """)
    st.markdown(r"""Then we just have to solve for intercept that ensures (still, ignoring $\epsilon$) that picking $f_{min}$ will result in $\beta_{0,min}$ being chosen (and likewise for max values). This yields
    """)
    st.latex(r"""
    \text{Intercept} = \log(\beta_{0,min}) - \text{Slope} \times \log(f_{min})
    """)
    st.markdown(r""" In summary, under this specification, when $\epsilon = 0$ there is a one-to-one mapping from $f$ to $\beta_0$ so that a choice of $f$, situatied at some point in the distribution of possible choices of $f$ picks out a roughly comparable $\beta_0$ from the possible space of $\beta_0$. Adding $\epsilon$ then induces some variation in this process that increases with choice of $\sigma$. Note however, that if this noise results in a choice of $\beta_0$ that is outside of the bounds I allow for $\beta_0$ I just set the choice for $\beta_0$ to be the nearest acceptable value (this will result in some bunching at the edges of allowed values of $\beta_0$). I'd encourage curious uses to run the 'multiple simulations' setting and display the empirical distributions to look at the observed correlation between $f$ and $\beta_0$.
    """)

    # Include a link to go back to the main page
    st.markdown("[Go back](?page=main)")
