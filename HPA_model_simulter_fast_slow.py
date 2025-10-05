import streamlit as st
import numpy as np
from scipy.optimize import fsolve
import plotly.graph_objects as go

# constants from the notebook
KA = 10**6 # Adrenal carrying capacity - very big (no carrying capacity) at default
KP = 10**6 # Pituitary carrying capacity - very big (no carrying capacity) at default

# Drift function for the HPA axis hormones
def hpa_hormones_fast(x, t, a1, b1, a2, b2, a3, b3, bP, bA, aP, aA, k, u, kgr):
    x1, x2, x3, x3b, P, A = x
    # Avoid division-by-zero
    x3 = max(x3, 1e-6)
    x3b = max(x3b, 1e-6)
    mrb = 1.0 / x3b
    gr = 1.0 / (1 + (x3 / kgr) ** 3)
    grb = 1.0 / (1 + (x3b / kgr) ** 3)
    dx1 = b1 * grb * mrb * u - a1 * x1
    dx2 = b2 * x1 * P * gr - a2 * x2
    dx3 = b3 * x2 * A - a3 * x3
    dx3b = k * (x3 - x3b) - a3 * x3b
    return np.array([dx1, dx2, dx3, dx3b])

# handle glands on the "slow axis"
# see Milo 2025 paper and supplementary material
def glands_drift_slow(x, t, a1, b1, a2, b2, a3, b3, bP, bA, aP, aA, k, u, kgr):
    x1, x2, x3, x3b, P, A = x
    # adding carrying capacities from the notebook code but otherwise these match the paper
    dP = P * (bP * x1 * (1 - P/KP) - aP)    # aP varies from the notebook code but matches the paper
    dA = A * (bA * x2 * (1 - A/KA) - aA)    # aA varies from the notebook code but matches the paper
    return np.array([dP, dA])

def hpa_system(x, t, a1, b1, a2, b2, a3, b3, bP, bA, aP, aA, k, u, kgr):
    fast = hpa_hormones_fast(x, t, a1, b1, a2, b2, a3, b3, bP, bA, aP, aA, k, u, kgr)
    slow = glands_drift_slow(x, t, a1, b1, a2, b2, a3, b3, bP, bA, aP, aA, k, u, kgr)
    return np.concatenate([fast, slow]) # bundle the fast and slow axes into a 6-dimensional array


# Euler–Maruyama SDE solver for systems
def sde_solver_system(
    drift, x0, t, sigma, params, amplitude, period, noise, stress_params=None
):
    n = len(t)
    d = len(x0)
    x = np.zeros((n, d))
    x[0] = x0
    dt = t[1] - t[0]

    # Unpack stress parameters
    if stress_params:
        stress_amplitude, stress_start, stress_duration = stress_params
    else:
        stress_amplitude, stress_start, stress_duration = 0, 0, 0

    for i in range(1, n):
        sin_wave = amplitude * np.sin(2 * np.pi * t[i - 1] / period)

        # Calculate current u based on stress parameters
        current_u = params[11]  # Original u value

        # Apply stress if within the stress period
        if stress_params and stress_start <= t[i - 1] < (
            stress_start + stress_duration
        ):
            current_u = stress_amplitude

        # Create parameters with updated u
        current_params = list(params)
        current_params[11] = current_u

        dw = noise[
            i - 1
        ]  # updated from a prior version that generated noise inside the function, we are now passing in an array and consuming per-step
        x[i] = x[i - 1] + drift(x[i - 1], t[i - 1], *current_params) * dt
        x[i][0] += sigma * dw + sin_wave  # Apply noise and sin wave to x1
    return x

# deal with streamlit entropy session states
if "entropy" not in st.session_state:
    st.session_state.entropy = int(np.random.SeedSequence().entropy)
if "entropy_input" not in st.session_state:
    st.session_state.entropy_input = "" # the input must be a textinput due to streamlit numberinput max value

# function to generate random noise with optional seed
def generate_noise(t: np.ndarray, entropy: int = None) -> np.ndarray:
    # use the generator pattern from numpy
    dt = t[1] - t[0]
    sq = (
        np.random.SeedSequence(entropy)
        if entropy is not None
        else np.random.SeedSequence()
    )
    rng = np.random.default_rng(sq)
    return rng.normal(scale=np.sqrt(dt), size=len(t))  # original noise 


def add_entropy_controls():
    # entropy: random on first run, persisted across reruns, user-adjustable
    # see notes at numpy.random.SeedSequence() documentation regarding best practices

    def _handle_new_entropy_input():  # private callback function for the entropy label
        # try to convert the input to an integer. must use try/except to handle bad input
        try:
            st.session_state.entropy = int(st.session_state.entropy_input)
        except (ValueError, TypeError):
            pass  # keep previous entropy value on bad input

    if st.sidebar.button("Randomize entropy"):
        st.session_state.entropy = int(np.random.SeedSequence().entropy)
        # update the input since we've hit the button
        st.session_state.entropy_input = ""

    st.sidebar.caption(f"Using entropy: {st.session_state.entropy}")

    st.sidebar.text_input(
        "Entropy manual override",
        help="please use only numbers",
        key="entropy_input",
        on_change=_handle_new_entropy_input,
    )

    return st.session_state.entropy

st.title("HPA Axis Fast-Slow Simulation")
st.write(
    "This app simulates the HPA axis using a system of stochastic differential equations (SDE)."
)
st.write("**Fast layer (hormones):**")
st.latex(
    r"""
\begin{aligned}
\frac{dx_1}{dt} &= b_1 \frac{1}{1 + \left(\frac{x_{3b}}{k_{gr}}\right)^3} \frac{1}{x_{3b}} u - a_1 x_1 \\
\frac{dx_2}{dt} &= b_2 x_1 P \frac{1}{1 + \left(\frac{x_3}{k_{gr}}\right)^3} - a_2 x_2 \\
\frac{dx_3}{dt} &= b_3 x_2 A - a_3 x_3 \\
\frac{dx_{3b}}{dt} &= k (x_3-x_{3b}) - a_3 x_{3b}
\end{aligned}
"""
)
st.write("**Slow layer (glands):**")
st.latex(
    r"""
\begin{aligned}
\frac{dP}{dt} &= P \left( b_P x_1 \left(1 - \frac{P}{K_P}\right) - a_P \right) \\
\frac{dA}{dt} &= A \left( b_A x_2 \left(1 - \frac{A}{K_A}\right) - a_A \right)
\end{aligned})
"""
)

# new timescale knob for experimentation
timescale_ratio = st.sidebar.slider(
    "Gland adaptation speed",
    min_value=1,
    max_value=100000,
    value=1000,
    step=1,
    help="1000 = paper default. >1000 = faster glands (more synchronized). <1000 = slower glands (more separated)."
)

# Parameters for the hormones
st.sidebar.title("Hormone Layer Parameters")
a1 = st.sidebar.number_input("a1", min_value=0.0, max_value=2.0, value=0.17, step=1e-4)
b1 = st.sidebar.number_input("b1", min_value=0.0, max_value=2.0, value=0.255, step=1e-4)
a2 = st.sidebar.number_input("a2", min_value=0.0, max_value=2.0, value=0.07, step=1e-4)
b2 = st.sidebar.number_input("b2", min_value=0.0, max_value=2.0, value=0.14, step=1e-4)
a3 = st.sidebar.number_input(
    "a3", min_value=0.0, max_value=2.0, value=0.0086, step=1e-4
)
b3 = st.sidebar.number_input("b3", min_value=0.0, max_value=2.0, value=0.086, step=1e-4)
k = st.sidebar.number_input("k", min_value=0.0, max_value=2.0, value=0.05, step=1e-4)
u = st.sidebar.number_input("u", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
kgr = st.sidebar.number_input("kgr", min_value=0.1, max_value=10.0, value=5.0, step=0.1)

st.sidebar.title("Gland Layer Parameters")
bP_per_day = st.sidebar.number_input(
    "bP (per day)", 
    value=np.log(2)/20,  # Paper default
    format="%.4f",
    help="From Milo et al. 2025"
)
bA_per_day = st.sidebar.number_input(
    "bA (per day)", 
    value=np.log(2)/30,  # Paper default
    format="%.4f",
    help="From Milo et al. 2025"
)
aP_per_day = st.sidebar.number_input(
    "aP (per day)", 
    value=np.log(2)/20,  # Paper default
    format="%.4f",
    help="From Milo et al. 2025"
)
aA_per_day = st.sidebar.number_input(
    "aA (per day)", 
    value=np.log(2)/30,  # Paper default
    format="%.4f",
    help="From Milo et al. 2025"
)
# scale the params from per day to per minute
bP = (bP_per_day / 1440) * (timescale_ratio / 1000)
bA = (bA_per_day / 1440) * (timescale_ratio / 1000)
aP = (aP_per_day / 1440) * (timescale_ratio / 1000)
aA = (aA_per_day / 1440) * (timescale_ratio / 1000)

amplitude = st.sidebar.slider(
    "Amplitude of sin wave", min_value=0.0, max_value=1.0, value=0.3, step=0.01
)
period_in_hours = st.sidebar.slider(
    "Period of sin wave (hours)", min_value=1, max_value=24, value=24, step=1
)
period = period_in_hours * 60  # Convert hours to minutes
sigma = st.sidebar.slider(
    "Noise Level (sigma)", min_value=0.0, max_value=1.0, value=0.2, step=0.01
)
T_in_hours = st.sidebar.slider(
    "Simulation Time (hours)", min_value=1, max_value=24*60, value=720, step=1
)
n_points = st.sidebar.slider(
    "Time Steps", min_value=100, max_value=10000, value=5000, step=50
)

# Simulation time
T = T_in_hours * 60
t = np.linspace(0, T, n_points)

st.sidebar.title("Noise Controls")
entropy =add_entropy_controls()

# now we should be able to generate the noise under a controlled entropy condition
noise = generate_noise(t, entropy)  # returns array size of t

# Add stress simulation parameters
st.sidebar.title("Stress Parameters")
enable_stress = st.sidebar.checkbox("Enable stress simulation", value=True)
stress_amplitude = st.sidebar.slider(
    "Stress level (u)",
    min_value=1.0,
    max_value=10.0,
    value=6.0,
    step=0.1,
    help="Value of u during stress period",
)
stress_start_hours = st.sidebar.slider(
    "Stress start time (hours)",
    min_value=0.0,
    max_value=float(T_in_hours - 1),
    value=24.0,
    step=0.5,
)
stress_duration_hours = st.sidebar.slider(
    "Stress duration (hours)",
    min_value=0.1,
    max_value=float(T_in_hours - stress_start_hours),
    value=24.0,
    step=0.1,
)

# Convert hours to minutes for simulation
stress_start = stress_start_hours * 60
stress_duration = stress_duration_hours * 60

# Pack parameters
#         0   1   2   3   4   5   6   7   8   9   10 11 12
params = (a1, b1, a2, b2, a3, b3, bP, bA, aP, aA, k, u, kgr)


# Compute steady-state initial conditions
def f_to_solve(x):
    return hpa_system(x, 0, *params)


guess = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
steady_state = fsolve(f_to_solve, guess)
x0 = steady_state

# Create stress parameters if enabled
stress_params = None
if enable_stress:
    stress_params = (stress_amplitude, stress_start, stress_duration)

# Simulate the system
sol = sde_solver_system(
    hpa_system, x0, t, sigma, params, amplitude, period, noise, stress_params
)
if st.checkbox("Normalise hormone concentrations"): # do not normalize glands!
    sol_normalized = sol.copy()
    sol_normalized[:, 0:4] = sol[:, 0:4] / np.max(sol[:, 0:4], axis=0)  # only hormones
    sol = sol_normalized

# Plotting for the hormone layer
t = t / 60  # Convert time to hours
fig = go.Figure()
fig.add_trace(go.Scatter(x=t, y=sol[:, 0], mode="lines", name="x1"))
fig.add_trace(go.Scatter(x=t, y=sol[:, 1], mode="lines", name="x2"))
fig.add_trace(go.Scatter(x=t, y=sol[:, 2], mode="lines", name="x3"))
fig.add_trace(go.Scatter(x=t, y=sol[:, 3], mode="lines", name="x3b"))

# Add shaded region for stress period if enabled
if enable_stress:
    fig.add_vrect(
        x0=stress_start_hours,
        x1=stress_start_hours + stress_duration_hours,
        fillcolor="rgba(255, 0, 0, 0.1)",
        opacity=0.5,
        layer="below",
        line_width=0,
        annotation_text="Stress Period",
        annotation_position="top left",
    )

fig.update_layout(
    title="Hormone Layer Simulation",
    xaxis_title="Time (hours)",
    yaxis_title="Concentrations",
    template="plotly_white",
)

# Add grid lines to the plot
fig.update_xaxes(showgrid=True)
fig.update_yaxes(showgrid=True)

st.plotly_chart(fig)

# Plotting for the gland layer
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=t, y=sol[:, 4], mode="lines", name="P"))
fig2.add_trace(go.Scatter(x=t, y=sol[:, 5], mode="lines", name="A"))

fig2.update_layout(
    title="Gland Layer Simulation",
    xaxis_title="Time (hours)",
    yaxis_title="Functional Mass",
    template="plotly_white",
)
fig2.update_xaxes(showgrid=True)
fig.update_yaxes(showgrid=True)

st.plotly_chart(fig2)

# SVG Download
filename = st.text_input("Filename", "hpa_simulation")
if "svg_data" not in st.session_state:
    st.session_state.svg_data = None
if "svg_data2" not in st.session_state:
    st.session_state.svg_data2 = None

def fig_to_html(fig):
    """Convert figure to HTML string"""
    return fig.to_html(include_plotlyjs='cdn', include_mathjax='cdn')

if st.button("Convert Plots to HTML"):
    st.session_state.svg_data = fig_to_html(fig)
    st.session_state.svg_data2 = fig_to_html(fig2)

if st.session_state.svg_data is not None:
    st.download_button(
        label="Download Hormone Layer Plot as HTML",
        data=st.session_state.svg_data.encode("utf-8"),
        file_name=f"{filename}.html",
        mime="text/html",
    )
if st.session_state.svg_data2 is not None:
    st.download_button(
        label="Download Gland Layer Plot as HTML",
        data=st.session_state.svg_data2.encode("utf-8"),
        file_name=f"{filename}_gland.html",
        mime="text/html",
    )

st.markdown(
    """

### Summary of the HPA Axis Model

#### Fast Layer (Hormone Cascade)

1. **Equation for $ \\frac{dx_1}{dt} $**:
   $$\\frac{dx_1}{dt} = b_1 \\frac{1}{1 + \\left(\\frac{x_{3b}}{k_{gr}}\\right)^3} \\frac{1}{x_{3b}} u - a_1 x_1$$
   - $ x_1 $: CRH (Corticotropin-Releasing Hormone)
   - $ b_1 $: Production rate constant for CRH
   - $ \\frac{1}{1 + \\left(\\frac{x_{3b}}{k_{gr}}\\right)^3} $: GR (Glucocorticoid Receptor) response inside the BBB (Blood-Brain Barrier)
   - $ \\frac{1}{x_{3b}} $: MR (Mineralocorticoid Receptor) response
   - $ u $: External stimulus (stressor input)
   - $ a_1 $: Degradation rate of CRH
   - $ k_{gr} $: Glucocorticoid receptor dissociation constant (EC50)

2. **Equation for $ \\frac{dx_2}{dt} $**:
   $$\\frac{dx_2}{dt} = b_2 x_1 P \\frac{1}{1 + \\left(\\frac{x_3}{k_{gr}}\\right)^3} - a_2 x_2$$
   - $ x_2 $: ACTH (Adrenocorticotropic Hormone)
   - $ P $: Pituitary corticotroph functional mass (scales ACTH production)
   - $ b_2 $: Production rate constant for ACTH
   - $ x_1 $: CRH (upstream signal)
   - $ \\frac{1}{1 + \\left(\\frac{x_3}{k_{gr}}\\right)^3} $: GR negative feedback (outside the BBB)
   - $ a_2 $: Degradation rate of ACTH

3. **Equation for $ \\frac{dx_3}{dt} $**:
   $$\\frac{dx_3}{dt} = b_3 x_2 A - a_3 x_3$$
   - $ x_3 $: Cortisol (in blood)
   - $ A $: Adrenal cortex functional mass (scales cortisol production)
   - $ b_3 $: Production rate constant for cortisol
   - $ x_2 $: ACTH (upstream signal)
   - $ a_3 $: Degradation rate of cortisol

4. **Equation for $ \\frac{dx_{3b}}{dt} $**:
   $$\\frac{dx_{3b}}{dt} = k (x_3 - x_{3b}) - a_3 x_{3b}$$
   - $ x_{3b} $: Cortisol inside the BBB (brain compartment)
   - $ k $: Transfer rate constant between blood and brain
   - $ x_3 $: Cortisol in blood
   - $ a_3 $: Degradation rate of cortisol in the BBB


### Slow Layer (Gland Adaptation)

5. **Equation for $ \\frac{dP}{dt} $**:
   $$\\frac{dP}{dt} = P \\left( b_P x_1 \\left(1 - \\frac{P}{K_P}\\right) - a_P \\right)$$
   - $ P $: Pituitary corticotroph functional mass
   - $ b_P $: Growth rate of pituitary tissue (driven by CRH)
   - $ a_P $: Removal rate of pituitary tissue
   - $ K_P $: Pituitary carrying capacity
   - $ \\left(1 - \\frac{P}{K_P}\\right) $: Logistic growth term (prevents unbounded growth)

6. **Equation for $ \\frac{dA}{dt} $**:
   $$\\frac{dA}{dt} = A \\left( b_A x_2 \\left(1 - \\frac{A}{K_A}\\right) - a_A \\right)$$
   - $ A $: Adrenal cortex functional mass
   - $ b_A $: Growth rate of adrenal tissue (driven by ACTH)
   - $ a_A $: Removal rate of adrenal tissue
   - $ K_A $: Adrenal carrying capacity
   - $ \\left(1 - \\frac{A}{K_A}\\right) $: Logistic growth term

#### Timescale Separation

- **Fast layer:** Hormones equilibrate on the scale of minutes to hours
- **Slow layer:** Glands adapt on the scale of weeks to months (half-life ~20-30 days)
- This separation enables **dynamical compensation** - glands adjust their mass to buffer shocks (see plots), explaining why, according to Milo et al. 2025, HPA-targeting drugs fail in chronic stress conditions

**Reference:** Milo et al. (2025) "Hormone circuit explains why most HPA drugs fail for mood disorders and predicts the few that work" *Molecular Systems Biology* 21(3):254-273
    """
)


