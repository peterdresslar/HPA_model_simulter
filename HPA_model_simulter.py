import streamlit as st
import numpy as np
from scipy.optimize import fsolve
import plotly.graph_objects as go


# Drift function for the HPA axis
def hpa_drift(x, t, a1, b1, a2, b2, a3, b3, k, u, kgr):
    x1, x2, x3, x3b = x
    # Avoid division-by-zero
    x3 = max(x3, 1e-6)
    x3b = max(x3b, 1e-6)
    mrb = 1.0 / x3b
    gr = 1.0 / (1 + (x3 / kgr) ** 3)
    grb = 1.0 / (1 + (x3b / kgr) ** 3)
    dx1 = b1 * grb * mrb * u - a1 * x1
    dx2 = b2 * x1 * gr - a2 * x2
    dx3 = b3 * x2 - a3 * x3
    dx3b = k * (x3 - x3b) - a3 * x3b
    return np.array([dx1, dx2, dx3, dx3b])


# Euler–Maruyama SDE solver for systems
def sde_solver_system(drift, x0, t, sigma, params, amplitude, period):
    n = len(t)
    d = len(x0)
    x = np.zeros((n, d))
    x[0] = x0
    dt = t[1] - t[0]
    start_sin_with_delay = st.slider("Start sin wave after (minutes)", min_value=0, max_value=100, value=0, step=10)
    for i in range(1, n):
        dw = np.random.normal(scale=np.sqrt(dt))  # Single noise term for x1
        # Add a sinusoidal term to the drift of x1
        if i > start_sin_with_delay:
            sin_wave = amplitude * np.sin(2 * np.pi * t[i - 1] / period)
        x[i] = x[i - 1] + drift(x[i - 1], t[i - 1], *params) * dt
        if i > start_sin_with_delay:
            x[i][0] += sigma * dw + sin_wave  # Apply noise and sin wave to x1
    return x


st.title("HPA Axis Simulation using Stochastic Differential Equations (SDE)")

# Sidebar controls for parameters
st.sidebar.title("Simulation Parameters")

# Parameters for the drift function
a1 = st.sidebar.number_input("a1", min_value=0.0, max_value=2.0, value=0.17, step=1e-4)
b1 = st.sidebar.number_input("b1", min_value=0.0, max_value=2.0, value=0.255, step=1e-4)
a2 = st.sidebar.number_input("a2", min_value=0.0, max_value=2.0, value=0.07, step=1e-4)
b2 = st.sidebar.number_input("b2", min_value=0.0, max_value=2.0, value=0.14, step=1e-4)
a3 = st.sidebar.number_input("a3", min_value=0.0, max_value=2.0, value=0.0086, step=1e-4)
b3 = st.sidebar.number_input("b3", min_value=0.0, max_value=2.0, value=0.086, step=1e-4)
k = st.sidebar.number_input("k", min_value=0.0, max_value=2.0, value=0.05, step=1e-4)
u = st.sidebar.number_input("u", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
kgr = st.sidebar.number_input("kgr", min_value=0.1, max_value=10.0, value=5.0, step=0.1)
amplitude = st.sidebar.slider("Amplitude of sin wave", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
period_in_hours = st.sidebar.slider("Period of sin wave (hours)", min_value=1, max_value=24, value=24, step=1)
period = period_in_hours * 60  # Convert hours to minutes
sigma = st.sidebar.slider("Noise Level (sigma)", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
T_in_hours = st.sidebar.slider("Simulation Time (hours)", min_value=1, max_value=48, value=24, step=1)
n_points = st.sidebar.slider("Time Steps", min_value=100, max_value=1000, value=400, step=50)


# Pack parameters
params = (a1, b1, a2, b2, a3, b3, k, u, kgr)


# Compute steady-state initial conditions
def f_to_solve(x):
    return hpa_drift(x, 0, *params)


guess = [1.0, 1.0, 1.0, 1.0]
steady_state = fsolve(f_to_solve, guess)
x0 = steady_state

# Simulation time
T = T_in_hours * 60
t = np.linspace(0, T, n_points)

# Simulate the system
sol = sde_solver_system(hpa_drift, x0, t, sigma, params, amplitude, period)
normalise = st.checkbox("Normalise the concentrations")
if normalise:
    sol = sol / np.max(sol, axis=0)
# if normalise:
# Plotting
t = t  # Convert time to hours
fig = go.Figure()
fig.add_trace(go.Scatter(x=t, y=sol[:, 0], mode="lines", name="x1"))
fig.add_trace(go.Scatter(x=t, y=sol[:, 1], mode="lines", name="x2"))
fig.add_trace(go.Scatter(x=t, y=sol[:, 2], mode="lines", name="x3"))
fig.add_trace(go.Scatter(x=t, y=sol[:, 3], mode="lines", name="x3b"))
fig.update_layout(
    title="HPA Axis Simulation",
    xaxis_title="Time (minutes)",
    yaxis_title="Concentrations",
    template="plotly_white",
)

# Add grid lines to the plot
fig.update_xaxes(showgrid=True)
fig.update_yaxes(showgrid=True)

st.plotly_chart(fig)


# SVG Download
filename = st.text_input("Filename", "hpa_simulation")
if "svg_data" not in st.session_state:
    st.session_state.svg_data = None


def fig_to_svg(fig):
    img_bytes = fig.to_image(format="svg")
    return img_bytes.decode("utf-8")


if st.button("Convert Plot to SVG"):
    st.session_state.svg_data = fig_to_svg(fig)

if st.session_state.svg_data is not None:
    st.download_button(
        label="Download Plot as SVG",
        data=st.session_state.svg_data.encode("utf-8"),
        file_name=f"{filename}.svg",
        mime="image/svg+xml",
    )
