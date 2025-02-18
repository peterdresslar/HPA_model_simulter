import streamlit as st
import numpy as np
from scipy.optimize import fsolve
import io
from PIL import Image


# Drift function for the HPA axis
def hpa_drift(x, t, a1, b1, a2, b2, a3, b3, k, u, kgr):
    # x = [x1, x2, x3, x3b]
    x1, x2, x3, x3b = x
    # Avoid division-by-zero
    if x3 == 0:
        x3 = 1e-6
    if x3b == 0:
        x3b = 1e-6
    mrb = 1.0 / x3b
    gr = 1.0 / (1 + (x3 / kgr) ** 3)
    grb = 1.0 / (1 + (x3b / kgr) ** 3)
    dx1 = b1 * grb * mrb * u - a1 * x1
    dx2 = b2 * x1 * gr - a2 * x2
    dx3 = b3 * x2 - a3 * x3
    dx3b = k * (x3 - x3b) - a3 * x3b
    return np.array([dx1, dx2, dx3, dx3b])


# Euler–Maruyama SDE solver for systems
def sde_solver_system(drift, x0, t, sigma, params):
    n = len(t)
    d = len(x0)
    x = np.zeros((n, d))
    x[0] = x0
    dt = t[1] - t[0]
    period = 24 * 60  # 24 hours in minutes
    amplitude = st.sidebar.number_input("Amplitude of sin wave", min_value=0.0, max_value=5.0, value=1.0, step=0.1)

    for i in range(1, n):
        dw = np.random.normal(scale=np.sqrt(dt))  # Single noise term for x1
        # Add a sinusoidal term to the drift of x1
        sin_wave = amplitude * np.sin(2 * np.pi * t[i-1] / period)
        x[i] = x[i - 1] + drift(x[i - 1], t[i - 1], *params) * dt
        x[i][0] += sigma * dw + sin_wave  # Apply noise and sin wave to x1
    return x


st.title("HPA Axis Simulation using Stochastic Differential Equations (SDE)")
st.write("This app simulates the HPA axis using a system of stochastic differential equations (SDE).")
st.write("The equations of the HPA axis model are:")
st.latex(
    r"""
\begin{aligned}
\frac{dx_1}{dt} &= b_1 \frac{1}{1 + \left(\frac{x_{3b}}{k_{gr}}\right)^3} \frac{1}{x_{3b}} u - a_1 x_1 \\
\frac{dx_2}{dt} &= b_2 x_1 \frac{1}{1 + \left(\frac{x_3}{k_{gr}}\right)^3} - a_2 x_2 \\
\frac{dx_3}{dt} &= b_3 x_2 - a_3 x_3 \\
\frac{dx_{3b}}{dt} &= k (x_3-x_{3b}) - a_3 x_{3b}
\end{aligned}
"""
)
# Sidebar controls for parameters
st.sidebar.title("HPA Axis Simulation Parameters")

# Parameters for the drift function
a1 = st.sidebar.number_input("a1", min_value=0.0, max_value=2.0, value=0.17, step=1e-4)
b1 = st.sidebar.number_input("b1", min_value=0.0, max_value=2.0, value=0.255, step=1e-4)
a2 = st.sidebar.number_input("a2", min_value=0.0, max_value=2.0, value=0.07, step=1e-4)
b2 = st.sidebar.number_input("b2", min_value=0.0, max_value=2.0, value=0.14, step=1e-4)
a3 = st.sidebar.number_input("a3", min_value=0.0, max_value=2.0, value=0.0086, step=1e-4)
b3 = st.sidebar.number_input("b3", min_value=0.0, max_value=2.0, value=0.086, step=1e-4)
k = st.sidebar.number_input("k", min_value=0.0, max_value=2.0, value=0.05, step=1e-4)
u = st.sidebar.number_input("u", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
kgr = st.sidebar.number_input("kgr", min_value=0.1, max_value=5.0, value=5.0, step=0.1)


# Pack parameters for drift function
params = (a1, b1, a2, b2, a3, b3, k, u, kgr)


# Compute steady-state initial conditions by solving hpa_drift(x)=0
def f_to_solve(x):
    return hpa_drift(x, 0, *params)


# Initial guess for fsolve
guess = [1.0, 1.0, 1.0, 1.0]
steady_state = fsolve(f_to_solve, guess)
st.sidebar.write("Steady state initial conditions:")
st.sidebar.write(f"x1 = {steady_state[0]:.4f}, x2 = {steady_state[1]:.4f}, x3 = {steady_state[2]:.4f}, x3b = {steady_state[3]:.4f}")
x0 = steady_state  # use the computed steady state

# Simulation time and discretization
T = st.sidebar.slider("T (minutes)", min_value=60.0, max_value=2400.0, value=600.0, step=60.0)
n_points = st.sidebar.slider("Time Steps", min_value=100, max_value=1000, value=400, step=50)
t = np.linspace(0, T, n_points)

# Diffusion coefficient (noise level)
sigma = st.sidebar.slider("Noise Level (sigma)", min_value=0.0, max_value=1.0, value=0.5, step=0.001)

# Simulate the system using the SDE solver
sol = sde_solver_system(hpa_drift, x0, t, sigma, params)

# Create a matplotlib figure for plotting
import plotly.graph_objects as go

# Create traces for each variable
trace1 = go.Scatter(x=t, y=sol[:, 0], mode="lines", name="x1", line=dict(color="blue"))
trace2 = go.Scatter(x=t, y=sol[:, 1], mode="lines", name="x2", line=dict(color="green"))
trace3 = go.Scatter(x=t, y=sol[:, 2], mode="lines", name="x3", line=dict(color="red"))
trace4 = go.Scatter(x=t, y=sol[:, 3], mode="lines", name="x3b", line=dict(color="purple"))

# Define layout
layout = go.Layout(
    title="HPA Axis Simulation using SDE (Euler–Maruyama)",
    xaxis=dict(title="Time (minutes)", tickmode="array", tickvals=np.arange(0, T + 1, 60)),
    yaxis=dict(title="Concentrations"),
    legend=dict(x=0, y=1, traceorder="normal"),
    template="plotly_white",
)

# Create figure
layout.update(
    xaxis=dict(
        showgrid=True,
        gridcolor="lightgray",
        gridwidth=0.5,
        title="Time (minutes)",
        tickmode="array",
        tickvals=np.arange(0, T + 1, 60),
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor="lightgray",
        gridwidth=0.5,
        title="Concentrations",
    ),
)
fig = go.Figure(data=[trace1, trace2, trace3, trace4], layout=layout)

# Show figure in Streamlit
st.plotly_chart(fig)

# Provide a text input for the filename and a download button
filename = st.text_input("Filename", "hpa_simulation")

# Function to convert Plotly figure to SVG
def fig_to_svg(fig):
    img_bytes = fig.to_image(format="svg")
    return img_bytes.decode("utf-8")

# Initialize session state variable if not already present
if 'svg_data' not in st.session_state:
    st.session_state.svg_data = None

# Button to trigger SVG conversion
if st.button("Convert Plot to SVG"):
    st.session_state.svg_data = fig_to_svg(fig)

# If the SVG has been generated, provide a download button without reloading code
if st.session_state.svg_data is not None:
    st.download_button(
        label="Download Plot as SVG",
        data=st.session_state.svg_data.encode("utf-8"),
        file_name=f"{filename}.svg",
        mime="image/svg+xml",
    )