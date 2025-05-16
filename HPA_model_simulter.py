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
def sde_solver_system(drift, x0, t, sigma, params, amplitude, period, stress_params=None):
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
        current_u = params[7]  # Original u value
        
        # Apply stress if within the stress period
        if stress_params and stress_start <= t[i - 1] < (stress_start + stress_duration):
            current_u = stress_amplitude
        
        # Create parameters with updated u
        current_params = list(params)
        current_params[7] = current_u
        
        dw = np.random.normal(scale=np.sqrt(dt))  # Single noise term for x1
        x[i] = x[i - 1] + drift(x[i - 1], t[i - 1], *current_params) * dt
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
st.sidebar.title("Simulation Parameters")

b1= 0.17
b2= 0.07
b3= 0.0086
a1= 0.17
a2= 0.07
a3= 0.0086

# Parameters for the drift function
a1 = st.sidebar.number_input("a1", min_value=0.0, max_value=2.0, value=a1, step=1e-4,format="%.4f")
b1 = st.sidebar.number_input("b1", min_value=0.0, max_value=2.0, value=b1, step=1e-4,format="%.4f")
a2 = st.sidebar.number_input("a2", min_value=0.0, max_value=2.0, value=a2, step=1e-4,format="%.4f")
b2 = st.sidebar.number_input("b2", min_value=0.0, max_value=2.0, value=b2, step=1e-4,format="%.4f")
a3 = st.sidebar.number_input("a3", min_value=0.0, max_value=2.0, value=a3, step=1e-4,format="%.4f")
b3 = st.sidebar.number_input("b3", min_value=0.0, max_value=2.0, value=b3, step=1e-4,format="%.4f")
# k = st.sidebar.number_input("k", min_value=0.0, max_value=2.0, value=0.05, step=1e-3)
k = st.sidebar.slider("k", min_value=0.0, max_value=2.0, value=0.05, step=1e-3)
u = st.sidebar.number_input("u", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
kgr = st.sidebar.number_input("kgr", min_value=0.1, max_value=10.0, value=5.0, step=0.1)
amplitude = st.sidebar.slider("Amplitude of sin wave", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
period_in_hours = st.sidebar.slider("Period of sin wave (hours)", min_value=1, max_value=24, value=24, step=1)
period = period_in_hours * 60  # Convert hours to minutes
sigma = st.sidebar.slider("Noise Level (sigma)", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
T_in_hours = st.sidebar.slider("Simulation Time (hours)", min_value=1, max_value=48, value=24, step=1)
n_points = st.sidebar.slider("Time Steps", min_value=100, max_value=10000, value=400, step=50)

# Add stress simulation parameters
st.sidebar.title("Stress Parameters")
enable_stress = st.sidebar.checkbox("Enable stress simulation", value=False)
stress_amplitude = st.sidebar.slider("Stress level (u)", min_value=1.0, max_value=10.0, value=3.0, step=0.1, 
                                  help="Value of u during stress period")
stress_start_hours = st.sidebar.slider("Stress start time (hours)", min_value=0.0, 
                                   max_value=float(T_in_hours-1), value=5.0, step=0.5)
stress_duration_hours = st.sidebar.slider("Stress duration (hours)", min_value=0.1, 
                                     max_value=float(T_in_hours-stress_start_hours), value=2.0, step=0.1)

# Convert hours to minutes for simulation
stress_start = stress_start_hours * 60
stress_duration = stress_duration_hours * 60

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

# Create stress parameters if enabled
stress_params = None
if enable_stress:
    stress_params = (stress_amplitude, stress_start, stress_duration)

# Simulate the system
sol = sde_solver_system(hpa_drift, x0, t, sigma, params, amplitude, period, stress_params)
if st.checkbox("Normalise the concentrations"):
    sol = sol / np.max(sol, axis=0)
    
# Plotting



# b1: 0.17 ,
# b2: 0.07 ,
# b3: 0.0086, 
# a1: 0.17 ,
# a2: 0.07 ,
# a3: 0.0086 ,


t = t   # Convert time to hours
fig = go.Figure()
# remove the sin wave from the plot
if st.checkbox("Remove sin wave from the plot"):
    sol[:, 1] -= np.array([amplitude * np.sin(2 * np.pi * t[i] / period) for i in range(len(t))])
    sol[:, 2] -= np.array([amplitude * np.sin(2 * np.pi * t[i] / period) for i in range(len(t))])
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
    title="HPA Axis Simulation",
    xaxis_title="Time (hours)",
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
import numpy as np
from scipy.signal import find_peaks
import plotly.graph_objects as go

# Compute FFT for each signal
fft_x1  = np.fft.fft(sol[:, 0])
fft_x2  = np.fft.fft(sol[:, 1])
fft_x3  = np.fft.fft(sol[:, 2])
fft_x3b = np.fft.fft(sol[:, 3])

# Calculate frequency bins using the sampling interval from t (in minutes)
freq = np.fft.fftfreq(len(t), d=t[1] - t[0])  # frequency in cycles/minute

# Convert frequency to cycles per week:
freq_hours = freq * 60   # 60 minutes, 24 hours, 7 days = 10080

# Remove the 0 frequency (DC component) by filtering positive frequencies
#remove the day frequency
# positive = freq > 0.05
# mask = (freq_week > ) & (freq_week < 0.5)  # mask for frequencies between 0.05 and 0.5 cycles/week
mask= freq_hours>0

# Compute amplitude and phase for each signal
amp_x1  = np.abs(fft_x1)
amp_x2  = np.abs(fft_x2)
amp_x3  = np.abs(fft_x3)
amp_x3b = np.abs(fft_x3b)

phase_x1  = np.angle(fft_x1)
phase_x2  = np.angle(fft_x2)
phase_x3  = np.angle(fft_x3)
phase_x3b = np.angle(fft_x3b)

# Plot amplitude spectrum using frequencies in cycles per week
fig_amp = go.Figure()
fig_amp.add_trace(go.Scatter(x=freq_hours[mask], y=amp_x1[mask],
                             mode='lines', name='x1 amplitude'))
fig_amp.add_trace(go.Scatter(x=freq_hours[mask], y=amp_x2[mask],
                             mode='lines', name='x2 amplitude'))
fig_amp.add_trace(go.Scatter(x=freq_hours[mask], y=amp_x3[mask],
                             mode='lines', name='x3 amplitude'))
fig_amp.add_trace(go.Scatter(x=freq_hours[mask], y=amp_x3b[mask],
                             mode='lines', name='x3b amplitude'))
fig_amp.update_layout(title="Amplitude Spectrum",
                      xaxis_title="Frequency (cycles/hours)",
                      yaxis_title="Amplitude")
st.plotly_chart(fig_amp)

# Plot phase spectrum using frequencies in cycles per hoursfreq_hours
fig_phase = go.Figure()
fig_phase.add_trace(go.Scatter(x=freq_hours[mask], y=phase_x1[mask],
                               mode='lines', name='x1 phase'))
fig_phase.add_trace(go.Scatter(x=freq_hours[mask], y=phase_x2[mask],
                               mode='lines', name='x2 phase'))
fig_phase.add_trace(go.Scatter(x=freq_hours[mask], y=phase_x3[mask],
                               mode='lines', name='x3 phase'))
fig_phase.add_trace(go.Scatter(x=freq_hours[mask], y=phase_x3b[mask],
                               mode='lines', name='x3b phase'))
fig_phase.update_layout(title="Phase Spectrum",
                        xaxis_title="Frequency (cycles/week)",
                        yaxis_title="Phase (radians)")
st.plotly_chart(fig_phase)

# print the phase diff betwee x1 and x2, x2 and x3, x3 and x3b
# Calculate time delay using the phase difference and the corresponding frequency (f0)
f0 = freq[positive][0]  # fundamental frequency corresponding to the first positive frequency
# Find peaks for x1 and x2 signals
peaks_x1, _ = find_peaks(sol[:, 0])
peaks_x2, _ = find_peaks(sol[:, 1])
if len(peaks_x1) > 0 and len(peaks_x2) > 0:
    # Calculate the delay as the difference between the first detected peaks
    # delay_x1_x2 = t[peaks_x2[0]] - t[peaks_x1[0]]
# else:
    peaks_x1, _ = find_peaks(sol[:, 0])
    peaks_x2, _ = find_peaks(sol[:, 1])
    peaks_x3, _ = find_peaks(sol[:, 2])
    peaks_x3b, _ = find_peaks(sol[:, 3])
    if len(peaks_x1) > 0 and len(peaks_x2) > 0:
        delay_x1_x2_peaks = t[peaks_x2[0]] - t[peaks_x1[0]]
    else:
        delay_x1_x2_peaks = np.nan
        
    if len(peaks_x2) > 0 and len(peaks_x3) > 0:
        delay_x2_x3_peaks = t[peaks_x3[0]] - t[peaks_x2[0]]
    else:
        delay_x2_x3_peaks = np.nan

    if len(peaks_x3) > 0 and len(peaks_x3b) > 0:
        delay_x3_x3b_peaks = t[peaks_x3b[0]] - t[peaks_x3[0]]
    else:
        delay_x3_x3b_peaks = np.nan

    if len(peaks_x3b) > 0 and len(peaks_x1) > 0:
        delay_x3b_x1_peaks = t[peaks_x3b[0]] - t[peaks_x1[0]] 
    else:
        delay_x3b_x1_peaks = np.nan
    st.write(f"Peak-based time delay between x1 and x2: {delay_x1_x2_peaks:.2f} time units")
    st.write(f"Peak-based time delay between x2 and x3: {delay_x2_x3_peaks:.2f} time units")
    st.write(f"Peak-based time delay between x3 and x3b: {delay_x3_x3b_peaks:.2f} time units")
    st.write(f"Peak-based time delay between x3b and x1: {delay_x3b_x1_peaks:.2f} time units")
# delay_x2_x3 = (phase_x3[positive][0] - phase_x2[positive][0]) / (2 * np.pi * f0)
# delay_x3_x3b = (phase_x3b[positive][0] - phase_x3[positive][0]) / (2 * np.pi * f0)
# delay_x3b_x1 = (phase_x1[positive][0] - phase_x3b[positive][0]) / (2 * np.pi * f0)


# st.write(f"Time delay between x1 and x2: {delay_x1_x2:.2f} time units")
# st.write(f"Time delay between x2 and x3: {delay_x2_x3:.2f} time units")
# st.write(f"Time delay between x3 and x3b: {delay_x3_x3b:.2f} time units")
# st.write(f"Time delay between x3b and x1: {delay_x3b_x1:.2f} time units")

st.markdown(
    """

### Summary of the HPA Axis Model

1. **Equation for $ \\frac{dx_1}{dt} $**:
   $$\\frac{dx_1}{dt} = b_1 \\frac{1}{1 + \\left(\\frac{x_{3b}}{k_{gr}}\\right)^3} \\frac{1}{x_{3b}} u - a_1 x_1$$
   - $ x_1 $: CRH (Corticotropin-Releasing Hormone)
   - $ b_1 $: Production rate constant for CRH
   - $ \\frac{1}{1 + \\left(\\frac{x_{3b}}{k_{gr}}\\right)^3} $: GR (Glucocorticoid Receptor) response inside the BBB (Blood-Brain Barrier)
   - $ \\frac{1}{x_{3b}} $: MR (Mineralocorticoid Receptor) response
   - $ u $: External stimulus
   - $ a_1 $: Degradation rate of CRH
   - $ k_{gr} $: Concentration of cortisol at which the GR response is at half of its maximum effectiveness (EC50).
   - 

2. **Equation for $ \\frac{dx_2}{dt} $**:
   $$\\frac{dx_2}{dt} = b_2 x_1 \\frac{1}{1 + \\left(\\frac{x_3}{k_{gr}}\\right)^3} - a_2 x_2$$
   - $ x_2 $: Another form of CRH
   - $ b_2 $: Production rate constant for this form of CRH
   - $ x_1 $: Precursor CRH
   - $ \\frac{1}{1 + \\left(\\frac{x_3}{k_{gr}}\\right)^3} $: GR response (outside the BBB)
   - $ a_2 $: Degradation rate of this form of CRH

3. **Equation for $ \\frac{dx_3}{dt} $**:
   $$\\frac{dx_3}{dt} = b_3 x_2 - a_3 x_3$$
   - $ x_3 $: Cortisol
   - $ b_3 $: Production rate constant for cortisol
   - $ x_2 $: Precursor CRH
   - $ a_3 $: Degradation rate of cortisol

4. **Equation for $ \\frac{dx_{3b}}{dt} $**:
   $$\\frac{dx_{3b}}{dt} = k (x_3 - x_{3b}) - a_3 x_{3b}$$
   - $ x_{3b} $: Cortisol in the BBB
   - $ k $: Transfer rate constant between blood and brain
   - $ x_3 $: Cortisol
   - $ a_3 x_{3b} $: Degradation rate of cortisol in the BBB
    """
)
