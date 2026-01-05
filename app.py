import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import io

from utide import solve, reconstruct, ut_constants
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="Tidal Harmonic Analysis (UTide)", layout="wide")

st.title("🌊 Tidal Harmonic Analysis using UTide")
st.markdown("Paste **hourly water level data (cm)** directly.")

# -----------------------------
# Sidebar inputs
# -----------------------------
st.sidebar.header("Input Parameters")

latitude = st.sidebar.number_input(
    "Latitude (deg)",
    value=-0.870403,
    format="%.6f"
)

start_date = st.sidebar.date_input("Start date")
start_time = st.sidebar.time_input("Start time")
start_datetime = datetime.combine(start_date, start_time)

all_constituents = sorted([c.upper() for c in ut_constants.const.name])
default_constituents = ['M2', 'S2', 'N2', 'K2', 'K1', 'O1', 'P1', 'M4', 'MS4']

selected_constituents = st.sidebar.multiselect(
    "Select tidal constituents",
    options=all_constituents,
    default=[c for c in default_constituents if c in all_constituents]
)

run_button = st.sidebar.button("Run Analysis")

# -----------------------------
# Main input area
# -----------------------------
st.subheader("Paste Hourly Elevation Data (cm)")

data_text = st.text_area(
    "One value per line, hourly interval",
    height=250,
    placeholder="123.4\n125.1\n128.0\n130.2"
)

# -----------------------------
# Run analysis
# -----------------------------
if run_button:

    if not data_text.strip():
        st.error("Please paste elevation data.")
        st.stop()

    try:
        elev = pd.read_csv(
            io.StringIO(data_text),
            header=None,
            names=["elevation_cm"]
        )["elevation_cm"].astype(float).values
    except Exception as e:
        st.error(f"Error reading pasted data: {e}")
        st.stop()

    # Generate hourly datetime
    time_np = np.array(
        [start_datetime + timedelta(hours=i) for i in range(len(elev))],
        dtype="datetime64"
    )

    coef = solve(
        time_np,
        elev,
        lat=latitude,
        method="ols",
        conf_int="linear",
        constit=selected_constituents
    )

    tide_fit = reconstruct(time_np, coef)

    # -----------------------------
    # Plot
    # -----------------------------
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(time_np, elev, label="Observed")
    ax.plot(time_np, tide_fit.h, label="Fitted Tide", linewidth=2)
    ax.set_xlabel("Time")
    ax.set_ylabel("Water level (cm)")
    ax.set_title("Observed vs Fitted Tide")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

    # -----------------------------
    # Harmonic constituents table
    # -----------------------------
    names = [n.upper() for n in coef.name]
    amplitudes = coef.A
    phases = coef.g

    rows = []
    for c in selected_constituents:
        if c in names:
            i = names.index(c)
            rows.append([c, amplitudes[i], phases[i]])
        else:
            rows.append([c, np.nan, np.nan])

    rows.append(["S0", coef.mean, 0.0])

    coef_df = pd.DataFrame(
        rows,
        columns=["Constituent", "Amplitude (cm)", "Phase (deg)"]
    )

    st.subheader("Harmonic Constituents")
    st.dataframe(coef_df, use_container_width=True)

    # -----------------------------
    # Tidal datums (approximation)
    # -----------------------------
    MSL = coef.mean

    def get_amp(c):
        return amplitudes[names.index(c)] if c in names else 0

    A_M2 = get_amp("M2")
    A_S2 = get_amp("S2")
    A_K1 = get_amp("K1")
    A_O1 = get_amp("O1")

    Amp_spring = A_M2 + A_S2 + A_K1 + A_O1

    HWS  = MSL + Amp_spring
    MHWS = MSL + 0.707 * Amp_spring
    MHWL = MSL + 0.5 * (A_M2 + A_K1)
    MLWL = MSL - 0.5 * (A_M2 + A_K1)
    MLWS = MSL - 0.707 * Amp_spring
    LWS  = MSL - Amp_spring

    datum_df = pd.DataFrame({
        "Datum": ["HWS", "MHWS", "MHWL", "MSL", "MLWL", "MLWS", "LWS"],
        "Elevation (cm)": [HWS, MHWS, MHWL, MSL, MLWL, MLWS, LWS]
    })

    st.subheader("Tidal Elevations (Approximate)")
    st.table(datum_df)

    # -----------------------------
    # Accuracy metrics
    # -----------------------------
    rmse = np.sqrt(mean_squared_error(elev, tide_fit.h))
    tidal_range = HWS - LWS
    rmse_percent = (rmse / tidal_range) * 100 if tidal_range != 0 else np.nan

    st.subheader("Model Accuracy")
    st.metric("RMSE (cm)", f"{rmse:.2f}")
    st.metric("Tidal Range (cm)", f"{tidal_range:.2f}")
    st.metric("RMSE (%)", f"{rmse_percent:.2f}")

else:
    st.info("Paste elevation data, set latitude and start time, then click **Run Analysis**.")
