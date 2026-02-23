import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import io

from utide import solve, reconstruct, ut_constants
from datetime import datetime, timedelta, date, time
from sklearn.metrics import mean_squared_error

# -----------------------------
# Streamlit config
# -----------------------------
st.set_page_config(page_title="Tidal Harmonic Analysis (UTide)", layout="wide")

st.title("🌊 Tidal Harmonic Analysis using UTide")
st.markdown("Paste **hourly water level data (cm)** directly.")

# -----------------------------
# Session state initialization
# -----------------------------
if "results" not in st.session_state:
    st.session_state.results = None

# -----------------------------
# Sidebar inputs
# -----------------------------
st.sidebar.header("Input Parameters")

latitude = st.sidebar.number_input(
    "Latitude (deg)",
    value=-0.870403,
    format="%.6f"
)

start_date = st.sidebar.date_input("Start date", value=date(2026, 1, 1))
start_time = st.sidebar.time_input("Start time", value=time(0, 0))
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

    # Generate hourly datetime array
    time_np = np.array(
        [start_datetime + timedelta(hours=i) for i in range(len(elev))],
        dtype="datetime64"
    )

    # Harmonic analysis
    coef = solve(
        time_np,
        elev,
        lat=latitude,
        method="ols",
        conf_int="linear",
        constit=selected_constituents
    )

    tide_fit = reconstruct(time_np, coef)
    residual = elev - tide_fit.h

    # -----------------------------
    # Save results to session state
    # -----------------------------
    st.session_state.results = {
        "time": time_np,
        "observed": elev,
        "fit": tide_fit.h,
        "residual": residual,
        "coef": coef
    }

# -----------------------------
# Plot & outputs (persistent)
# -----------------------------
if st.session_state.results is not None:

    time_np = st.session_state.results["time"]
    elev = st.session_state.results["observed"]
    fit = st.session_state.results["fit"]
    residual = st.session_state.results["residual"]
    coef = st.session_state.results["coef"]

    # -----------------------------
    # Elsevier-style plot with residual
    # -----------------------------
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(12, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}
    )

    ax1.plot(
        time_np, elev,
        color="red", linewidth=2.4,
        label="Observed"
    )

    ax1.plot(
        time_np, fit,
        color="blue",
        linestyle=(0, (5, 3)),
        linewidth=1.4,
        marker="o",
        markersize=5,
        markerfacecolor="white",
        markeredgecolor="black",
        markeredgewidth=1.2,
        markevery=24,
        label="Harmonic Fit"
    )

    ax1.set_ylabel("Water Level (cm)")
    ax1.set_title("Observed and Harmonic-Fitted Tide")
    ax1.legend(frameon=False, handlelength=3)
    ax1.grid(False)

    ax2.plot(
        time_np, residual,
        color="black",
        linestyle=":",
        linewidth=1.2
    )

    ax2.axhline(0, color="black", linestyle="--", linewidth=1.0)
    ax2.set_ylabel("Residual (cm)")
    ax2.set_xlabel("Time")
    ax2.grid(False)

    plt.tight_layout()
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
    # Tidal Datums / Elevations
    # -----------------------------
    st.subheader("Tidal Elevations")

    MSL = coef.mean if hasattr(coef, 'mean') else np.mean(elev)

    A_M2 = amplitudes[names.index('M2')] if 'M2' in names else 0
    A_S2 = amplitudes[names.index('S2')] if 'S2' in names else 0
    A_K1 = amplitudes[names.index('K1')] if 'K1' in names else 0
    A_O1 = amplitudes[names.index('O1')] if 'O1' in names else 0

    Amp_spring = A_M2 + A_S2 + A_K1 + A_O1

    HWS  = MSL + Amp_spring
    MHWS = MSL + 0.707 * Amp_spring
    MHWL = MSL + 0.5 * (A_M2 + A_K1)
    MLWL = MSL - 0.5 * (A_M2 + A_K1)
    MLWS = MSL - 0.707 * Amp_spring
    LWS  = MSL - Amp_spring

    datum_df = pd.DataFrame(
        {
            "Datum": ["HWS", "MHWS", "MHWL", "MSL", "MLWL", "MLWS", "LWS"],
            "Elevation (cm)": [HWS, MHWS, MHWL, MSL, MLWL, MLWS, LWS],
            "Description": [
                "Highest Water Springs",
                "Mean High Water Springs",
                "Mean High Water Level",
                "Mean Sea Level",
                "Mean Low Water Level",
                "Mean Low Water Springs",
                "Lowest Water Springs",
            ],
        }
    )

    st.dataframe(datum_df, use_container_width=True)

    # -----------------------------
    # Model Accuracy
    # -----------------------------
    rmse = np.sqrt(mean_squared_error(elev, fit))

    tidal_range = 2 * Amp_spring
    rmse_percent = (rmse / tidal_range) * 100 if tidal_range != 0 else np.nan

    st.subheader("Model Accuracy")
    st.metric("RMSE", f"{rmse:.2f} cm / {rmse_percent:.2f} %")
    st.metric("Tidal Range", f"{tidal_range:.2f} cm")

    # -----------------------------
    # CSV download
    # -----------------------------
    output_df = pd.DataFrame({
        "datetime": pd.to_datetime(time_np),
        "observed_cm": elev,
        "harmonic_fit_cm": fit,
        "residual_cm": residual
    })

    csv_data = output_df.to_csv(index=False)

    st.subheader("Download Time Series Data")
    st.download_button(
        label="📥 Download CSV (Observed, Fit, Residual)",
        data=csv_data,
        file_name="tidal_harmonic_timeseries.csv",
        mime="text/csv"
    )

else:
    st.info("Paste elevation data, set parameters, then click **Run Analysis**.")

