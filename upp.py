# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import datetime
from calculator import load_data, get_gpu_tdp, get_region_params, calculate_metrics, get_simple_recommendations

st.set_page_config(page_title="AI Eco-Twin Dashboard", layout="wide")

# --- helper utilities ------------------------------------------------------
def find_image(basename):
    assets_dir = os.path.join(os.getcwd(), "assets")
    for ext in [".png", ".jpg", ".jpeg", ".webp"]:
        p = os.path.join(assets_dir, basename + ext)
        if os.path.exists(p):
            return p
    return None

def eco_score(co2_kg, water_l, total_energy_kwh, region_factor):
    co2_pen = min(co2_kg / 10.0, 1.0)
    water_pen = min(water_l / 50.0, 1.0)
    energy_pen = min(total_energy_kwh / 50.0, 1.0)
    region_pen = min(region_factor / 1.0, 1.0)
    penalty = 0.5 * co2_pen + 0.2 * water_pen + 0.2 * energy_pen + 0.1 * region_pen
    score = max(0, int((1 - penalty) * 100))
    return score

def save_run(record, folder="results"):
    os.makedirs(folder, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = os.path.join(folder, f"run_{ts}.csv")
    pd.DataFrame([record]).to_csv(fname, index=False)
    return fname

# --- load data --------------------------------------------------------------
gpus_df, regions_df = load_data()

# --- sidebar: inputs & scenario comparison -------------------------------
st.sidebar.header("Inputs — Primary Run")
gpu_name = st.sidebar.selectbox("GPU", gpus_df['gpu_name'].tolist())
region = st.sidebar.selectbox("Region", regions_df['region'].tolist())
runtime_hours = st.sidebar.number_input("Runtime (hours)", min_value=0.1, max_value=10000.0, value=10.0, step=0.5)
util_percent = st.sidebar.slider("GPU Utilization (%)", min_value=1, max_value=100, value=70)
utilization = util_percent / 100.0
n_gpus = st.sidebar.number_input("Number of GPUs", min_value=1, value=1, step=1)

st.sidebar.markdown("---")
st.sidebar.header("Optional: Compare With Another Scenario")
compare = st.sidebar.checkbox("Enable comparison run")
if compare:
    comp_gpu = st.sidebar.selectbox("Compare GPU", options=gpus_df['gpu_name'].tolist(), index=0, key="comp_gpu")
    comp_region = st.sidebar.selectbox("Compare Region", options=regions_df['region'].tolist(), index=0, key="comp_region")
    comp_runtime = st.sidebar.number_input("Compare Runtime (hours)", min_value=0.1, max_value=10000.0, value=20.0, step=0.5, key="comp_runtime")
    comp_util_percent = st.sidebar.slider("Compare Util (%)", min_value=1, max_value=100, value=50, key="comp_util")
    comp_util = comp_util_percent / 100.0
    comp_n_gpus = st.sidebar.number_input("Compare # GPUs", min_value=1, value=1, step=1, key="comp_ngpus")

st.sidebar.markdown("---")
if st.sidebar.button("Save this run to CSV"):
    gpu_watts = get_gpu_tdp(gpus_df, gpu_name)
    params = get_region_params(regions_df, region)
    metrics = calculate_metrics(gpu_watts, runtime_hours, utilization=utilization,
                                pue=params['pue'], emission_factor=params['emission_factor'],
                                water_factor=params['water_factor'])
    metrics['gpu_name'] = gpu_name
    metrics['region'] = region
    metrics['runtime_hours'] = runtime_hours
    metrics['utilization'] = utilization
    metrics['n_gpus'] = n_gpus
    path = save_run(metrics)
    st.sidebar.success(f"Saved run to {path}")

# --- main layout ------------------------------------------------------------
st.title("🌍 AI Eco-Twin: Energy, Carbon & Water Dashboard")
st.write("Interactive prototype: estimate energy, CO₂ and water footprint for an ML training run.")

# compute primary run
gpu_watts = get_gpu_tdp(gpus_df, gpu_name)
params = get_region_params(regions_df, region)
primary = calculate_metrics(gpu_watts, runtime_hours, utilization=utilization,
                            pue=params['pue'], emission_factor=params['emission_factor'],
                            water_factor=params['water_factor'])
primary['total_energy_kwh'] *= n_gpus
primary['co2_kg'] *= n_gpus
primary['water_l'] *= n_gpus
primary['gpu_energy_kwh'] *= n_gpus

# optional comparison run
comp = None
if compare:
    comp_gpu_watts = get_gpu_tdp(gpus_df, comp_gpu)
    comp_params = get_region_params(regions_df, comp_region)
    comp = calculate_metrics(comp_gpu_watts, comp_runtime, utilization=comp_util,
                             pue=comp_params['pue'], emission_factor=comp_params['emission_factor'],
                             water_factor=comp_params['water_factor'])
    comp['total_energy_kwh'] *= comp_n_gpus
    comp['co2_kg'] *= comp_n_gpus
    comp['water_l'] *= comp_n_gpus
    comp['gpu_energy_kwh'] *= comp_n_gpus

# --- Top-level numbers -----------------------------------------------------
col1, col2, col3, col4 = st.columns([2,2,2,2])
col1.metric("GPU Energy (kWh)", primary['gpu_energy_kwh'])
col2.metric("Total Energy (kWh)", primary['total_energy_kwh'])
col3.metric("CO₂ Emissions (kg)", primary['co2_kg'])
col4.metric("Water Usage (L)", primary['water_l'])

with st.expander("Show primary run details"):
    st.json(primary)

# --- visualization: energy breakdown chart --------------------------------
st.subheader("Energy breakdown")
breakdown_df = pd.DataFrame({
    "component": ["GPU (compute)", "DC overhead (PUE extra)"],
    "kWh": [
        primary['gpu_energy_kwh'],
        primary['total_energy_kwh'] - primary['gpu_energy_kwh']
    ]
})
fig_break = px.pie(breakdown_df, names="component", values="kWh", title="GPU vs. Data-center overhead")
st.plotly_chart(fig_break, use_container_width=True)

# --- multi-metric bar chart (primary vs comparison) ------------------------
st.subheader("Compare metrics")
metrics_list = ["gpu_energy_kwh", "total_energy_kwh", "co2_kg", "water_l"]
label_map = {"gpu_energy_kwh":"GPU kWh", "total_energy_kwh":"Total kWh", "co2_kg":"CO2 kg", "water_l":"Water L"}

comp_df = pd.DataFrame()
comp_df["metric"] = [label_map[m] for m in metrics_list]
comp_df["primary"] = [primary[m] for m in metrics_list]
if comp:
    comp_df["comparison"] = [comp[m] for m in metrics_list]
else:
    comp_df["comparison"] = [0]*len(metrics_list)

fig_comp = go.Figure()
fig_comp.add_trace(go.Bar(x=comp_df["metric"], y=comp_df["primary"], name="Primary"))
if comp:
    fig_comp.add_trace(go.Bar(x=comp_df["metric"], y=comp_df["comparison"], name="Comparison"))
fig_comp.update_layout(barmode='group', title="Primary vs Comparison run")
st.plotly_chart(fig_comp, use_container_width=True)

# --- Eco-Twin Visualization (Primary + Comparison) ------------------------
st.subheader("🌲 Eco-Twin Visualization")

def pick_forest_image(co2):
    if co2 < 1:
        return "forest_good", "Healthy Forest 🌳"
    elif co2 < 5:
        return "forest_mid", "Moderate Forest 🌲"
    else:
        return "forest_bad", "Damaged Forest 🔥"

def show_forest(col, co2, title):
    basename, caption = pick_forest_image(co2)
    path = find_image(basename)
    col.markdown(f"**{title}**")
    if path:
        col.image(path, caption=caption, use_container_width=True)
    else:
        col.warning(f"⚠️ Image not found for {title}: {basename}")

if comp:
    col1, col2 = st.columns(2)
    show_forest(col1, primary['co2_kg'], "Primary Run")
    show_forest(col2, comp['co2_kg'], "Comparison Run")
else:
    show_forest(st, primary['co2_kg'], "Primary Run")

# --- Carbon offset + water bottles ----------------------------------------
st.subheader("Interpretation & Offsets")
trees = primary['co2_kg'] / 20.0
st.write(f"Equivalent trees (annual absorption): **{trees:.2f} trees**")
st.progress(min(1.0, primary['co2_kg'] / 10.0))
bottles = primary['water_l'] / 0.5
st.write(f"Equivalent water bottles (0.5 L): **{int(bottles)} bottles**")

# --- Eco Score -------------------------------------------------------------
st.subheader("Eco Score")
score = eco_score(primary['co2_kg'], primary['water_l'], primary['total_energy_kwh'], params['emission_factor'])
gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=score,
    gauge={"axis": {"range": [0,100]},
           "bar": {"color": "green" if score>70 else "orange" if score>40 else "red"}},
    title={"text":"Eco Score (100 = best)"}
))
st.plotly_chart(gauge, use_container_width=True)

# --- Dynamic recommendations ----------------------------------------------
st.subheader("Recommendations")
base_tips = get_simple_recommendations(runtime_hours, region)
for t in base_tips:
    st.write("- " + t)

if primary['co2_kg'] > 5:
    st.warning("CO₂ is relatively high — consider moving training to a cleaner-grid region, or use lower runtime / more efficient instances.")
if runtime_hours > 24:
    st.info("Long runtime detected — consider checkpointing, mixed precision, or smaller batches.")
if primary['water_l'] > 50:
    st.info("High water usage — consider scheduling in cooler months/regions or using lower PUE datacenters.")

# --- Export results --------------------------------------------------------
st.subheader("Export & Save")
results_table = pd.DataFrame({
    "metric": [label_map[m] for m in metrics_list],
    "primary": [primary[m] for m in metrics_list],
    "comparison": [comp[m] for m in metrics_list] if comp else [None]*len(metrics_list)
})
st.dataframe(results_table)
csv_bytes = results_table.to_csv(index=False).encode('utf-8')
st.download_button("Download metrics CSV", data=csv_bytes, file_name="thura_run_metrics.csv", mime="text/csv")

if st.button("Save run summary (detailed)"):
    record = {
        "timestamp": datetime.datetime.now().isoformat(),
        "gpu": gpu_name, "region": region, "runtime_hours": runtime_hours,
        "utilization": utilization, "n_gpus": n_gpus,
        "gpu_energy_kwh": primary['gpu_energy_kwh'],
        "total_energy_kwh": primary['total_energy_kwh'],
        "co2_kg": primary['co2_kg'], "water_l": primary['water_l'],
        "eco_score": score
    }
    saved = save_run(record)
    st.success(f"Saved run summary: {saved}")

st.markdown("---")
st.markdown("**Notes:** The model uses simplified factors (TDP, PUE, emission factors, water factors). For the final report cite reliable sources (cloud provider docs, IEA, LCA studies).")
