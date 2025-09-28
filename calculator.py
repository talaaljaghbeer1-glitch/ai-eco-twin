# calculator.py
import pandas as pd

def load_data(gpu_csv='data/gpus.csv', region_csv='data/regions.csv'):
    gpus = pd.read_csv(gpu_csv)
    regions = pd.read_csv(region_csv)
    return gpus, regions

def get_gpu_tdp(gpus_df, gpu_name):
    row = gpus_df[gpus_df['gpu_name'] == gpu_name]
    if row.empty:
        raise ValueError(f"GPU {gpu_name} not found in gpus.csv")
    return float(row.iloc[0]['tDP_watts'])

def get_region_params(regions_df, region):
    row = regions_df[regions_df['region'] == region]
    if row.empty:
        raise ValueError(f"Region {region} not found in regions.csv")
    return {
        'emission_factor': float(row.iloc[0]['emission_factor_kgCO2_per_kWh']),
        'pue': float(row.iloc[0]['pue']),
        'water_factor': float(row.iloc[0]['water_factor_L_per_kWh'])
    }

def calculate_metrics(gpu_watts, runtime_hours, utilization=0.7, pue=1.2, emission_factor=0.4, water_factor=1.8):
    gpu_kw = gpu_watts / 1000.0  # convert W to kW
    gpu_energy = gpu_kw * utilization * runtime_hours  # kWh consumed by GPU itself
    total_energy = gpu_energy * pue  # add PUE overhead
    co2_kg = total_energy * emission_factor
    water_l = total_energy * water_factor
    return {
        'gpu_energy_kwh': round(gpu_energy, 4),
        'total_energy_kwh': round(total_energy, 4),
        'co2_kg': round(co2_kg, 4),
        'water_l': round(water_l, 4)
    }

def get_simple_recommendations(runtime_hours, region):
    tips = []
    if runtime_hours > 8:
        tips.append("🟢 Consider using mixed precision or gradient accumulation to reduce training time.")
    if region.lower() in ['asia']:
        tips.append("🟢 Running in cleaner grid regions (e.g., Europe) can reduce CO₂ emissions.")
    tips.append("🟢 Profile your training to reduce idle GPU time and optimize batch sizes.")
    return tips

# Quick test
if __name__ == "__main__":
    gpus, regions = load_data()
    tdp = get_gpu_tdp(gpus, 'A100')
    params = get_region_params(regions, 'Europe')
    metrics = calculate_metrics(tdp, runtime_hours=10, utilization=0.7, 
                                pue=params['pue'],
                                emission_factor=params['emission_factor'],
                                water_factor=params['water_factor'])
    print(metrics)
    print(get_simple_recommendations(10, 'Europe'))
