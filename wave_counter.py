#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
=============================================================================
Redefining Epidemiological Waves: Structural Stability and Global Empirical Validation in Sobolev Spaces
=============================================================================
Author: Santi García-Cremades
Institution: Miguel Hernandez University of Elche, Center of Operations Research
Description: 
This script fetches global COVID-19 data (JHU) and World Bank population data, 
applies an H3 Sobolev space regularization to filter institutional noise, and 
extracts the exact kinematic boundaries (start, peak, end) of epidemic waves.
=============================================================================
"""

import os
import numpy as np
import pandas as pd
import requests
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# 0. HYPERPARAMETERS & CONFIGURATION
# =============================================================================

LAMBDA_H3 = 5000 
MIN_AMPLITUDE_IA14 = 50  
MIN_POPULATION = 100000  
FIG_DIR = "figures"

if not os.path.exists(FIG_DIR):
    os.makedirs(FIG_DIR)

# =============================================================================
# 1. CORE MATHEMATICAL FUNCTIONS
# =============================================================================

def tikhonov_h3_sparse(y, lam=5000):
    """
    Projects the raw time series 'y' into the Sobolev space H^3
    using sparse matrix operations for O(n) computational efficiency.
    """
    n = len(y)
    if n < 4:
        return y
    D3 = sparse.diags([-1, 3, -3, 1], [0, 1, 2, 3], shape=(n-3, n))
    I = sparse.eye(n)
    A = I + lam * D3.T.dot(D3)
    return spsolve(A, y)

def find_kinematic_waves(x_h3, min_amplitude=50):
    """
    Identifies structurally stable epidemic waves based on the 
    zero-crossings of velocity and acceleration in H^3 space.
    """
    v = np.gradient(x_h3)
    a = np.gradient(v)
    
    waves = []
    in_wave = False
    start_idx = 0
    
    for i in range(1, len(v) - 1):
        # 1. Wave Start
        if v[i-1] <= 0 and v[i] > 0 and not in_wave:
            in_wave = True
            start_idx = i
            
        # 2. Kinematic Peak
        elif v[i-1] > 0 and v[i] <= 0 and a[i] < 0 and in_wave:
            peak_idx = i
            end_idx = peak_idx
            
            # 3. Wave End
            while end_idx < len(v) - 1 and v[end_idx] <= 0:
                end_idx += 1
                
            # 4. Amplitude Validation
            if (x_h3[peak_idx] - x_h3[start_idx]) >= min_amplitude:
                waves.append({
                    'start_idx': start_idx,
                    'peak_idx': peak_idx,
                    'end_idx': end_idx
                })
            in_wave = False
            
    return waves

# =============================================================================
# 2. DATA FETCHING (JHU & WORLD BANK)
# =============================================================================

print("1. Fetching data from Johns Hopkins University (JHU)...")
url_jhu = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
df_jhu = pd.read_csv(url_jhu)
df_jhu = df_jhu.drop(columns=['Lat', 'Long', 'Province/State']).groupby('Country/Region').sum().T
df_jhu.index = pd.to_datetime(df_jhu.index)

print("2. Fetching population data from the World Bank...")
url_pop = "https://api.worldbank.org/v2/country/all/indicator/SP.POP.TOTL?format=json&per_page=300&date=2021"
response = requests.get(url_pop).json()
pop_data = {item['country']['value']: item['value'] for item in response[1] if item['value'] is not None}

# Country name normalization dictionary (mapping World Bank to JHU)
pop_map = {
    'US': 'United States', 'Russia': 'Russian Federation', 
    'Korea, South': 'Korea, Rep.', 'Iran': 'Iran, Islamic Rep.',
    'Egypt': 'Egypt, Arab Rep.', 'Venezuela': 'Venezuela, RB',
    'Czechia': 'Czech Republic', 'Slovakia': 'Slovak Republic',
    'Syria': 'Syrian Arab Republic', 'Yemen': 'Yemen, Rep.'
}

all_waves_detailed = []
summary_data = []
countries_plotted = 0

print("3. Computing kinematics, generating tables, and rendering plots (This will take a few minutes)...")

# =============================================================================
# 3. MAIN EXECUTION LOOP
# =============================================================================

for country in df_jhu.columns:
    wb_name = pop_map.get(country, country)
    pop = pop_data.get(wb_name, None)
    
    if pop is None or pop < MIN_POPULATION:
        continue
        
    # Calculate True Incidence and IA14
    cumulative = df_jhu[country].values
    incidence = np.diff(cumulative, prepend=0)
    incidence[incidence < 0] = 0  # Clean negative data corrections
    
    df_country = pd.DataFrame({'Date': df_jhu.index, 'Incidence': incidence})
    df_country['IA14'] = df_country['Incidence'].rolling(window=14, min_periods=1).sum() / (pop / 100000)
    
    ia14_raw = df_country['IA14'].values
    ia14_h3 = tikhonov_h3_sparse(ia14_raw, LAMBDA_H3)
    df_country['IA14_H3'] = ia14_h3
    
    # Detect Waves
    waves = find_kinematic_waves(ia14_h3, MIN_AMPLITUDE_IA14)
    
    if len(waves) == 0:
        continue
        
    total_epidemic_days = 0
    total_infected_in_waves = 0
    
    for w_idx, w in enumerate(waves):
        start = df_country['Date'].iloc[w['start_idx']]
        peak = df_country['Date'].iloc[w['peak_idx']]
        end = df_country['Date'].iloc[w['end_idx']]
        duration = w['end_idx'] - w['start_idx']
        peak_ia = ia14_h3[w['peak_idx']]
        
        infected_in_wave = incidence[w['start_idx']:w['end_idx']].sum()
        
        total_epidemic_days += duration
        total_infected_in_waves += infected_in_wave
        
        all_waves_detailed.append({
            'Country': country,
            'Wave_Number': w_idx + 1,
            'Start_Date': start.strftime('%Y-%m-%d'),
            'Peak_Date': peak.strftime('%Y-%m-%d'),
            'End_Date': end.strftime('%Y-%m-%d'),
            'Duration_Days': duration,
            'Peak_IA14': round(peak_ia, 2),
            'Infected_In_Wave': int(infected_in_wave)
        })
        
    summary_data.append({
        'Country': country,
        'Total_Waves': len(waves),
        'Total_Infected_In_Epidemic_State': int(total_infected_in_waves),
        'Total_Epidemic_Days': total_epidemic_days
    })
    
    # Generate and save plot
    plt.figure(figsize=(12, 6))
    plt.plot(df_country['Date'], ia14_raw, color='lightgray', alpha=0.6, label='Raw IA14')
    plt.plot(df_country['Date'], ia14_h3, color='black', linewidth=1.5, label='H3 Regularized IA14')
    
    for w in waves:
        plt.axvspan(df_country['Date'].iloc[w['start_idx']], 
                    df_country['Date'].iloc[w['end_idx']], 
                    color='red', alpha=0.2)
        plt.axvline(df_country['Date'].iloc[w['peak_idx']], color='darkred', linestyle='--', alpha=0.7)
        
    plt.title(f"{country} - Topo-Kinematic Epidemic Waves ($H^3$ Space)", fontsize=14)
    plt.ylabel("Incidence per 100k (IA14)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/{country.replace('*', '').replace(' ', '_')}_H3.png", dpi=150)
    plt.close()
    
    countries_plotted += 1

# =============================================================================
# 4. EXPORT RESULTS
# =============================================================================

pd.DataFrame(all_waves_detailed).to_csv("Detailed_Waves_H3.csv", index=False)
df_summary = pd.DataFrame(summary_data).sort_values(by='Total_Epidemic_Days', ascending=False).reset_index(drop=True)
df_summary.to_csv("Country_Summary_H3.csv", index=False)

print("\nProcess Completed Successfully!")
print(f"Total countries analyzed and plotted: {countries_plotted}")
print("\nOutput files generated:")
print(f" - '{FIG_DIR}/' folder (Contains all kinematic plots)")
print(" - 'Detailed_Waves_H3.csv' (Wave-by-wave phase boundaries)")
print(" - 'Country_Summary_H3.csv' (Global topological condemnation summary)")
