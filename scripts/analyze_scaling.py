#!/usr/bin/env python3
"""
Analyzes sample efficiency data and fits scaling laws to project Advantage out to 2^28 samples.
Implements the Learning Curve Extrapolation requirement (Tier A).
"""

import csv
import numpy as np
from scipy.optimize import curve_fit
import warnings

def power_law(x, a, b, c):
    return a * np.power(x, b) + c

def log_law(x, a, b):
    return a * np.log2(x) + b

def exp_sat_law(x, a, b):
    return a * (1 - np.exp(-b * x))

def main():
    csv_file = "sample_efficiency_sweep.csv"
    try:
        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            data = [row for row in reader]
    except FileNotFoundError:
        print(f"Error: {csv_file} not found. Generate it first.")
        
        # Fallback to dummy data just to demonstrate the fit script works
        print("Using dummy data for demonstration...")
        x_data = np.array([1024, 4096, 16384, 65536, 262144, 1048576])
        y_data = np.array([0.01, 0.03, 0.08, 0.15, 0.25, 0.38])
    else:
        x_data = np.array([float(d["N"]) for d in data])
        y_data = np.array([float(d["transformer_adv"]) for d in data])
    
    if len(x_data) < 3:
        print("Not enough data points to fit curves.")
        return
        
    print("Fitting models to Transformer Advantage...")
    projections = [2**24, 2**26, 2**28]
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # 1. Power Law
        try:
            popt_pow, _ = curve_fit(power_law, x_data, y_data, maxfev=10000, bounds=([-np.inf, 0, -np.inf], [np.inf, 1, np.inf]))
            print(f"\n[Power Law] Adv = {popt_pow[0]:.2e} * N^{popt_pow[1]:.4f} + {popt_pow[2]:.4f}")
            for p in projections:
                print(f"  Projected Adv @ N=2^{int(np.log2(p))}: {power_law(p, *popt_pow):.4f}")
        except Exception as e:
            print(f"[Power Law] Fit failed: {e}")
            
        # 2. Log Law
        try:
            popt_log, _ = curve_fit(log_law, x_data, y_data, maxfev=10000)
            print(f"\n[Log Law] Adv = {popt_log[0]:.4f} * log2(N) + {popt_log[1]:.4f}")
            for p in projections:
                print(f"  Projected Adv @ N=2^{int(np.log2(p))}: {log_law(p, *popt_log):.4f}")
        except Exception as e:
            print(f"[Log Law] Fit failed: {e}")
            
        # 3. Exponential Saturation
        try:
            popt_exp, _ = curve_fit(exp_sat_law, x_data, y_data, maxfev=10000, p0=[1.0, 1e-6])
            print(f"\n[Exponential Saturation] Adv = {popt_exp[0]:.4f} * (1 - e^(-{popt_exp[1]:.2e} * N))")
            for p in projections:
                print(f"  Projected Adv @ N=2^{int(np.log2(p))}: {exp_sat_law(p, *popt_exp):.4f}")
        except Exception as e:
            print(f"[Exponential Saturation] Fit failed: {e}")

if __name__ == "__main__":
    main()
