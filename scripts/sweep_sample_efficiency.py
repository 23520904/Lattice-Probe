#!/usr/bin/env python3
"""
Automates the Classical-vs-ML Sample Efficiency sweep (Tier A).
Generates datasets of exponentially increasing sizes, trains models, 
and records the Advantage of Transformer, Logistic Regression, MLP, and Chi^2.
"""

import csv
import json
import subprocess
import sys
from pathlib import Path

def main():
    # Sweep N from 2^10 (1024) to 2^20 (1M)
    N_list = [2**i for i in range(10, 21, 2)]
    
    out_csv = "sample_efficiency_sweep.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["N", "transformer_adv", "chi2_adv", "lr_adv", "mlp_adv"])
    
    for n in N_list:
        print(f"\n=============================================")
        print(f"Sweeping N = {n:,} samples")
        print(f"=============================================")
        
        train_dir = f"data/sweep_train_{n}"
        test_dir = f"data/sweep_test_{n}"
        
        print(f"1/3 Generating datasets...")
        subprocess.run([
            "python", "scripts/generate_dataset.py", 
            "--param-set", "ML-KEM-512", 
            "--n-samples", str(n), 
            "--output-dir", train_dir, 
            "--quiet"
        ], check=True)
        
        subprocess.run([
            "python", "scripts/generate_dataset.py", 
            "--param-set", "ML-KEM-512", 
            "--n-samples", "8192", 
            "--output-dir", test_dir, 
            "--quiet"
        ], check=True)
        
        print(f"2/3 Training Transformer...")
        ckpt_dir = f"checkpoints/sweep_tf_{n}"
        subprocess.run([
            "python", "scripts/train.py", 
            "--param-set", "ML-KEM-512", 
            "--model", "transformer", 
            "--train-dir", train_dir, 
            "--val-dir", test_dir, 
            "--output-dir", ckpt_dir,
            "--epochs", "20",
            "--batch-size", "128"
        ], check=True)
        
        print(f"3/3 Evaluating all baselines...")
        eval_json = f"{ckpt_dir}/eval.json"
        subprocess.run([
            "python", "scripts/evaluate.py",
            "--checkpoint", f"{ckpt_dir}/best.pt",
            "--model", "transformer",
            "--param-set", "ML-KEM-512",
            "--test-dir", test_dir,
            "--train-dir", train_dir,
            "--output-json", eval_json
        ], check=True)
        
        with open(eval_json) as f:
            res = json.load(f)
            
        tf_adv = res.get("model_advantage", 0)
        chi2_adv = res.get("chi2_adv", 0)
        lr_adv = res.get("lr_adv", 0)
        mlp_adv = res.get("mlp_adv", 0)
        
        with open(out_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([n, tf_adv, chi2_adv, lr_adv, mlp_adv])
            
        print(f"-> N={n:,} | TF_Adv: {tf_adv:.4f} | Chi2_Adv: {chi2_adv:.4f} | LR_Adv: {lr_adv:.4f}")

    print(f"\nSweep complete! Results saved to {out_csv}")

if __name__ == "__main__":
    main()
