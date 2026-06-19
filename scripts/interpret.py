#!/usr/bin/env python3
"""
Model Interpretability (Tier C).
Performs Feature Occlusion to determine the most critical coefficients 
driving the Neural Network's Cryptographic Advantage.
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

# Allow running from root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from latticeprobe.params import get_params
from scripts.evaluate import load_checkpoint, _build_loader, _inference

def parse_args():
    p = argparse.ArgumentParser(description="Feature Occlusion Interpretability")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model", required=True, choices=["transformer", "gnn"])
    p.add_argument("--param-set", required=True)
    p.add_argument("--test-dir", required=True)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--n-occlude", type=int, default=2000, help="Number of samples to occlude")
    p.add_argument("--device", default="auto")
    p.add_argument("--repr", default="coeff", choices=["coeff", "ntt", "dual"])
    return p.parse_args()

def main():
    args = parse_args()
    
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        
    params = get_params(args.param_set)
    
    model, _, _ = load_checkpoint(args.checkpoint, args.model, params, device)
    model.eval()
    
    loader, ds = _build_loader(args.model, args.test_dir, params, args.batch_size, repr_type=args.repr)
    
    print(f"Running Baseline Inference...")
    baseline_logits, labels = _inference(model, loader, device, args.model)
    baseline_auroc = roc_auc_score(labels, baseline_logits)
    baseline_adv = 2 * baseline_auroc - 1
    print(f"Baseline AUROC: {baseline_auroc:.4f} | Advantage: {baseline_adv:.4f}")
    
    if args.model == "transformer":
        print(f"\nRunning Feature Occlusion on {args.n_occlude} samples...")
        
        X, Y = [], []
        for x, y in loader:
            X.append(x)
            Y.append(y)
            if sum(len(b) for b in X) >= args.n_occlude:
                break
        
        X = torch.cat(X)[:args.n_occlude]
        Y = torch.cat(Y)[:args.n_occlude].numpy()
        
        seq_len = X.shape[1]
        importance = np.zeros(seq_len)
        
        for i in range(seq_len):
            # Occlude token i
            X_occ = X.clone()
            X_occ[:, i] = 0
            
            logits_occ = []
            with torch.no_grad():
                for j in range(0, len(X_occ), args.batch_size):
                    batch = X_occ[j:j+args.batch_size].to(device)
                    out = model(batch).cpu().squeeze(1).numpy()
                    logits_occ.append(out)
            
            logits_occ = np.concatenate(logits_occ)
            
            try:
                occ_auroc = roc_auc_score(Y, logits_occ)
                # Drop in Advantage
                drop = baseline_adv - (2 * occ_auroc - 1)
            except ValueError:
                drop = 0
                
            importance[i] = drop
            
        print("\nTop 20 Most Important Features (by Advantage drop):")
        top_idx = np.argsort(importance)[::-1][:20]
        
        for rank, idx in enumerate(top_idx, 1):
            if idx < params.k * params.n:
                poly = idx // params.n
                coeff = idx % params.n
                name = f"a[{poly}][{coeff}]"
            else:
                idx_b = idx - params.k * params.n
                name = f"b[{idx_b}]"
                
            if args.repr == "dual":
                if idx >= (params.k + 1) * params.n:
                    name += " (NTT)"
                    
            print(f"{rank:2d}. {name:15s} : Drop = {importance[idx]:.4f}")
            
    else:
        print("Feature occlusion currently implemented for Transformer sequence representations only.")

if __name__ == "__main__":
    main()
