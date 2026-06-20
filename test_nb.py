#!rm -rf /content/Lattice-Probe

import torch
print(f"PyTorch version: {torch.__version__}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    raise RuntimeError("GPU is unavailable. Please enable GPU in Colab (Runtime -> Change runtime type).")

import os
import subprocess

REPO_URL = "https://github.com/23520904/Lattice-Probe.git"
REPO_DIR = "/content/Lattice-Probe"

if not os.path.exists(REPO_DIR):
    print(f"Cloning repository from {REPO_URL}...")
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
else:
    print("Repository already exists. Skipping clone.")

os.chdir(REPO_DIR)
print(f"Current working directory: {os.getcwd()}")

print("Installing dependencies...")
subprocess.run(["pip", "install", "-q", "numpy", "scipy", "scikit-learn", "pandas", "matplotlib", "tqdm", "pynacl", "torch-geometric", "openpyxl"], check=True)
print("Dependencies installed successfully.")

print("Verifying CLI commands...")
subprocess.run(["python", "scripts/generate_dataset.py", "--help"], check=True, stdout=subprocess.DEVNULL)
subprocess.run(["python", "scripts/train.py", "--help"], check=True, stdout=subprocess.DEVNULL)
subprocess.run(["python", "scripts/evaluate.py", "--help"], check=True, stdout=subprocess.DEVNULL)
print("CLI verification passed.")

import subprocess

for script in [
    "scripts/generate_dataset.py",
    "scripts/train.py",
    "scripts/evaluate.py",
]:
    print(f"\n===== {script} =====")

    result = subprocess.run(
        ["python", script, "--help"],
        capture_output=True,
        text=True
    )

    print("Return code:", result.returncode)

    if result.stdout:
        print("\nSTDOUT:")
        print(result.stdout[:2000])

    if result.stderr:
        print("\nSTDERR:")
        print(result.stderr[:2000])
PARAM_SET = "ML-KEM-512"

TRAIN_SAMPLES = 1048576
VAL_SAMPLES = 8192
TEST_SAMPLES = 8192

MODEL = "transformer"

BASE_DIR = "experiments/mlkem512"
DATA_DIR = f"{BASE_DIR}/data"
CKPT_DIR = f"{BASE_DIR}/checkpoints"
RESULTS_DIR = f"{BASE_DIR}/results"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print("Configuration loaded.")

train_dir = f"{DATA_DIR}/train"
val_dir = f"{DATA_DIR}/val"
test_dir = f"{DATA_DIR}/test"

# Generate Train
if not os.path.exists(f"{train_dir}/secrets.npy"):
    print("Generating train dataset...")
    subprocess.run([
        "python", "scripts/generate_dataset.py",
        "--param-set", PARAM_SET,
        "--n-samples", str(TRAIN_SAMPLES),
        "--output-dir", train_dir
    ], check=True)
else:
    print("Train dataset already exists. Skipping.")

# Generate Val
if not os.path.exists(f"{val_dir}/secrets.npy"):
    print("Generating validation dataset (reusing train secret)...")
    subprocess.run([
        "python", "scripts/generate_dataset.py",
        "--param-set", PARAM_SET,
        "--n-samples", str(VAL_SAMPLES),
        "--output-dir", val_dir,
        "--secret-file", f"{train_dir}/secrets.npy"
    ], check=True)
else:
    print("Val dataset already exists. Skipping.")

# Generate Test
if not os.path.exists(f"{test_dir}/secrets.npy"):
    print("Generating test dataset (reusing train secret)...")
    subprocess.run([
        "python", "scripts/generate_dataset.py",
        "--param-set", PARAM_SET,
        "--n-samples", str(TEST_SAMPLES),
        "--output-dir", test_dir,
        "--secret-file", f"{train_dir}/secrets.npy"
    ], check=True)
else:
    print("Test dataset already exists. Skipping.")

ckpt_out = f"{CKPT_DIR}/transformer"
best_model_path = f"{ckpt_out}/best.pt"

if not os.path.exists(best_model_path):
    print("Training transformer model...")
    subprocess.run([
        "python", "scripts/train.py",
        "--param-set", PARAM_SET,
        "--model", MODEL,
        "--train-dir", train_dir,
        "--val-dir", val_dir,
        "--output-dir", ckpt_out,
        "--epochs", "20",
        "--compute-log", f"{ckpt_out}/compute_log.csv"
    ], check=True)
else:
    print("Model already trained. Skipping.")

import json

res_standard_path = f"{RESULTS_DIR}/results_standard.json"

if not os.path.exists(res_standard_path):
    print("Evaluating model...")
    subprocess.run([
        "python", "scripts/evaluate.py",
        "--checkpoint", best_model_path,
        "--model", MODEL,
        "--param-set", PARAM_SET,
        "--test-dir", test_dir,
        "--train-dir", train_dir,
        "--output-json", res_standard_path
    ], check=True)
else:
    print("Evaluation results already exist. Skipping.")

with open(res_standard_path, "r") as f:
    res = json.load(f)

print("| Metric | Value |")
print("|---|---|")
print(f"| AUROC | {res.get('model_auroc', 'N/A'):.4f} |")
print(f"| Advantage | {res.get('model_advantage', 'N/A'):.4f} |")
print(f"| Cohen d | {res.get('cohens_d', 'N/A'):.4f} |")
print(f"| Logit Separation | {res.get('logit_separation', 'N/A'):.4f} |")

res_perm_path = f"{RESULTS_DIR}/results_permutation.json"

if not os.path.exists(res_perm_path):
    print("Running permutation test...")
    subprocess.run([
        "python", "scripts/evaluate.py",
        "--checkpoint", best_model_path,
        "--model", MODEL,
        "--param-set", PARAM_SET,
        "--test-dir", test_dir,
        "--train-dir", train_dir,
        "--shuffle-labels",
        "--output-json", res_perm_path
    ], check=True)
else:
    print("Permutation test results already exist. Skipping.")

with open(res_perm_path, "r") as f:
    res_perm = json.load(f)

auroc_perm = res_perm.get("model_auroc", 0.5)
adv_perm = res_perm.get("model_advantage", 0.0)

print(f"Expected: AUROC ≈ 0.5, Advantage ≈ 0")
print(f"Actual: AUROC = {auroc_perm:.4f}, Advantage = {adv_perm:.4f}")

if abs(auroc_perm - 0.5) > 0.05:
    print("⚠️ WARNING: Permutation test violated! Model may be learning artifacts.")
else:
    print("✅ Permutation test passed.")

secret_b_dir = f"{DATA_DIR}/secret_B"
res_cross_path = f"{RESULTS_DIR}/results_cross_secret.json"

if not os.path.exists(f"{secret_b_dir}/secrets.npy"):
    print("Generating Secret B test set...")
    subprocess.run([
        "python", "scripts/generate_dataset.py",
        "--param-set", PARAM_SET,
        "--n-samples", str(TEST_SAMPLES),
        "--output-dir", secret_b_dir
    ], check=True)

if not os.path.exists(res_cross_path):
    print("Evaluating on Secret B...")
    subprocess.run([
        "python", "scripts/evaluate.py",
        "--checkpoint", best_model_path,
        "--model", MODEL,
        "--param-set", PARAM_SET,
        "--test-dir", secret_b_dir,
        "--output-json", res_cross_path
    ], check=True)

with open(res_standard_path, "r") as f:
    res_same = json.load(f)
with open(res_cross_path, "r") as f:
    res_cross = json.load(f)

print("| Metric | Same Secret | Cross Secret |")
print("|---|---|---|")
print(f"| AUROC | {res_same.get('model_auroc', 'N/A'):.4f} | {res_cross.get('model_auroc', 'N/A'):.4f} |")
print(f"| Advantage | {res_same.get('model_advantage', 'N/A'):.4f} | {res_cross.get('model_advantage', 'N/A'):.4f} |")

import pandas as pd
import matplotlib.pyplot as plt

NOISE_SCALES = [1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50]
phase_csv = f"{RESULTS_DIR}/phase_transition.csv"

results = []

if not os.path.exists(phase_csv):
    for scale in NOISE_SCALES:
        print(f"\n--- Noise Scale: {scale:.2f} ---")
        out_dir = f"{DATA_DIR}/noise_{scale:.2f}"
        ckpt_dir = f"{CKPT_DIR}/noise_{scale:.2f}"
        json_path = f"{RESULTS_DIR}/noise_{scale:.2f}.json"
        
        if not os.path.exists(json_path):
            subprocess.run(["python", "scripts/generate_dataset.py", "--param-set", PARAM_SET, "--n-samples", "10240", "--output-dir", f"{out_dir}/train", "--noise-scale", str(scale), "--quiet"], check=True)
            subprocess.run(["python", "scripts/generate_dataset.py", "--param-set", PARAM_SET, "--n-samples", "1024", "--output-dir", f"{out_dir}/val", "--secret-file", f"{out_dir}/train/secrets.npy", "--noise-scale", str(scale), "--quiet"], check=True)
            subprocess.run(["python", "scripts/train.py", "--param-set", PARAM_SET, "--model", MODEL, "--train-dir", f"{out_dir}/train", "--val-dir", f"{out_dir}/val", "--output-dir", ckpt_dir, "--epochs", "5"], check=True)
            subprocess.run(["python", "scripts/evaluate.py", "--checkpoint", f"{ckpt_dir}/best.pt", "--model", MODEL, "--param-set", PARAM_SET, "--test-dir", f"{out_dir}/val", "--output-json", json_path], check=True)
        
        with open(json_path, "r") as f:
            data = json.load(f)
            results.append({"noise_scale": scale, "advantage": data.get("model_advantage", 0)})
            
    df = pd.DataFrame(results)
    df.to_csv(phase_csv, index=False)
else:
    df = pd.read_csv(phase_csv)

plt.figure(figsize=(8,5))
plt.plot(df["noise_scale"], df["advantage"], marker='o')
plt.title("Advantage vs Noise Scale")
plt.xlabel("Noise Scale")
plt.ylabel("Advantage")
plt.grid(True)
plt.gca().invert_xaxis()
plt.show()

SECRET_COUNTS = [1, 10, 100, 1000]
div_csv = f"{RESULTS_DIR}/secret_diversity.csv"
div_results = []

if not os.path.exists(div_csv):
    for count in SECRET_COUNTS:
        print(f"\n--- Secrets: {count} ---")
        out_dir = f"{DATA_DIR}/sec_div_{count}"
        ckpt_dir = f"{CKPT_DIR}/sec_div_{count}"
        json_path = f"{RESULTS_DIR}/sec_div_{count}.json"
        
        if not os.path.exists(json_path):
            subprocess.run(["python", "scripts/generate_dataset.py", "--param-set", PARAM_SET, "--n-samples", "10240", "--output-dir", f"{out_dir}/train", "--num-secrets", str(count), "--quiet"], check=True)
            subprocess.run(["python", "scripts/generate_dataset.py", "--param-set", PARAM_SET, "--n-samples", "1024", "--output-dir", f"{out_dir}/val", "--num-secrets", str(count), "--quiet"], check=True)
            subprocess.run(["python", "scripts/train.py", "--param-set", PARAM_SET, "--model", MODEL, "--train-dir", f"{out_dir}/train", "--val-dir", f"{out_dir}/val", "--output-dir", ckpt_dir, "--epochs", "5"], check=True)
            subprocess.run(["python", "scripts/evaluate.py", "--checkpoint", f"{ckpt_dir}/best.pt", "--model", MODEL, "--param-set", PARAM_SET, "--test-dir", f"{out_dir}/val", "--output-json", json_path], check=True)
        
        with open(json_path, "r") as f:
            data = json.load(f)
            div_results.append({"num_secrets": count, "advantage": data.get("model_advantage", 0)})
            
    df_div = pd.DataFrame(div_results)
    df_div.to_csv(div_csv, index=False)
else:
    df_div = pd.read_csv(div_csv)

plt.figure(figsize=(8,5))
plt.plot(df_div["num_secrets"], df_div["advantage"], marker='s', color='orange')
plt.title("Advantage vs Number of Secrets")
plt.xlabel("Number of Secrets")
plt.ylabel("Advantage")
plt.xscale("log")
plt.grid(True)
plt.show()

SAMPLE_COUNTS = [1024, 4096, 16384, 65536, 262144, 1048576, 4194304]
eff_csv = f"{RESULTS_DIR}/sample_efficiency.csv"
eff_results = []

if not os.path.exists(eff_csv):
    for count in SAMPLE_COUNTS:
        print(f"\n--- Samples: {count} ---")
        out_dir = f"{DATA_DIR}/eff_{count}"
        ckpt_dir = f"{CKPT_DIR}/eff_{count}"
        json_path = f"{RESULTS_DIR}/eff_{count}.json"
        
        if not os.path.exists(json_path):
            subprocess.run(["python", "scripts/generate_dataset.py", "--param-set", PARAM_SET, "--n-samples", str(count), "--output-dir", f"{out_dir}/train", "--quiet"], check=True)
            subprocess.run(["python", "scripts/generate_dataset.py", "--param-set", PARAM_SET, "--n-samples", "1024", "--output-dir", f"{out_dir}/val", "--secret-file", f"{out_dir}/train/secrets.npy", "--quiet"], check=True)
            subprocess.run(["python", "scripts/train.py", "--param-set", PARAM_SET, "--model", MODEL, "--train-dir", f"{out_dir}/train", "--val-dir", f"{out_dir}/val", "--output-dir", ckpt_dir, "--epochs", "10"], check=True)
            subprocess.run(["python", "scripts/evaluate.py", "--checkpoint", f"{ckpt_dir}/best.pt", "--model", MODEL, "--param-set", PARAM_SET, "--test-dir", f"{out_dir}/val", "--output-json", json_path], check=True)
        
        with open(json_path, "r") as f:
            data = json.load(f)
            eff_results.append({"samples": count, "advantage": data.get("model_advantage", 0)})
            
    df_eff = pd.DataFrame(eff_results)
    df_eff.to_csv(eff_csv, index=False)
else:
    df_eff = pd.read_csv(eff_csv)

plt.figure(figsize=(8,5))
plt.plot(df_eff["samples"], df_eff["advantage"], marker='^', color='green')
plt.title("Advantage vs Samples")
plt.xlabel("Training Samples")
plt.ylabel("Advantage")
plt.xscale("log")
plt.grid(True)
plt.show()

PAPER_RESULTS = {
    "ML-KEM-512": {
        "AUROC": 0.505,
        "Advantage": 0.010
    }
}

print("Comparing reproduced results with paper...")
if PARAM_SET in PAPER_RESULTS:
    paper = PAPER_RESULTS[PARAM_SET]
    our_auroc = res.get('model_auroc', 0)
    our_adv = res.get('model_advantage', 0)
    
    auroc_diff = abs(our_auroc - paper["AUROC"])
    adv_diff = abs(our_adv - paper["Advantage"])
    
    print(f"| Metric | Paper | Reproduced | Abs Diff |")
    print(f"|---|---|---|---|")
    print(f"| AUROC | {paper['AUROC']:.4f} | {our_auroc:.4f} | {auroc_diff:.4f} |")
    print(f"| Advantage | {paper['Advantage']:.4f} | {our_adv:.4f} | {adv_diff:.4f} |")
else:
    print(f"No paper baseline available for {PARAM_SET}")

print("Exporting all results to Excel...")
excel_path = f"{RESULTS_DIR}/results_summary.xlsx"

with pd.ExcelWriter(excel_path) as writer:
    if os.path.exists(phase_csv):
        pd.read_csv(phase_csv).to_excel(writer, sheet_name="Phase Transition", index=False)
    if os.path.exists(div_csv):
        pd.read_csv(div_csv).to_excel(writer, sheet_name="Secret Diversity", index=False)
    if os.path.exists(eff_csv):
        pd.read_csv(eff_csv).to_excel(writer, sheet_name="Sample Efficiency", index=False)
    
    # Summary of individual JSONs
    summary_data = []
    for f in os.listdir(RESULTS_DIR):
        if f.endswith('.json') and f != 'reproducibility.json':
            with open(os.path.join(RESULTS_DIR, f), 'r') as jf:
                try:
                    jd = json.load(jf)
                    summary_data.append({
                        "Experiment": f,
                        "AUROC": jd.get("model_auroc"),
                        "Advantage": jd.get("model_advantage")
                    })
                except:
                    pass
    if summary_data:
        pd.DataFrame(summary_data).to_excel(writer, sheet_name="JSON Summaries", index=False)
        
print(f"Successfully saved {excel_path}")

import platform
import datetime

metadata = {
    "python_version": platform.python_version(),
    "torch_version": torch.__version__,
    "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
    "gpu_model": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
    "date": datetime.datetime.now().isoformat()
}

try:
    git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode('utf-8')
    metadata["git_commit"] = git_hash
except Exception:
    metadata["git_commit"] = "Unknown"

metadata_path = f"{RESULTS_DIR}/reproducibility.json"
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=4)
    
print("Reproducibility metadata saved:")
for k, v in metadata.items():
    print(f"  {k}: {v}")
