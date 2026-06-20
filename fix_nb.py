import json

with open('notebook/Lattice_Probe_Experiments.ipynb', 'r', encoding='utf-8') as f:
    d = json.load(f)

d['cells'][3]['source'][4] = 'REPO_DIR = "/content/Lattice-Probe"\\n'
d['cells'][3]['source'][8] = '    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)\\n'

with open('notebook/Lattice_Probe_Experiments.ipynb', 'w', encoding='utf-8') as f:
    json.dump(d, f, indent=2)
