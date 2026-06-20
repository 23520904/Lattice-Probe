import json

with open('notebook/Lattice_Probe_Experiments.ipynb', 'r', encoding='utf-8') as f:
    d = json.load(f)

new_source = [
    'import os\n',
    'import subprocess\n',
    '\n',
    '# Ensure we are in a valid directory before cloning\n',
    'if os.path.exists("/content"):\n',
    '    os.chdir("/content")\n',
    '\n',
    'REPO_URL = "https://github.com/23520904/Lattice-Probe.git"\n',
    'REPO_DIR = "/content/Lattice-Probe"\n',
    '\n',
    'if not os.path.exists(REPO_DIR):\n',
    '    print(f"Cloning repository from {REPO_URL}...")\n',
    '    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)\n',
    'else:\n',
    '    print("Repository already exists. Skipping clone.")\n',
    '\n',
    'os.chdir(REPO_DIR)\n',
    'print(f"Current working directory: {os.getcwd()}")\n'
]

d['cells'][4]['source'] = new_source

with open('notebook/Lattice_Probe_Experiments.ipynb', 'w', encoding='utf-8') as f:
    json.dump(d, f, indent=2)
