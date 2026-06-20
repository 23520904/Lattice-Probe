import json

with open('notebook/Lattice_Probe_Experiments.ipynb', 'r', encoding='utf-8') as f:
    d = json.load(f)

for cell in d.get('cells', []):
    if cell.get('cell_type') == 'code':
        # Clear outputs
        cell['outputs'] = []
        cell['execution_count'] = None
        
        # Clean up source if it contains wrong \n
        new_source = []
        for line in cell.get('source', []):
            line = line.replace('\\n\n', '\n')
            new_source.append(line)
        cell['source'] = new_source

with open('notebook/Lattice_Probe_Experiments.ipynb', 'w', encoding='utf-8') as f:
    json.dump(d, f, indent=2)
