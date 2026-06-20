import json

def patch_notebook():
    path = 'notebook/Lattice_Probe_Experiments.ipynb'
    with open(path, 'r', encoding='utf-8') as f:
        d = json.load(f)
    
    for c in d.get('cells', []):
        if c.get('cell_type') == 'code':
            s = ''.join(c.get('source', []))
            if 'subprocess.run([' in s and 'check=True' in s and '--epochs' in s:
                c['source'] = [
                    'import subprocess\n',
                    '\n',
                    'result = subprocess.run([\n',
                    '    "python", "scripts/train.py",\n',
                    '    "--param-set", PARAM_SET,\n',
                    '    "--model", MODEL,\n',
                    '    "--train-dir", train_dir,\n',
                    '    "--val-dir", val_dir,\n',
                    '    "--output-dir", ckpt_out,\n',
                    '    "--epochs", "20",\n',
                    '    "--compute-log", f"{ckpt_out}/compute_log.csv"\n',
                    '], capture_output=True, text=True)\n',
                    '\n',
                    'print("Return code:", result.returncode)\n',
                    'print("\\nSTDOUT:")\n',
                    'print(result.stdout)\n',
                    'print("\\nSTDERR:")\n',
                    'print(result.stderr)\n'
                ]
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(d, f, indent=1)

if __name__ == '__main__':
    patch_notebook()
