from pathlib import Path
import ast
import contextlib
import io
import json
import os
import sys

import nbformat
import torch
from scipy.io import loadmat
from scipy.sparse import issparse

root = Path(__file__).resolve().parents[1]
os.chdir(root)
sys.path.insert(0, str(root))
torch.set_num_threads(1)
paths = sorted(root.glob('*_10^4.ipynb'))
report = []
for p in paths:
    nb = nbformat.read(p, as_version=4)
    nbformat.validate(nb)
    source = '\n'.join(c.source for c in nb.cells)
    assert all(name not in source for name in ['bcsstk09', 'west1505', 'bcspwr09', 'RUN_10_4'])
    ns = {'__name__': '__main__'}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        for c in nb.cells:
            if c.cell_type != 'code':
                continue
            ast.parse(c.source)
            if c.source.startswith('MATRIX ='):
                if p.name == 'Gaussian_elimination_10^4.ipynb':
                    exec(c.source, ns)
                    path = ns['path']
                else:
                    exec(c.source.split('A_raw =')[0], ns)
                    path = ns['data_path'](ns['MATRIX'], ns['MAT_PATH'])
                    assert ns['data_path']('bcsstk17', path) == path
                assert ns['MATRIX'] == 'bcsstk17'
                A = loadmat(path)['Problem'][0, 0]['A']
                assert issparse(A) and A.shape == (10974, 10974)
                print('Verified bcsstk17 path:', path, 'shape:', A.shape, 'nnz:', A.nnz)
                break
            exec(compile(c.source, str(p), 'exec'), ns)
    report.append({'notebook': p.name, 'status': 'passed',
                   'scope': 'format/syntax, 3x3 execution, bcsstk17 discovery and sparse load; full 10^4 algorithms not rerun',
                   'output': output.getvalue()})
    print(p.name, 'passed', flush=True)
(root / 'validation_kaggle_setup.json').write_text(json.dumps(report, indent=2) + '\n')
