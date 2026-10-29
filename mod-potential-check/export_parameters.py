#!/usr/bin/env python3
"""Convert a local eamReader input to NPZ without printing any coefficients."""
import argparse
import importlib.util
import sys
from pathlib import Path
import numpy as np
from mod_model import load_parameters

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--reader', type=Path, required=True, help='path to your existing eamReader.py')
p.add_argument('--input', type=Path, required=True, help='local EAMW1044.dat or other coefficient file')
p.add_argument('--out', type=Path, required=True, help='new local .npz file (keep it private)')
a = p.parse_args()
if a.out.suffix.lower() != '.npz':
    p.error('--out must end in .npz')
if a.out.exists():
    p.error('Output already exists; choose a new filename')
sys.path.insert(0, str(a.reader.resolve().parent))
spec = importlib.util.spec_from_file_location('local_eam_reader', a.reader.resolve())
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
values = module.eamReader(str(a.input))
keys = ('Rcutoff', 'aVr', 'bVr', 'rVr', 'V', 'W', 'C', 'D', 'F', 'G')
if len(values) != len(keys):
    raise ValueError('eamReader must return (rcut, aVr, bVr, rVr, V, W, C, D, F, G)')
with a.out.open('xb') as output:
    np.savez(output, **dict(zip(keys, values)))
load_parameters(a.out)
print('Local parameter file written and validated; coefficients were not printed.')
