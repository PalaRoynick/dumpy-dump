#!/usr/bin/env python3
"""Run multiple local coefficient sets, then overlay their six comparison curves."""
import argparse
import csv
from pathlib import Path
import subprocess
import sys
import numpy as np
from mod_model import CLASSES

ROOT = Path(__file__).resolve().parent
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--case', nargs=3, action='append', required=True, metavar=('LABEL', 'CLASS', 'PARAMS'))
p.add_argument('--reference', type=Path, default=ROOT/'reference')
p.add_argument('--out', type=Path, default=Path('comparison-many'))
p.add_argument('--backend', choices=('mdcraft', 'formula'), default='mdcraft')
p.add_argument('--density-column', choices=('r-dn-dr', 'dn-dr'), default='r-dn-dr')
p.add_argument('--symlog', action='store_true')
p.add_argument('--rtol', type=float, default=1e-3)
p.add_argument('--atol', type=float, default=1e-6)
a = p.parse_args()
labels = [item[0] for item in a.case]
if len(labels) != len(set(labels)):
    p.error('Case labels must be unique')
if any(not label or label in ('.', '..') or '/' in label or '\\' in label for label in labels):
    p.error('Labels must be simple directory names')
if any(cls not in CLASSES for _, cls, _ in a.case):
    p.error('Unknown MOD class')
a.out.mkdir(parents=True, exist_ok=True)
failed = False
for label, cls, params in a.case:
    command = [sys.executable, str(ROOT/'compare_mod.py'), '--params', params,
               '--class', cls, '--backend', a.backend, '--reference', str(a.reference),
               '--out', str(a.out/label), '--density-column', a.density_column,
               '--rtol', str(a.rtol), '--atol', str(a.atol)]
    if a.symlog:
        command.append('--symlog')
    result = subprocess.run(command)
    if result.returncode not in (0, 2):
        sys.exit(result.returncode)
    failed |= result.returncode != 0
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, axes = plt.subplots(3, 2, figsize=(13, 12))
for row, group in enumerate(('pair', 'density', 'embedding')):
    for i, label in enumerate(labels):
        path = a.out/label/(group+'_comparison.csv')
        with path.open() as f:
            header = next(csv.reader(f))
        data = np.loadtxt(path, delimiter=',', skiprows=1)
        for col in range(2):
            offset = 1+4*col
            ax = axes[row, col]
            if i == 0:
                ax.plot(data[:, 0], data[:, offset], color='black', label='Reference', lw=2)
                ax.set_title(group+'/'+header[offset].removesuffix('_reference'))
            ax.plot(data[:, 0], data[:, offset+1], '--', label=label)
            ax.grid(alpha=.3)
            ax.set_xlabel('n' if group == 'embedding' else 'r [nm]')
            if a.symlog:
                ax.set_yscale('symlog', linthresh=1e-4)
            ax.legend()
fig.tight_layout()
fig.savefig(a.out/'overlay.png', dpi=180)
fig.savefig(a.out/'overlay.pdf')
plt.close(fig)
sys.exit(2 if failed else 0)
