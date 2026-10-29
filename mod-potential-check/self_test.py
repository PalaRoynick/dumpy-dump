#!/usr/bin/env python3
"""Test scripts using synthetic public coefficients, not a fitted force field."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
from mod_model import CLASSES, Formula

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--backend', choices=('formula', 'mdcraft'), default='formula')
a = p.parse_args()
root = Path(__file__).resolve().parent
with tempfile.TemporaryDirectory(prefix='mod-check-selftest-') as directory:
    work = Path(directory)
    params = {k: np.zeros(12) for k in 'VWCDFG'}
    params.update(Rcutoff=1., aVr=.002, bVr=.03, rVr=.04)
    params['V'][2:5] = [.2, -.03, .004]
    params['W'][2] = .02
    params['C'][2:5] = [.4, .03, .004]
    params['D'][2] = .01
    params['F'][2:5] = [.1, .02, .001]
    params['G'][2] = .01
    np.savez(work/'public-synthetic.npz', **params)
    for cls in CLASSES:
        f = Formula(cls, params)
        reference = work/cls
        reference.mkdir()
        r = np.linspace(.00011, 1., 81)
        v, dv, _, _ = f.radial(r)
        np.savetxt(reference/'V_Zha.dat', np.c_[r, r*v, dv], header='r rV rVirX', comments='')
        r = np.linspace(0., 1., 81)
        _, _, rho, drho = f.radial(r)
        np.savetxt(reference/'density_Zha.dat', np.c_[r, rho, r*drho], header='r n rdn/dr', comments='')
        n = np.linspace(-1., 3.8, 81)
        emb, df = f.embedding(n)
        np.savetxt(reference/'Femb_Zha.dat', np.c_[n, emb, df], header='n F dF/dn', comments='')
        subprocess.run([sys.executable, str(root/'compare_mod.py'), '--params', str(work/'public-synthetic.npz'),
                        '--class', cls, '--backend', a.backend, '--reference', str(reference),
                        '--out', str(work/(cls+'-result')), '--no-plots', '--rtol', '1e-7', '--atol', '1e-9'], check=True)
        report = json.loads((work/(cls+'-result')/'report.json').read_text())
        assert report['all_reference_pass'] and report['all_derivative_checks_pass']
print('PASS: all three classes; six reference columns; analytic/finite-difference derivatives; weighted density convention')
