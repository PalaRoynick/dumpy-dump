#!/usr/bin/env python3
"""Compare local MOD parameters with the supplied Zha tables; no MPI launcher required."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
from mod_model import CLASSES, Formula, Native, load_parameters

ROOT = Path(__file__).resolve().parent


def references(folder):
    result = {}
    for key, name in (('pair', 'V_Zha.dat'), ('density', 'density_Zha.dat'), ('embedding', 'Femb_Zha.dat')):
        data = np.loadtxt(folder/name, skiprows=1)
        if data.ndim != 2 or data.shape[1] != 3 or len(data) < 5:
            raise ValueError(f'{name}: expected three columns and at least five rows')
        if not np.isfinite(data).all() or not np.all(np.diff(data[:, 0]) > 0):
            raise ValueError(f'{name}: nonfinite values or unsorted/duplicate abscissas')
        result[key] = data
    if (result['pair'][:, 0] <= 0).any() or (result['density'][:, 0] < 0).any():
        raise ValueError('Pair distances must be positive; density distances nonnegative')
    return result


def metrics(reference, calculated, rtol, atol):
    reference, calculated = np.asarray(reference), np.asarray(calculated)
    valid = np.isfinite(reference) & np.isfinite(calculated)
    delta = calculated[valid] - reference[valid]
    nonzero = valid & (np.abs(reference) > max(atol, np.max(np.abs(reference))*1e-12))
    passed = valid & (np.abs(calculated-reference) <= atol + rtol*np.abs(reference))
    def number(value):
        value = float(value)
        return value if np.isfinite(value) else None
    return dict(points=len(reference), nonfinite=int((~valid).sum()),
                failed=int((~passed).sum()), passed=bool(passed.all()),
                max_abs=number(np.max(np.abs(delta))) if len(delta) else None,
                rmse=number(np.sqrt(np.mean(delta**2))) if len(delta) else None,
                relative_l2=number(np.linalg.norm(delta)/max(np.linalg.norm(reference[valid]), 1e-300)),
                max_relative_away_from_zero=number(np.max(np.abs((calculated[nonzero]-reference[nonzero])/reference[nonzero]))) if nonzero.any() else None)


def reference_audit(ref):
    """Five-point numerical derivatives, excluding 5% at either edge.

    This is a data-consistency diagnostic, never an automatic convention switch.
    """
    result = {}
    for key, data in ref.items():
        x, y, derivative_column = data.T
        if key == 'pair':
            y = y/x  # The stored energy column is r*V, not V.
        h = np.mean(np.diff(x))
        if not np.allclose(np.diff(x), h, rtol=1e-6, atol=1e-12):
            raise ValueError('Reference audit expects the supplied uniform grids')
        derivative = (y[:-4]-8*y[1:-3]+8*y[3:-1]-y[4:])/(12*h)
        interior = x[2:-2]
        keep = (interior > x[0]+.05*(x[-1]-x[0])) & (interior < x[-1]-.05*(x[-1]-x[0]))
        target = derivative_column[2:-2][keep]
        scale = max(np.linalg.norm(target), 1e-300)
        result[key] = {label: float(np.linalg.norm(factor*derivative[keep]-target)/scale)
                       for label, factor in (('d_function_dx', 1.), ('x_times_d_function_dx', interior[keep]))}
    return result


def derivative_check(model, ref):
    """Native force derivatives versus numerical derivatives of native values."""
    r = np.linspace(max(2e-4, model.rcut*.03), model.rcut*.95, 33)
    h = np.minimum.reduce([r*.02, (model.rcut-r)*.02, np.full_like(r, model.rcut*1e-4)])
    if (h <= 0).any():
        raise ValueError('Rcutoff too small for native derivative probes')
    p, dp, rho, drho = model.radial(r)
    def radial_values(t):
        if isinstance(model, Native):
            return model.pair_value(t), np.array([model.density_probe.density(float(v)) for v in t])
        a, _, b, _ = model.radial(t)
        return a, b
    values = [radial_values(r+offset*h) for offset in (-2, -1, 1, 2)]
    fd_pair = (values[0][0]-8*values[1][0]+8*values[2][0]-values[3][0])/(12*h)
    fd_rho = (values[0][1]-8*values[1][1]+8*values[2][1]-values[3][1])/(12*h)
    n = np.linspace(ref['embedding'][0, 0], ref['embedding'][-1, 0], 33)
    hn = 2e-5*np.maximum(1., np.abs(n))
    _, df = model.embedding(n)
    vals = [model.embedding(n+offset*hn)[0] for offset in (-2, -1, 1, 2)]
    fd_emb = (vals[0]-8*vals[1]+8*vals[2]-vals[3])/(12*hn)
    return {key: metrics(fd, exact, 5e-5, 1e-7) for key, fd, exact in
            (('dV_dr', fd_pair, dp), ('dn_dr', fd_rho, drho), ('dF_dn', fd_emb, df))}


def save_comparison(path, x, curves, rtol, atol):
    summary = {}
    headers, columns = ['x'], [x]
    for key, (ref, calc, _) in curves.items():
        error = calc-ref
        relative = np.full_like(calc, np.nan)
        mask = np.abs(ref) > max(atol, np.max(np.abs(ref))*1e-12)
        relative[mask] = error[mask]/np.abs(ref[mask])
        headers.extend([key+'_reference', key+'_calculated', key+'_error', key+'_relative_error'])
        columns.extend([ref, calc, error, relative])
        summary[key] = metrics(ref, calc, rtol, atol)
    np.savetxt(path, np.column_stack(columns), delimiter=',', header=','.join(headers), comments='')
    return summary


def plot_comparison(pdf, out, key, x, curves, args):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(curves), 3, figsize=(16, 4*len(curves)), squeeze=False)
    for row, (name, (ref, calc, label)) in enumerate(curves.items()):
        a, b, c = axes[row]
        a.plot(x, ref, label='Reference', lw=1.7)
        a.plot(x, calc, '--', label=args.potential_class, lw=1.2)
        a.set_ylabel(label)
        a.legend()
        b.plot(x, calc-ref)
        b.set_ylabel('Calculated - reference')
        mask = np.abs(ref) > max(args.atol, np.max(np.abs(ref))*1e-12)
        relative = np.full_like(calc, np.nan)
        relative[mask] = (calc[mask]-ref[mask])/np.abs(ref[mask])
        c.plot(x, relative)
        c.set_ylabel('Error / |reference| (near-zero excluded)')
        for ax in (a, b, c):
            ax.grid(alpha=.3)
            ax.set_xlabel('n' if key == 'embedding' else 'r [nm]')
            if args.symlog:
                ax.set_yscale('symlog', linthresh=args.linthresh)
            limits = (args.n_min, args.n_max) if key == 'embedding' else (args.r_min, args.r_max)
            if any(v is not None for v in limits):
                ax.set_xlim(*limits)
    fig.suptitle(key+' — full grid metrics in report.json; axis limits only affect display')
    fig.tight_layout()
    fig.savefig(out/(key+'.png'), dpi=160)
    pdf.savefig(fig)
    plt.close(fig)


def build_curves(model, ref, args):
    result, extras = {}, {}
    r, rv_ref, vir_ref = ref['pair'].T
    v, dv, _, _ = model.radial(r)
    result['pair'] = dict(rV=(rv_ref, r*v, 'r V(r) [nm kJ/mol]'),
                          rVirX=(vir_ref, dv, 'r VirX = dV/dr [kJ/mol/nm]'))
    extras['pair'] = (r, dict(V=v, dV_dr=dv, rV=r*v, r_dV_dr=r*dv,
                              dV_dr_over_r=dv/r, d_rV_dr=v+r*dv, V_dV_dr=v*dv))
    r, rho_ref, rho_der_ref = ref['density'].T
    _, _, rho, drho = model.radial(r)
    weighted = r*drho if args.density_column == 'r-dn-dr' else drho
    result['density'] = dict(n=(rho_ref, rho, 'n(r)'),
                             derivative=(rho_der_ref, weighted,
                                         'r dn/dr' if args.density_column == 'r-dn-dr' else 'dn/dr [1/nm]'))
    over_r = np.full_like(r, np.nan)
    np.divide(drho, r, out=over_r, where=r != 0)
    extras['density'] = (r, dict(n=rho, dn_dr=drho, r_dn_dr=r*drho,
                                 dn_dr_over_r=over_r, n_dn_dr=rho*drho))
    n, f_ref, df_ref = ref['embedding'].T
    f, df = model.embedding(n)
    result['embedding'] = dict(F=(f_ref, f, 'F(n) [kJ/mol]'),
                               dF_dn=(df_ref, df, 'dF/dn [kJ/mol]'))
    extras['embedding'] = (n, dict(F=f, dF_dn=df, n_dF_dn=n*df, F_dF_dn=f*df))
    return result, extras


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--params', type=Path, help='local JSON/NPZ file; never copied to the results')
    p.add_argument('--class', dest='potential_class', choices=CLASSES, default='EAMMOD1044')
    p.add_argument('--backend', choices=('mdcraft', 'formula'), default='mdcraft')
    p.add_argument('--reference', type=Path, default=ROOT/'reference')
    p.add_argument('--out', type=Path, default=Path('comparison'))
    p.add_argument('--density-column', choices=('r-dn-dr', 'dn-dr'), default='r-dn-dr')
    p.add_argument('--rtol', type=float, default=1e-3)
    p.add_argument('--atol', type=float, default=1e-6)
    p.add_argument('--audit-only', action='store_true', help='inspect reference column consistency; no parameters/MDcraft needed')
    p.add_argument('--symlog', action='store_true', help='symmetric log Y axes (keeps negative values)')
    p.add_argument('--linthresh', type=float, default=1e-4)
    p.add_argument('--r-min', type=float)
    p.add_argument('--r-max', type=float)
    p.add_argument('--n-min', type=float)
    p.add_argument('--n-max', type=float)
    p.add_argument('--no-plots', action='store_true')
    args = p.parse_args(argv)
    if args.rtol < 0 or args.atol < 0 or args.linthresh <= 0:
        p.error('rtol/atol must be nonnegative; linthresh must be positive')
    ref = references(args.reference)
    args.out.mkdir(parents=True, exist_ok=True)
    audit = reference_audit(ref)
    (args.out/'reference_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    print('Reference audit (relative L2 errors; not a convention switch):')
    print(json.dumps(audit, indent=2))
    if args.audit_only:
        return 0
    if args.params is None:
        p.error('--params is required except with --audit-only')
    params = load_parameters(args.params)
    if args.backend == 'mdcraft' and params['Rcutoff'] <= 2e-4:
        p.error('Native atom-level probes require Rcutoff > 2e-4 nm')
    model = Native(args.potential_class, params) if args.backend == 'mdcraft' else Formula(args.potential_class, params)
    print(f'Computing {args.potential_class} ({args.backend}); density reference = {args.density_column}', flush=True)
    curves, extras = build_curves(model, ref, args)
    report = {'class': args.potential_class, 'backend': args.backend,
              'density_column': args.density_column, 'pair_column_2': 'r*V(r)',
              'pair_column_3': 'r*VirX = dV/dr', 'energy_units': 'kJ/mol', 'distance_units': 'nm',
              'rtol': args.rtol, 'atol': args.atol, 'reference_audit': audit,
              'metadata': getattr(model, 'metadata', {'backend': 'independent Python formulas; not a native-code test'}),
              'reference_comparison': {}, 'derivative_consistency': derivative_check(model, ref)}
    for key, series in curves.items():
        report['reference_comparison'][key] = save_comparison(args.out/(key+'_comparison.csv'),
                                                             ref[key][:, 0], series, args.rtol, args.atol)
    for key, (x, series) in extras.items():
        np.savetxt(args.out/(key+'_all_quantities.csv'), np.column_stack([x, *series.values()]),
                   delimiter=',', header=','.join(['x', *series]), comments='')
    all_reference_pass = all(m['passed'] for part in report['reference_comparison'].values() for m in part.values())
    all_derivative_pass = all(m['passed'] for m in report['derivative_consistency'].values())
    report.update(all_reference_pass=all_reference_pass, all_derivative_checks_pass=all_derivative_pass)
    (args.out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    with (args.out/'summary.csv').open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['group', 'quantity', 'passed', 'failed_points', 'max_abs', 'rmse', 'relative_l2'])
        for group, quantities in report['reference_comparison'].items():
            for quantity, m in quantities.items():
                writer.writerow([group, quantity, m['passed'], m['failed'], m['max_abs'], m['rmse'], m['relative_l2']])
                print(f'{group}/{quantity}: {"PASS" if m["passed"] else "FAIL"}, relative L2={m["relative_l2"]}, failed={m["failed"]}')
    if not args.no_plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(args.out/'comparison.pdf') as pdf:
            for key, series in curves.items():
                plot_comparison(pdf, args.out, key, ref[key][:, 0], series, args)
            for key, (x, series) in extras.items():
                fig, axes = plt.subplots((len(series)+1)//2, 2, figsize=(12, 3*((len(series)+1)//2)), squeeze=False)
                for ax, (name, values) in zip(axes.flat, series.items()):
                    ax.plot(x, values)
                    ax.set_title(name)
                    ax.set_xlabel('n' if key == 'embedding' else 'r [nm]')
                    ax.grid(alpha=.3)
                    if args.symlog:
                        ax.set_yscale('symlog', linthresh=args.linthresh)
                    limits = (args.n_min, args.n_max) if key == 'embedding' else (args.r_min, args.r_max)
                    if any(v is not None for v in limits):
                        ax.set_xlim(*limits)
                for ax in list(axes.flat)[len(series):]:
                    ax.set_visible(False)
                fig.suptitle(key+' — derived quantities (calculated; no extra reference columns)')
                fig.tight_layout()
                fig.savefig(args.out/(key+'_all_quantities.png'), dpi=160)
                pdf.savefig(fig)
                plt.close(fig)
    print(f'Results: {args.out.resolve()}')
    print('Derivative consistency:', 'PASS' if all_derivative_pass else 'FAIL')
    return 0 if all_reference_pass and all_derivative_pass else 2


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, ImportError, OSError, FloatingPointError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
