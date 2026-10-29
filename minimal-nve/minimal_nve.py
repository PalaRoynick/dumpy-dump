#!/usr/bin/env python3
"""Four-atom CC+CH+HH NVE test. No thermostat, no external reader required."""
import argparse
import csv
from pathlib import Path
import re
import numpy as np
from mdcraft.data import Atoms
from mdcraft.lattice import Domain
from mdcraft.neibs import VerletList
from mdcraft.tools import Threads
from mdcraft.solver import Multi, potential
from mdcraft.solver.boundary import Periodic
from mdcraft.solver.stepper import Verlet


def read_parameters(filename):
    if filename.suffix.lower() == '.npz':
        with np.load(filename, allow_pickle=False) as z:
            p = {k: z[k].copy() for k in ('Rcutoff', 'aVr', 'bVr', 'rVr', 'V', 'W', 'C', 'D', 'F', 'G')}
    else:
        p = {k: np.zeros(12) for k in 'VWCDFG'}
        found = set()
        for line in filename.read_text().splitlines():
            if not line.strip() or line.lstrip().startswith(('!', ';')):
                continue
            try:
                value = float(line.split()[0].replace('D', 'E').replace('d', 'e'))
            except ValueError:
                continue
            if '=' in line and line.split('=', 1)[1].strip().split()[0] in ('R1cut', 'Rcutoff'):
                p['Rcutoff'] = value
            if '!' not in line:
                continue
            tag = line.split('!', 1)[1].strip().split()[0]
            if tag in ('aVr', 'bVr', 'rVr'):
                p[tag] = value
            match = re.fullmatch(r'([VWCDFG])(\d{2})', tag)
            if match and 2 <= int(match[2]) <= 11:
                p[match[1]][int(match[2])] = value
                found.add(tag)
        expected = {f'{k}{i:02}' for k in 'VWCDFG' for i in range(2, 10 if k in 'VW' else 12)}
        if not expected <= found:
            raise ValueError(f'{filename}: missing labels {sorted(expected-found)}')
    for k in ('Rcutoff', 'aVr', 'bVr', 'rVr'):
        p[k] = float(p[k])
        if not np.isfinite(p[k]):
            raise ValueError(f'{filename}: invalid {k}')
    for k in 'VWCDFG':
        p[k] = np.ascontiguousarray(p[k], dtype=float)
        if p[k].ndim != 1 or not 12 <= p[k].size <= 32 or not np.isfinite(p[k]).all():
            raise ValueError(f'{filename}: invalid coefficient array {k}')
    if p['Rcutoff'] <= 0 or len({p[k].size for k in 'VWCDFG'}) != 1:
        raise ValueError(f'{filename}: invalid cutoff or unequal array lengths')
    return p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cc', 'ch', 'hh'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--like', choices=('modified', 'original'), default='modified')
    parser.add_argument('--mpi', action='store_true', help='exercise the MPI solver overload; exactly one rank')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--dt', type=float, default=0.0001, help='ps')
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--temperature', type=float, default=300., help='initial velocities only, K')
    parser.add_argument('--every', type=int, default=20, help='console interval; CSV records every step')
    parser.add_argument('--periodic', action='store_true')
    parser.add_argument('--csv', type=Path, default=Path('nve.csv'))
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    if args.dt <= 0 or args.steps < 0 or args.threads < 1 or args.every < 1 or args.temperature < 0:
        parser.error('Invalid timestep, steps, threads, output interval or temperature')
    comm = None
    if args.mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        if comm.size != 1:
            parser.error('This minimal test requires exactly one MPI rank')
    parameters = {k: read_parameters(getattr(args, k)) for k in ('cc', 'ch', 'hh')}
    classes = ('EAMWMOD1046', 'EAMWMOD0644') if args.like == 'modified' else ('EAMW1046', 'EAMW0644')
    pots = [getattr(potential, classes[0])(**parameters['cc']),
            potential.EAMMOD1044(**parameters['ch']),
            getattr(potential, classes[1])(**parameters['hh'])]
    threads = Threads(args.threads) if comm is None else Threads(args.threads, comm)
    domain = Domain(-1.5, 1.5, -1.5, 1.5, -1.5, 1.5)
    for axis in range(3):
        domain.set_periodic(axis, args.periodic)
    boundary = Periodic(domain)
    solver = Multi(boundary=boundary, nkinds=2, threads=threads)  # no thermostat
    for i, j, pot in ((0, 0, pots[0]), (0, 1, pots[1]), (1, 1, pots[2])):
        solver.add_potential(i, j, pot)
    atoms = Atoms(4)
    d = atoms.data
    d['uid'], d['kind'], d['tag'], d['m'] = [1, 2, 3, 4], [0, 0, 1, 1], 1, [12., 12., 1., 1.]
    d['rcut'] = max(p['Rcutoff'] for p in parameters.values())
    d['rns'] = 1.5*d['rcut']
    xyz = np.array([[-.0725, 0, 0], [.0725, 0, 0], [0, -.0775, .06239296355923341], [0, .0775, .06239296355923341]])
    mass = d['m'].copy()
    xyz -= np.sum(mass[:, None]*xyz, axis=0)/mass.sum()
    if args.periodic:
        xyz[:, 0] = (xyz[:, 0]+1.48+1.5) % 3-1.5
    velocity = np.random.default_rng(10291).normal(size=(4, 3))/np.sqrt(mass[:, None])
    velocity -= np.sum(mass[:, None]*velocity, axis=0)/mass.sum()
    velocity *= np.sqrt((3*4-3)*.008314462175*args.temperature/np.sum(mass[:, None]*velocity**2))
    for axis, name in enumerate('xyz'):
        d['r'][name], d['v'][name] = xyz[:, axis], velocity[:, axis]
    vd = None
    if comm is not None:
        from mdcraft.decomp import VD3
        vd = VD3(comm=comm, atoms=atoms, domain=domain, threads=threads, dimension=3, centers=[[0., 0., 0.]])
        vd.update(verbose=False)
        atoms = vd.locals
    local = VerletList(atoms=atoms, neighbors=atoms, domain=domain, threads=threads, half=False)
    halo = None if vd is None else VerletList(atoms=atoms, neighbors=vd.aliens, domain=domain, threads=threads, half=False)
    def update_lists():
        local.update()
        if halo is not None:
            halo.update()
    def inputs():
        return (atoms, atoms, local.get()) if vd is None else (vd, local.get(), halo.get())
    update_lists()
    solver.prepare(*inputs())
    solver.charge(*inputs())
    solver.forces(*inputs())  # initial full force before the first Verlet step
    stepper = Verlet(solver=solver, threads=threads)
    names = ['step', 't_ps', 'E_pair', 'E_embed', 'E_kin', 'E_total', 'delta_E', 'sum_Fx', 'sum_Fy', 'sum_Fz', 'norm_sum_F', 'norm_P']
    rows = []
    e0 = None
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    print(f'# thermostat=OFF; ranks=1; threads={threads.count}; dt={args.dt} ps; like={args.like}', flush=True)
    print('# ' + ' '.join(names), flush=True)
    with args.csv.open('w', newline='') as output:
        writer = csv.writer(output)
        writer.writerow(names)
        for step in range(args.steps+1):
            if step:
                # Rebuild at every step. The buffered list also covers the next position.
                update_lists()
                if vd is None:
                    stepper.make_step(atoms, local.get(), args.dt)
                else:
                    stepper.make_step(vd, local.get(), halo.get(), args.dt)
            # Only calculate energies here: keep the force used by the integrator.
            solver.virials(*inputs())
            d = atoms.data
            v = np.column_stack([d['v'][axis] for axis in 'xyz'])
            f = np.column_stack([d['f'][axis] for axis in 'xyz'])
            ep, em = float(d['Ep'].sum()), float(d['Em'].sum())
            ek = .5*float(np.sum(d['m'][:, None]*v**2))  # laboratory kinetic energy
            total = ep+em+ek
            if e0 is None:
                e0 = total
            sum_f = f.sum(axis=0)
            momentum = np.sum(d['m'][:, None]*v, axis=0)
            row = [step, step*args.dt, ep, em, ek, total, total-e0, *sum_f, np.linalg.norm(sum_f), np.linalg.norm(momentum)]
            writer.writerow(row)
            rows.append(row)
            if step % args.every == 0 or step == args.steps:
                print(' '.join(f'{x:.12e}' for x in row), flush=True)
            if not np.isfinite(row).all():
                raise RuntimeError(f'Nonfinite state at step {step}; inspect timestep and potential poles')
    data = np.asarray(rows)
    print(f'# max|delta_E|={np.max(np.abs(data[:, 6])):.8e}; max|sum_F|={np.max(data[:, 10]):.8e}', flush=True)
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
        for column in (2, 3, 4, 5):
            axes[0].plot(data[:, 1], data[:, column], label=names[column])
        axes[0].set_ylabel('Energy [kJ/mol]')
        axes[1].plot(data[:, 1], data[:, 6], label='E(t)-E(0)')
        axes[1].set_ylabel('Energy error [kJ/mol]')
        for column in (7, 8, 9, 10):
            axes[2].plot(data[:, 1], data[:, column], label=names[column])
        axes[2].set_ylabel('Force [kJ/(mol nm)]')
        axes[2].set_xlabel('Time [ps]')
        for ax in axes:
            ax.grid(alpha=.3)
            ax.legend()
        fig.tight_layout()
        fig.savefig(args.csv.with_suffix('.png'), dpi=160)
        fig.savefig(args.csv.with_suffix('.pdf'))
        plt.close(fig)


if __name__ == '__main__':
    main()
