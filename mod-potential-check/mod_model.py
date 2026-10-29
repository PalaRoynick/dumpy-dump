"""MOD curves and access to the installed MDcraft implementation. No coefficients bundled."""
import json
from pathlib import Path
import numpy as np

CLASSES = ('EAMWMOD1046', 'EAMWMOD0644', 'EAMMOD1044')


def load_parameters(filename):
    path = Path(filename)
    if path.suffix.lower() == '.npz':
        with np.load(path, allow_pickle=False) as data:
            p = {k: data[k].copy() for k in data.files}
    else:
        p = json.loads(path.read_text())
    required = set('VWCDFG') | {'Rcutoff', 'aVr', 'bVr', 'rVr'}
    missing = required - p.keys()
    if missing:
        raise ValueError(f'Missing parameter keys: {sorted(missing)}')
    p = {k: p[k] for k in required}
    for k in ('Rcutoff', 'aVr', 'bVr', 'rVr'):
        p[k] = float(p[k])
        if not np.isfinite(p[k]):
            raise ValueError(f'{k} must be finite')
    if p['Rcutoff'] <= 0:
        raise ValueError('Rcutoff must be positive (nm)')
    for k in 'VWCDFG':
        p[k] = np.ascontiguousarray(p[k], dtype=np.float64)
        if p[k].ndim != 1 or not 12 <= p[k].size <= 32 or not np.isfinite(p[k]).all():
            raise ValueError(f'{k}: expected 12..32 finite coefficients, including slots 0 and 1')
    if len({len(p[k]) for k in 'VWCDFG'}) != 1:
        raise ValueError('All coefficient arrays must have equal length')
    return p


class Formula:
    """Independent analytic evaluation; does not execute the C++ implementation."""
    def __init__(self, name, parameters):
        self.name, self.p = name, parameters
        self.rcut = parameters['Rcutoff']

    @staticmethod
    def polynomial(x, a, start, stop, offset):
        value, derivative = np.zeros_like(x), np.zeros_like(x)
        for i in range(start, stop + 1):
            power = i + offset
            value += a[i] * x ** power
            derivative += power * a[i] * x ** (power - 1)
        return value, derivative

    def radial(self, r):
        r = np.asarray(r, dtype=float)
        x = np.maximum(self.rcut ** 2 - r ** 2, 0.)
        p = self.p
        v, vx = self.polynomial(x, p['V'], 2, 9, 1)
        w, wx = self.polynomial(x, p['W'], 2, 9, -1)
        c, cx = self.polynomial(x, p['C'], 2, 11, 1)
        d, dx = self.polynomial(x, p['D'], 2, 11, -1)
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            pair = v / (1 + w)
            dpair = -2 * r * (vx * (1 + w) - v * wx) / (1 + w) ** 2
            if p['aVr'] != 0:
                # WMOD1046: K=6,M=10; WMOD0644: K=4,M=6; MOD1044: K=4,M=10.
                # All three use L=4.
                k = 6 if self.name == 'EAMWMOD1046' else 4
                m = 6 if self.name == 'EAMWMOD0644' else 10
                s = p['bVr'] * r ** (2*k) + x ** 4
                ds = 2*k*p['bVr']*r**(2*k-1) - 8*r*x**3
                q = 1/r**2 - p['rVr']
                pair += p['aVr'] * q * s * x**m
                dpair += p['aVr'] * (-2/r**3*s*x**m + q*ds*x**m - 2*m*r*q*s*x**(m-1))
            density = c / (1 + d)
            ddensity = -2*r*(cx*(1+d)-c*dx)/(1+d)**2
        outside = r >= self.rcut
        return tuple(np.where(outside, 0., a) for a in (pair, dpair, density, ddensity))

    def embedding(self, n):
        n = np.asarray(n, dtype=float)
        cross = self.name == 'EAMMOD1044'
        x = n if cross else n - 1
        f, df, g, dg = (np.zeros_like(x) for _ in range(4))
        for i in range(2, 12):
            basis = x**i if cross else x**i - (-1.)**i
            f += self.p['F'][i]*basis
            g += self.p['G'][i]*basis
            df += self.p['F'][i]*i*x**(i-1)
            dg += self.p['G'][i]*i*x**(i-1)
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            return f/(1+g), (df*(1+g)-f*dg)/(1+g)**2


class TwoAtoms:
    """Nonperiodic, one-thread probe; never calls an MPI communicator or a thermostat."""
    def __init__(self, name, p, pot):
        from mdcraft.data import Atoms
        from mdcraft.lattice import Domain
        from mdcraft.neibs import VerletList
        from mdcraft.solver import Multi
        from mdcraft.solver.boundary import Periodic
        from mdcraft.tools import Threads
        self.cross = name == 'EAMMOD1044'
        self.threads = Threads(1)
        self.atoms = Atoms(2)
        a = self.atoms.data
        for field in ('r', 'v', 'f'):
            for axis in 'xyz':
                a[field][axis] = 0
        a['kind'] = [0, 1] if self.cross else [0, 0]
        a['uid'] = [1, 2]
        a['m'], a['rcut'], a['rns'] = 1., p['Rcutoff'], p['Rcutoff']*1.01
        a['tag'] = 1
        length = max(1., p['Rcutoff']*4)
        self.domain = Domain(-length, length, -length, length, -length, length)
        for axis in range(3):
            self.domain.set_periodic(axis, False)
        self.boundary = Periodic(self.domain)
        self.solver = Multi(boundary=self.boundary, nkinds=2, threads=self.threads)
        self.pot = pot
        self.solver.add_potential(0, int(self.cross), pot)
        self.nlist = VerletList(atoms=self.atoms, neighbors=self.atoms,
                                domain=self.domain, threads=self.threads, half=False)

    def prepare(self, r):
        self.atoms.data['r']['x'] = [0., r]
        self.nlist.update()
        self.solver.prepare(self.atoms, self.atoms, self.nlist.get())

    def density(self, r):
        self.prepare(r)
        self.solver.charge(self.atoms, self.atoms, self.nlist.get())
        return float(self.atoms.data['nc_mod'][int(self.cross), 0])

    def force(self, r, density=None):
        self.prepare(r)
        self.solver.charge(self.atoms, self.atoms, self.nlist.get())
        if density is not None:
            self.atoms.data['nc_mod'][:, 0] = density
        self.solver.forces(self.atoms, self.atoms, self.nlist.get())
        return float(self.atoms.data['f']['x'][0])


class Native:
    """C++ values and analytic derivatives recovered through force probes.

    The probes isolate pair and embedding terms using copies of the local
    coefficients. The user's potential object and parameter file are unmodified.
    """
    def __init__(self, name, p):
        from mdcraft.solver import potential
        self.name, self.p, self.rcut = name, p, p['Rcutoff']
        self.cls = getattr(potential, name)
        self.pot = self.cls(**p)
        self.formula = Formula(name, p)
        self.cross = name == 'EAMMOD1044'
        self.density_probe = TwoAtoms(name, p, self.pot)
        zero = np.zeros_like(p['F'])
        pair_p = dict(p, F=zero, G=zero)
        self.pair_probe = TwoAtoms(name, pair_p, self.cls(**pair_p))
        # F_probe(n) = (n-1)^2/2 - 1/2 for like pairs; n^2/2 for cross.
        f = zero.copy()
        f[2] = .5
        rho_p = dict(p, V=zero, W=zero, aVr=0., F=f, G=zero)
        self.rho_probe = TwoAtoms(name, rho_p, self.cls(**rho_p))
        emb_p = dict(p, V=zero, W=zero, aVr=0.)
        self.emb_probe = TwoAtoms(name, emb_p, self.cls(**emb_p))
        self.metadata = {'backend': 'mdcraft', 'class': name,
                         'module': str(Path(potential.__file__).resolve()),
                         'pair_scalar_multiplier': 2 if self.cross else 1,
                         'density_endpoint': 'r < 1e-4 nm: independent analytic limit (C++ skips these pairs)'}
        self.anchor = None

    def pair_value(self, r):
        factor = 2 if self.cross else 1
        return np.array([factor*self.pot.value(float(t)) if 0 < t < self.rcut else
                         (0. if t >= self.rcut else np.nan) for t in r])

    def radial(self, r):
        r = np.asarray(r, dtype=float)
        pair = self.pair_value(r)
        dpair, rho, drho = (np.full_like(r, np.nan) for _ in range(3))
        for i, t in enumerate(r):
            if t >= self.rcut:
                dpair[i] = rho[i] = drho[i] = 0.
            elif t < 1e-4:
                # C++ atom-level kernels skip r^2 < 1e-8. Do not report zero
                # as the mathematical radial density at the origin.
                _, _, rn, rd = self.formula.radial(np.array([t]))
                rho[i], drho[i] = rn[0], rd[0]
            else:
                rho[i] = self.density_probe.density(float(t))
                dpair[i] = self.pair_probe.force(float(t))
                imposed = 2 - rho[i] if self.cross else 2.
                drho[i] = self.rho_probe.force(float(t), imposed)/2
        return pair, dpair, rho, drho

    def _anchor(self):
        r = np.linspace(max(1.1e-4, self.rcut*.03), self.rcut*.95, 100)
        _, _, _, derivative = self.formula.radial(r)
        finite = np.isfinite(derivative)
        if not finite.any() or np.max(np.abs(derivative[finite])) == 0:
            raise ValueError('Cannot recover C++ dF/dn: density is constant. Use --backend formula or a custom adapter.')
        r0 = float(r[np.flatnonzero(finite)[np.argmax(np.abs(derivative[finite]))]])
        rho = self.density_probe.density(r0)
        imposed = 2-rho if self.cross else 2.
        derivative = self.rho_probe.force(r0, imposed)/2
        if not np.isfinite(derivative) or derivative == 0:
            raise ValueError('Degenerate native density derivative probe')
        base = self.emb_probe.force(r0, 0.) if self.cross else 0.
        self.anchor = r0, derivative, base

    def embedding(self, n):
        n = np.asarray(n, dtype=float)
        if self.cross:
            value = np.array([self.pot.embeddingEnergy(float(t)) for t in n])
        else:
            value = np.asarray(self.pot.embeddingEnergy(np.ascontiguousarray(n)))
        if self.anchor is None:
            self._anchor()
        r0, drho, base = self.anchor
        divisor = drho if self.cross else 2*drho
        deriv = np.array([(self.emb_probe.force(r0, float(t))-base)/divisor for t in n])
        return value, deriv
