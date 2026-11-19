''' Bunya sweep over (kx, kz, c+) for all walls: one script, one input, one output.
    usage:  python main/bunya_sweep.py input/sweep_v4_in.npz output/sweep_v4_out.npz

    Primitive-variable resolvent for channel flow with a compliant-wall admittance BC.
    Based on Luhar, Sharma & McKeon (2014) JFM 749:597-626 (M. Luhar 2015, R. McMullen 2017,
    L. Cappello 2026). Mean flow from the Reynolds-Tiederman eddy viscosity.               '''

import os
for v in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[v] = '1'          # one BLAS thread per worker (Slurm may set these to all cores)

import sys
import time
import platform
from functools import lru_cache
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import scipy.linalg
from scipy.linalg import toeplitz

SCALARS = ['s1', 's2', 'RS', 'RS_all', 'eta', 'eta_all', 'p_wall', 'u_max', 'v_max']
PROFILES = ['uu', 'vv', 'ww', 'uv', 'pp']
SAVE_PROFILES = True             # P ~3.3 GB for v4; the NNLS fit needs the rigid uv profiles


# ============================== numerics ==============================

def clencurt(N):
    ''' Clenshaw-Curtis quadrature weights on N+1 Chebyshev points '''
    theta = np.pi * np.arange(N + 1) / N
    x = np.cos(theta)
    w = np.zeros(N + 1)
    ii = np.arange(1, N)
    v = np.ones(N - 1)
    if N % 2 == 0:
        w[0] = w[-1] = 1 / (N**2 - 1)
        for k in range(1, N // 2):
            v = v - 2 * np.cos(2 * k * theta[ii]) / (4 * k**2 - 1)
        v = v - np.cos(N * theta[ii]) / (N**2 - 1)
    else:
        w[0] = w[-1] = 1 / N**2
        for k in range(1, (N - 1) // 2 + 1):
            v = v - 2 * np.cos(2 * k * theta[ii]) / (4 * k**2 - 1)
    w[ii] = 2 * v / N
    return x, w


def chebdif(N, M):
    ''' Chebyshev points and differentiation matrices up to order M (Weideman & Reddy) '''
    L = np.eye(N, dtype=bool)
    n1, n2 = N // 2, (N + 1) // 2
    k = np.arange(N)
    th = k * np.pi / (N - 1)
    x = np.sin(np.pi * np.arange(N - 1, -N, -2) / (2 * (N - 1)))
    T = np.tile(th[:, None] / 2, (1, N))
    DX = 2 * np.sin(T.T + T) * np.sin(T.T - T)
    DX = np.vstack((DX[:n1, :], -np.flipud(np.fliplr(DX[:n2, :]))))
    DX[L] = 1
    C = toeplitz((-1.0)**k)
    C[0, :] *= 2
    C[-1, :] *= 2
    C[:, 0] /= 2
    C[:, -1] /= 2
    Z = 1 / DX
    Z[L] = 0
    D = np.eye(N)
    DM = np.zeros((N, N, M))
    for ell in range(1, M + 1):
        D = ell * Z * (C * np.tile(np.diag(D)[:, None], (1, N)) - D)
        D[L] = -np.sum(D, axis=1)
        DM[:, :, ell - 1] = D
    return x, DM


@lru_cache(maxsize=None)
def mean_velocity(Re, N):
    ''' Turbulent mean profile from the Reynolds-Tiederman eddy viscosity (computed once per worker) '''
    kappa, Amp = 0.426, 25.4
    y, DM = chebdif(N, 2)
    D1, D2 = DM[:, :, 0], DM[:, :, 1]
    _, dy = clencurt(N - 1)
    nuT = 0.5 * ((1 + ((1 / 3) * kappa * Re * (1 - y**2) * (1 + 2 * y**2)
                       * (1 - np.exp(-(1 - np.abs(y)) * Re / Amp)))**2)**0.5 - 1)
    LHS = np.diag(1 + nuT) @ D2 + np.diag(D1 @ nuT) @ D1
    RHS = -Re * np.ones(N)
    LHS[[0, -1], :] = 0
    LHS[0, 0] = LHS[-1, -1] = 1
    RHS[[0, -1]] = 0
    U = np.linalg.solve(LHS, RHS)
    return y, D1, D2, dy, U


def robust_svd(A):
    ''' Dense SVD; falls back to LAPACK gesvd if gesdd does not converge '''
    try:
        return np.linalg.svd(A, full_matrices=False)
    except np.linalg.LinAlgError:
        return scipy.linalg.svd(A, full_matrices=False, lapack_driver='gesvd')


def resolvent(Re, kx, kz, cP, N, nsvd, Y):
    ''' Leading nsvd response modes [u;v;w;p] (4N x nsvd) and gains of the channel resolvent with
        admittance Y = v/p at both walls and the transpiration BC i*omega*u = v*dU at the wall.
        Y = 0 gives the rigid (no-slip) wall.                                                  '''
    omega = cP * kx
    y, D1, D2, dy, U = mean_velocity(Re, N)
    y = y + 1                                   # y in [0, 2]
    dU = D1 @ U

    I, Z = np.eye(N), np.zeros((N, N))
    A = -1j * kx * np.diag(U) + (-(kx**2 + kz**2) * I + D2) / Re
    L = np.block([[A, -np.diag(dU), Z, -1j * kx * I],
                  [Z, A, Z, -D1],
                  [Z, Z, A, -1j * kz * I],
                  [-1j * kx * I, -D1, -1j * kz * I, Z]])
    M = np.block([[I, Z, Z, Z], [Z, I, Z, Z], [Z, Z, I, Z], [Z, Z, Z, Z]])
    LHS, RHS = -1j * omega * M - L, M.copy()

    # wall rows of u, v, w (top and bottom)
    for r in [0, N - 1, N, 2 * N - 1, 2 * N, 3 * N - 1]:
        LHS[r, :] = 0
        RHS[r, :] = 0
    LHS[0, 0] = LHS[N - 1, N - 1] = -1j * omega            # i*omega*u = v*dU
    LHS[0, N], LHS[N - 1, 2 * N - 1] = dU[0], dU[N - 1]
    LHS[N, N], LHS[2 * N - 1, 2 * N - 1] = -1, 1            # v = Y*p (sign: outward normal)
    LHS[N, 3 * N], LHS[2 * N - 1, 4 * N - 1] = -Y, -Y
    LHS[2 * N, 2 * N] = LHS[3 * N - 1, 3 * N - 1] = 1       # w = 0
    H = np.linalg.solve(LHS, RHS)

    # energy-weighted SVD (pressure weight 0)
    w = np.sqrt(dy)
    W = np.concatenate([w, w, w, np.zeros(N)])
    iW = np.concatenate([1 / w, 1 / w, 1 / w, np.zeros(N)])
    uW, s, vhW = robust_svd(W[:, None] * H * iW[None, :])
    s = s[:nsvd]
    f = iW[:, None] * vhW[:nsvd, :].conj().T                # forcing modes

    # fix the SVD phase at the point nearest the critical layer
    ind = np.argmax(U > cP) if cP < U.max() else N - 1
    f = f * np.exp(-1j * np.angle(iW[ind] * uW[ind, :nsvd]))[None, :]

    # weights zero the pressure -> recover full modes from the unweighted resolvent
    u = (H @ f) / s[None, :]
    return u, s, y, dy, U


# ============================== one point ==============================

def point(kx, kz, cP, Y, RE, N, NSVD):
    psi, s, y, dy, _ = resolvent(RE, kx, kz, cP, N, NSVD, Y)
    q = psi * s                                             # sigma_j * psi_j, columns = ranks
    u, v, w, p = q[:N], q[N:2 * N], q[2 * N:3 * N], q[3 * N:]
    wP = cP * kx / RE
    walls = [0, N - 1]

    uv = np.sum(np.real(np.conj(u) * v), axis=1)            # all ranks
    uv2 = np.sum(np.real(np.conj(u[:, :2]) * v[:, :2]), axis=1)   # ranks 1+2 (Luhar eq. 2.17)
    amp2 = lambda Qm: np.sqrt(np.sum(np.abs(Qm[:, :2])**2, axis=1))
    amp = lambda Qm: np.sqrt(np.sum(np.abs(Qm)**2, axis=1))

    sc = dict(s1=s[0], s2=s[1],
              RS=np.sum(uv2 * (y - 1) * dy), RS_all=np.sum(uv * (y - 1) * dy),
              eta=amp2(v)[walls].max() / wP, eta_all=amp(v)[walls].max() / wP,
              p_wall=amp2(p)[walls].max(), u_max=amp2(u).max(), v_max=amp2(v).max())
    pr = dict(uu=np.sum(np.abs(u)**2, axis=1), vv=np.sum(np.abs(v)**2, axis=1),
              ww=np.sum(np.abs(w)**2, axis=1), uv=uv, pp=np.sum(np.abs(p)**2, axis=1))
    return sc, pr


def solve_task(args):
    ''' one (kx, kz, c) for all walls; rigid first, skipped walls copy the rigid result '''
    kx, kz, cP, Ys, (RE, N, NSVD, Y_SKIP) = args
    try:
        sc0, pr0 = point(kx, kz, cP, 0.0, RE, N, NSVD)
        rows, profs = [sc0], [pr0]
        for Y in Ys[1:]:
            sc, pr = (sc0, pr0) if abs(Y) < Y_SKIP else point(kx, kz, cP, Y, RE, N, NSVD)
            rows.append(sc)
            profs.append(pr)
        S = np.array([[r[k] for k in SCALARS] for r in rows])
        P = np.array([[p[k] for k in PROFILES] for p in profs], dtype=np.float32)
    except Exception as e:
        print(f'FAILED kx={kx:.4g} kz={kz:.4g} c={cP:.4g}: {e}', flush=True)
        S = np.full((len(Ys), len(SCALARS)), np.nan)
        P = np.full((len(Ys), len(PROFILES), N), np.nan, dtype=np.float32)
    return S, P


# ============================== sweep ==============================

def main():
    fin, fout = Path(sys.argv[1]), Path(sys.argv[2])
    I = np.load(fin)
    KX, KZ, CPall, Yall = I['kx'], I['kz'], I['cP'], I['Y']
    RE, N, NSVD, Y_SKIP = float(I['Re']), int(I['N']), int(I['nsvd']), float(I['Y_skip'])
    cfg = (RE, N, NSVD, Y_SKIP)
    names = [str(n) for n in I['walls']]
    nkx, nkz, nck, nW = len(KX), len(KZ), CPall.shape[1], Yall.shape[2]

    nwork = int(os.environ.get('SLURM_CPUS_PER_TASK') or os.environ.get('NWORK') or os.cpu_count() or 1)
    parts = fout.parent / (fout.stem + '_parts')            # per-kx checkpoints -> restart skips done kx
    parts.mkdir(parents=True, exist_ok=True)

    _, _, y, dy, U = resolvent(RE, 12.0, 120.0, 10.0, N, NSVD, 0.0)    # grid + smoke test
    t0 = time.time()
    solve_task((12.0, 120.0, 10.0, Yall[0, 0], cfg))
    t1 = time.time() - t0
    print(f'{fin.name}: {nkx*nkz*nck} points, {nW} walls, {nwork} workers, host {platform.node()}')
    print(f'one point: {t1:.2f} s  ->  estimate {t1*nkx*nkz*nck/nwork/3600:.1f} h', flush=True)

    t_start = time.time()
    with ProcessPoolExecutor(nwork) as ex:
        for i, kx in enumerate(KX):
            f = parts / f'kx_{i:03d}.npz'
            if f.exists():
                print(f'kx = {kx:.3f} done already, skipped', flush=True)
                continue
            tasks = [(kx, kz, c, Yall[i, l], cfg) for kz in KZ for l, c in enumerate(CPall[i])]
            out = list(ex.map(solve_task, tasks, chunksize=2))
            S = np.array([o[0] for o in out]).reshape(nkz, nck, nW, len(SCALARS))
            P = np.array([o[1] for o in out]).reshape(nkz, nck, nW, len(PROFILES), N)
            tmp = parts / f'kx_{i:03d}.tmp.npz'
            if SAVE_PROFILES:
                np.savez(tmp, S=S, P=P)
            else:
                np.savez(tmp, S=S)
            tmp.replace(f)                                   # atomic: no half-written checkpoints
            print(f'[{i+1}/{nkx}] kx = {kx:8.3f}   failed {np.isnan(S).any(axis=-1).sum()}   '
                  f'elapsed {(time.time()-t_start)/3600:.2f} h', flush=True)

    # ---- merge into one output file ----
    S = np.stack([np.load(parts / f'kx_{i:03d}.npz')['S'] for i in range(nkx)])     # (nkx, nkz, nck, nW, nS)
    res = {k: I[k] for k in I.files}                                                 # copy all input meta
    res.update(S=S, Y=Yall, cP=CPall, skipped=np.abs(Yall) < Y_SKIP, walls=names,
               scalars=SCALARS, profiles=PROFILES, y=y, dy=dy, U=U)
    if SAVE_PROFILES:
        res['P'] = np.stack([np.load(parts / f'kx_{i:03d}.npz')['P'] for i in range(nkx)])  # (nkx, nkz, nck, nW, 5, N)
    np.savez(fout, **res)
    print(f'wrote {fout}   total failed points: {np.isnan(S).any(axis=-1).sum()}', flush=True)

    iRS = SCALARS.index('RS')
    for name, (kx, kz, c) in {'NW': (12, 120, 10), 'VLSM': (1, 10, 16)}.items():
        a, b = np.argmin(abs(KX - kx)), np.argmin(abs(KZ - kz))
        l = np.argmin(abs(CPall[a] - c))
        r = S[a, b, l, :, iRS] / S[a, b, l, 0, iRS]
        print(f'{name}: RS/RS_rig = ' + ', '.join(f'{n} {x:.4f}' for n, x in zip(names, r)))


if __name__ == '__main__':
    main()