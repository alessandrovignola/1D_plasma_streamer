from scipy.sparse import spdiags, csr_matrix
from numba import njit, prange
import numpy as np
import matplotlib, time
import matplotlib.pyplot as plt
plt.style.use('default')
matplotlib.use('TkAgg')
from scipy.sparse.linalg import factorized
from scipy.sparse import csc_matrix

# Parameters
hx = 20e-6
L = 1e-2
Nx = int(L/hx)
x = np.linspace(hx/2, L-hx/2, Nx)

T = 50e-9  # End time

mu = 0.03  # m^2/Vs
D = 0.1  # m^2/s
ne = np.zeros(len(x))
ne[(x > 4e-3) & (x < 6e-3)] = 1e20
ni = ne.copy()
phi_0 = 10e3  # V
phi_L = 0  # V
e = 1.60217e-19
eps = 8.9e-12  # C^2/(N m^2)


def precompute_poisson_matrix():
    de = -eps / hx * np.ones(Nx)
    dw = de.copy()
    dc = -dw - de
    A = spdiags([dw, dc, de], [-1, 0, 1], Nx, Nx).tolil()
    A[0, 0] *= 3 / 2
    A[-1, -1] *= 3 / 2
    return csc_matrix(A), dw, de

A_poisson, dw_poisson, de_poisson = precompute_poisson_matrix()
solve_poisson = factorized(A_poisson)

def poisson(n):
    b = hx * (ni - n) * e
    b[0] -= 2 * dw_poisson[0] * phi_0
    b[-1] -= 2 * de_poisson[-1] * phi_L
    phi = solve_poisson(b)
    field = np.empty_like(phi)
    field[:-1] = -(phi[1:] - phi[:-1]) / hx
    return field

@njit
def koren(x):
    return np.maximum(0,np.minimum(1,np.minimum((2+x)/6,x)))

@njit(parallel=True)
def drift(n,field):
    nu = -mu*field
    n_plus = np.empty_like(n)
    n_minus = np.empty_like(n)
    n_plus[:-1] = n[1:]
    n_plus[-1] = n[-1]
    n_minus[1:] = n[:-1]
    n_minus[0] = n[0]

    dn = n_plus - n
    r = (n - n_minus) / (dn + 1e-20)
    r[0] = 0
    r[-1] = 0

    lim = koren(r)
    lim2 = koren(1 / (np.roll(r, -1) + 1e-20))

    flux = np.where(nu >= 0,
                    nu*(n+lim*dn),
                    nu*(n_plus-lim2*dn))
    return flux

@njit(parallel=True)
def diff(n):
    n_plus = np.empty_like(n)
    n_plus[:-1] = n[1:]
    n_plus[-1] = n[-1]
    flux = -D*(n_plus-n)/hx
    return flux

@njit()
def fluxes(n,field):

    # Returns the fluxes at cell face i+1/2
    return drift(n,field)+diff(n)


@njit
def grad(n, field):
    flux = fluxes(n, field)
    g = (flux-np.roll(flux, 1))/hx
    return g

@njit
def trapezoid(y,dt,field):
    # Performs explicit time integration according to trapezoidal rule

    f = -grad(y,field)
    y_tilde = y + dt*f
    y = y + dt/2*(f-grad(y_tilde,field))
    return y

def trap1(y,dt,field):
    y = y - dt*grad(y,field)
    return y