from fontTools.misc.psLib import endofthingRE
from scipy.sparse import spdiags, csr_matrix
from scipy.sparse.linalg import spsolve
import numpy as np
import matplotlib, time
import matplotlib.pyplot as plt
plt.style.use('default')
matplotlib.use('TkAgg')
from scipy.sparse.linalg import factorized
from scipy.sparse import csc_matrix

# Parameters
nx = 500
L = 1e-2
hx = L/nx
ht = 80e-12

x = ((np.arange(-1, nx+3) - 0.5) * hx)  # Lunghezza = nx + 4
T = 50e-9  # End time
Nt = int(T / ht)


mu = 0.03  # m^2/Vs
D = 0.1  # m^2/s
ne = np.zeros(len(x))
ne[(x > 4e-3) & (x < 6e-3)] = 1e20
ni = ne.copy()
phi_0 = 10e3  # V
phi_L = 0  # V
e = 1.60217e-19
eps = 8.9e-12  # C^2/(N m^2)


def koren(x):
    return np.maximum(0,np.minimum(1,np.minimum((2+x)/6,x)))

def drift(n,field):
    nu = -mu*field
    n_plus = np.empty_like(n)
    n_minus = np.empty_like(n)
    n_plus[:-1] = n[1:]
    n_plus[-1] = n[-1]
    n_minus[1:] = n[:-1]
    n_minus[0] = n[0]

    dn = n_plus - n
    r = (n - n_minus) / (dn + 1e-50)
    r[0] = 0
    r[-1] = 0

    lim = koren(r)
    lim2 = koren(1 / (np.roll(r, -1) + 1e-50))

    flux = np.where(nu >= 0,
                    nu*(n+lim*dn),
                    nu*(n_plus-lim2*dn))
    return flux


def diff(n):
    n_plus = np.empty_like(n)
    n_plus[:-1] = n[1:]
    n_plus[-1] = n[-1]
    flux = -D*(n_plus-n)/hx
    return flux


def fluxes(n,field):

    # Returns the fluxes at cell face i+1/2
    return drift(n,field)+diff(n)

def grad(n, field):
    flux = fluxes(n, field)
    g = np.empty_like(flux)
    g[1:] = (flux[1:] - flux[:-1]) / hx
    g[0] = 0; g[-1] = 0
    return g


def trapezoid(y,dt,field):
    # Performs explicit time integration according to trapezoidal rule
    f = -grad(y,field)
    y_tilde = y + dt*f
    y = y + dt/2*(f-grad(y_tilde,field))
    return y

def impl_poisson(n,dt):
    n_face = .5*(n[1:-3]+n[2:-2])
    print(n_face)
    coeff = -eps/e -dt*mu*n_face

    l_diag = coeff[:-2]/hx**2; l_diag[0] = 2*l_diag[0]
    u_diag = coeff[1:-1]/hx**2; u_diag[-1] = 2*u_diag[-1]
    diag = -l_diag-u_diag

    rhs = ni[2:-2] - n[2:-2]
    rhs = rhs - dt*D/hx**2*(n[1:-3]-2*n[2:-2]+n[3:-1])
    rhs[0] = rhs[0] - phi_0*l_diag[0]; rhs[-1] = rhs[-1] - phi_L*u_diag[-1]

    A = spdiags([l_diag, diag, u_diag], [-1, 0, 1], nx, nx).tolil()
    phi = spsolve(A.tocsr(), rhs)

    field_fc = np.zeros(nx+1)
    field_fc[0] = -(phi[0] - phi_0) / (0.5 * hx)
    field_fc[-1] = -(phi_L - phi[-1]) / (0.5 * hx)

    field_fc[1:nx] = -(phi[1:] - phi[:-1]) / hx


    field_cc = 0.5* (field_fc[1:] + field_fc[: nx])

    return field_cc


for i in range(1):
    field = impl_poisson(ne, ht)
    ne = trapezoid(ne,ht,field)

