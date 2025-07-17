# Solver using a semi-implicit discretization
from scipy.linalg import solve

from shared import *

def impl_poisson(n,dt):
    n_plus = np.concatenate((n[1:], [n[0]]))
    n_minus = np.concatenate(([n[-1]], n[:-1]))

    n_fp = (n+n_plus)/2 # Need to compute density at faces
    n_fm = (n+n_minus)/2

    dal = (-eps/e-dt*mu*n_fm)/hx**2
    dar = (-eps/e-dt*mu*n_fp)/hx**2

    #dal[0] = 2*dal[0]; dar[-1] = 2*dar[-1]
    da = -dal-dar
    da[0] += (eps/e + dt*mu*(n_fp[0]+2*n_fm[0]))/hx**2
    da[-1] += (eps/e + dt*mu*(n_fm[-1]+2*n_fp[-1]))/hx**2


    b = ni-n-dt*D*(n_minus-2*n+n_plus)/hx**2
    #b[0] -= phi_0*dal[0]; b[-1] -= phi_L*dal[-1]
    b[0] += (2*eps/e + 2*n_fm[0]*dt*mu)*phi_0/hx**2
    b[-1] += (2*eps/e + 2*n_fp[-1]*dt*mu)*phi_L/hx**2


    phi = solve_tridiag(dal, da, dar, b)
    field = -(np.roll(phi, -1)-phi) / hx
    field[0] = field[1]; field[-1] = field[-2]
    return field  # Returns  E at face i+1/2

def solve_tridiag(a, b, c, d):
    n = len(b)
    cp = np.zeros(n)
    dp = np.zeros(n)
    x = np.zeros(n)

    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]

    for i in range(1, n):
        m = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / m if i < n - 1 else 0.0
        dp[i] = (d[i] - a[i] * dp[i - 1]) / m

    x[-1] = dp[-1]
    for i in reversed(range(n - 1)):
        x[i] = dp[i] - cp[i] * x[i + 1]

    return x

def semi_implicit():
    ht = 80e-12
    Nt = int(T / ht)
    n = ne
    for i in range(Nt):
        field = impl_poisson(n, ht)
        n = trapezoid(n,ht,field)

    return (field+np.roll(field,1))/2, n