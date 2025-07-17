# Solver using a fourth order Runge-Kutta method with small time-step

from shared import *

@njit
def RK4(y,dt,field):

    k1 = -grad(y,field)
    k2 = -grad(y+dt/2*k1,field)
    k3 = -grad(y+dt/2*k2,field)
    return y + dt/6*(k1+2*k2+2*k3-grad(y+dt*k3,field))

def exact():
    ht = 0.1e-12  # Time step
    Nt = int(T / ht)
    n = ne

    for i in range(Nt):
        field = poisson(n)
        n = RK4(n,ht,field)
    return (field+np.roll(field,1))/2, n

