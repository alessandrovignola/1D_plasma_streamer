# Solver based on Streamer's current limited approach

from shared import *

@njit
def trap_lim(y,dt,field):
    flux = fluxes_lim(y,field,dt)
    f1 = -(flux - np.roll(flux, 1)) / hx
    tilde = y + dt*f1
    flux2 = fluxes_lim(tilde,field,dt)
    f2 = -(flux2-np.roll(flux2, 1)) / hx

    new = y + dt/2*(f1 + f2)
    return new

@njit
def e_star(n,field):
    dn_num = np.abs(np.roll(n,-1)-n)
    dn_den = hx*np.maximum(.1,np.maximum(n,np.roll(n,-1)))
    dn = dn_num/dn_den

    field_star = np.maximum(np.abs(field), dn*D/mu)
    return field_star

@njit
def fluxes_lim(n,field,dt):
    flux = fluxes(n,field)
    E_star = e_star(n,field)
    mask = np.abs(flux) > eps * E_star / (e * dt)
    flux[mask] = np.sign(flux[mask]) * eps * E_star[mask] / (e * dt)
    return flux

def current_lim():
    ht = 80e-12
    Nt = int(T / ht)
    n = ne
    field = poisson(n)

    for i in range(Nt):
        n = trap_lim(n,ht,field)
        field = poisson(n)

    return (field+np.roll(field,1))/2, n