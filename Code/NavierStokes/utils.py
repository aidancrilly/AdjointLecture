from solver import *
from matplotlib.image import imsave
import matplotlib.pyplot as plt

def NS_animate(u0,v0,rho0,nu,NGS,Nt,Nsub,Nsave):

    NS_Solver,poisson_solver = FluidSolver(Nsub,NGS)
    
    i = 0
    p0 = poisson_solver(jnp.zeros_like(rho0),u0,v0)
    imsave(f"step_{i:03d}.png",rho0,cmap='Blues')

    input = {'rho' : rho0, 'u' : u0, 'v' : v0, 'p' : p0, 'nu' : nu, 'NGS' : NGS, 'Nsub' : Nsub}
    for i in range(1,Nt+1):
        input,_ = NS_Solver(input,i)
        rho,p,u,v = input['rho'],input['p'],input['u'],input['v']
        if(i%Nsave == 0):
            imsave(f"step_{i:03d}.png",rho,cmap='Blues')