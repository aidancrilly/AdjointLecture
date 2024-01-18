import jax
import jax.random as jrandom
import jax.numpy as jnp
from diffrax import diffeqsolve, ControlTerm, Euler, MultiTerm, ODETerm, SaveAt, VirtualBrownianTree

def ParticleSwarmSolver(damping,noise,key,ts,pos_bins,Nparticles):

    def drift(t,p,args):
        r,v = p
        return jnp.array([v,-damping*v])

    def diffusion(t,p,args):
        return jnp.array([[0.0,0.0],[0.0,noise*t]])
    
    @jax.jit
    def solver(ts,y0,key):

        t0,t1 = ts[0],ts[-1]

        brownian_motion = VirtualBrownianTree(t0, t1, tol=1e-3, shape=(2,), key=key)
        terms = MultiTerm(ODETerm(drift), ControlTerm(diffusion, brownian_motion))
        solver = Euler()

        saveat = SaveAt(ts=ts)

        sol = diffeqsolve(terms, solver, t0, t1, dt0=dt, y0=y0, saveat=saveat)
        return sol.ys

    dt = 0.5*(ts[1]-ts[0])
    vmap_solver = jax.vmap(solver,in_axes=(None,1,0))

    y0 = jnp.array([jrandom.uniform(key,shape=(Nparticles,),minval=-0.5,maxval=0.5),jrandom.normal(key,shape=(Nparticles,))])
    keys = jrandom.split(key,Nparticles)
    ys = vmap_solver(ts,y0,keys)

    t_histogram = jax.vmap(lambda a : jnp.histogram(a,bins=pos_bins)[0],in_axes=(1,))

    counts = t_histogram(ys[:,:,0])/Nparticles

    return counts

vmapped_ParticleSwarmSolver = jax.vmap(ParticleSwarmSolver,in_axes=(0,0,0,None,None,None))