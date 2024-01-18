from motion import *
from scipy.stats import qmc

def produce_samples(d,n,nfastforward,l_bounds,u_bounds):
	""" Use scipy Halton sequence sampling to produce uniform sampling for model """
	sampler = qmc.Halton(d=d)
	if(nfastforward != 0):
		_ = sampler.fast_forward(nfastforward)
	sample      = sampler.random(n=n)
	sample      = qmc.scale(sample, l_bounds, u_bounds)
	return sample

# Input parameters
seed = 0
key = jrandom.PRNGKey(seed)
t0 = 0.0
t1 = 2.5
Nt = 100
ts = jnp.linspace(t0,t1,Nt)
Nparticles = int(1e5)
xextent = 10.0
Nbins = 200

# Sampling parameters
dimensionality = 2
n_samples      = 50
n_samples_prev = 0

l_bounds = [0.0,0.0]
u_bounds = [2.0,2.0]

sample = produce_samples(dimensionality,n_samples,n_samples_prev,l_bounds,u_bounds)
damping = sample[:,0]
noise   = sample[:,1]
keys = jrandom.split(key,damping.shape[0])
bins = jnp.linspace(-xextent,xextent,Nbins+1)

# Solve particle trajectories
ys = vmapped_ParticleSwarmSolver(damping,noise,keys,ts,bins,Nparticles)

import h5py
with h5py.File("./Data/BrownianData.h5", "w") as f:
    grp = f.create_group("Inputs")
    grp.attrs['Nparticles'] = Nparticles
    dset = grp.create_dataset("damping", damping.shape, dtype='f')
    dset[...] = damping
    dset = grp.create_dataset("noise", noise.shape, dtype='f')
    dset[...] = noise
    dset = grp.create_dataset("times", ts.shape, dtype='f')
    dset[...] = ts
    dset = grp.create_dataset("bins", bins.shape, dtype='f')
    dset[...] = bins
    grp = f.create_group("Outputs")
    dset = grp.create_dataset("counts", ys.shape, dtype='f')
    dset[...] = ys