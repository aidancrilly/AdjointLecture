from motion import *
import numpy as np
import matplotlib.pyplot as plt

seed = 0
key = jrandom.PRNGKey(seed)
t0 = 0.0
t1 = 2.5
Nt = 100
ts = jnp.linspace(t0,t1,Nt)
Nparticles = int(1e5)

damping = jnp.array([0.0,0.5,1.0])
noise   = jnp.array([0.1,0.1,0.1])
keys = jrandom.split(key,damping.shape[0])

ys = vmapped_ParticleSwarmSolver(damping,noise,keys,ts,Nparticles)

bins = jnp.linspace(-10,10,100)
for it in range(Nt):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    for j in range(damping.shape[0]):
        plt.hist(np.array(ys[j,:,it,0]),bins=bins,alpha=0.5,density=True)
    fig.tight_layout()
    fig.savefig(f"./Plots/Brownian_{str(it).zfill(4)}.png")
    plt.close(fig)