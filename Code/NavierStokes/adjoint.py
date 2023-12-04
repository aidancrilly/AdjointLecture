from solver import *
from utils import *
import os
from imageio import imread
import matplotlib.pyplot as plt
import optax

def learnable_params_to_n(p,args):
    # Get to correct shape
    p_reshape = p.reshape(args['Nx'],args['Ny'],9)
    n_p = args['rho_init'][:,:,None]*jax.nn.softmax(p_reshape)
    return n_p

def params_loss(p,args):
    n = learnable_params_to_n(p,args)
    n = LBM_Nt_time_step(n,args['Nsimsteps'])
    rho,_,_ = project(n)
    return jnp.mean((rho-args['target'])**2)

print("Loading initial and target states...")
basepath = os.path.dirname(__file__)
target = 1-jnp.array(imread(os.path.join(basepath, 'inputs/CIFS_target.png')))[::4,::4,0]/255
initial = 1-jnp.array(imread(os.path.join(basepath, 'inputs/init_bars.png')))[::4,::4,0]/255
Nx,Ny = target.shape

xb = (jnp.arange(Nx+1)-Nx/2)/Nx
yb = (jnp.arange(Ny+1)-Ny/2)/Ny
x  = 0.5*(xb[1:]+xb[:-1])
y  = 0.5*(yb[1:]+yb[:-1])

xx,yy = jnp.meshgrid(xb,y,indexing='ij')
u0 = 0.05*jnp.sin(2*jnp.pi*yy/y[-1])#jnp.zeros_like(xx)
xx,yy = jnp.meshgrid(x,yb,indexing='ij')
v0 = 0.5*jnp.sin(2*jnp.pi*xx/x[-1])#jnp.zeros_like(yy)

rho0 = initial
nu  = 0.01
NGS = 100
Nsub = 10
Nt = 500
Nsave = 10

# args = {'target' : target, 'rho_init' : initial, 
#         'Nx' : Nx, 'Ny' : Ny, 'Nsimsteps' : Nsimsteps}

# LossGradFunc = jax.value_and_grad(params_loss)

# optimizer = optax.adam(learning_rate=0.01)
# opt_state = optimizer.init(p_init)

# p = p_init.copy()
# loss_history = []
# Noptsteps = 100
# for i in range(Noptsteps):
#     loss, grad_loss = LossGradFunc(p,args)
#     print(i,loss)
#     updates, opt_state = optimizer.update(grad_loss, opt_state)
#     p = optax.apply_updates(p, updates)

#     loss_history.append(loss)

# n0 = learnable_params_to_n(p,args)
# LBM_animate(n0,Nsimsteps)

NS_animate(u0,v0,rho0,nu,NGS,Nt,Nsub,Nsave)