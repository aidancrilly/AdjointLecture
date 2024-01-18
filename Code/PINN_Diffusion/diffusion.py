import optax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.nn as jnn
import jax.random as jrandom
import matplotlib.pyplot as plt

# Diffusivity
D = 1.0

def diffusion_solution(t,x):
    return jnp.exp(-x**2/(4*D*t))/jnp.sqrt(4*jnp.pi*D*t)

class PINN(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, in_size, out_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.mlp = eqx.nn.MLP(
            in_size=in_size,
            out_size=out_size,
            width_size=width_size,
            depth=depth,
            activation=jnn.tanh,
            key=key,
        )

    def __call__(self, x, t):
        input = jnp.array([x,t])
        y_PINN = self.mlp(input)
        return y_PINN.reshape(())
    
def train_PINN(
    training_ts,training_xs,training_sol,
    plotting_ts,plotting_xs,
    lr_strategy=(1e-3,),
    steps_strategy=(2000,),
    width_size=32,
    depth=3,
    seed=5678,
    plot=True,
    print_every=50,
):
    key = jrandom.PRNGKey(seed)
    __, model_key = jrandom.split(key)

    model = PINN(2, 1, width_size, depth, key=model_key)

    tt,xx = jnp.meshgrid(plotting_ts,plotting_xs,indexing='ij')
    sol_y = diffusion_solution(tt.flatten(),xx.flatten())

    # Training loop like normal.

    @eqx.filter_jit
    def PDE_loss(model,ti,xi):
        dydt   = jax.vmap(jax.grad(model,argnums=1))
        d2ydx2 = jax.vmap(jax.grad(jax.grad(model,argnums=0),argnums=0))
        g_PDE  = dydt(xi,ti)-D*d2ydx2(xi,ti)
        return jnp.mean(g_PDE**2)

    @eqx.filter_jit
    def MSE_loss(model,ti,xi,yi):
        y_pred = jax.vmap(model)(xi,ti)
        return jnp.mean((yi - y_pred) ** 2)

    @eqx.filter_value_and_grad
    def grad_loss(model, ti, xi, yi):
        MSE    = MSE_loss(model,ti,xi,yi)
        g_PDE  = PDE_loss(model,ti,xi)
        return MSE+g_PDE

    @eqx.filter_jit
    def make_step(ti, xi, yi, model, opt_state):
        loss, grads = grad_loss(model, ti, xi, yi)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    count = 0
    for lr, steps in zip(lr_strategy, steps_strategy):
        optim = optax.adabelief(lr)
        opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
        for step in range(steps):
            count += 1
            loss, model, opt_state = make_step(training_ts, training_xs, training_sol, model, opt_state)
            
            if (step % print_every) == 0 or step == steps - 1:
                print(step,MSE_loss(model,training_ts, training_xs, training_sol),PDE_loss(model,training_ts, training_xs))
                if plot:
                    fig = plt.figure(dpi=200)
                    ax1 = fig.add_subplot(121)
                    ax2 = fig.add_subplot(122)
                    model_y = jax.vmap(model)(xx.flatten(),tt.flatten())
                    ax1.pcolormesh(tt,xx,model_y.reshape(plotting_ts.shape[0],plotting_xs.shape[0]),vmin=0,vmax=jnp.amax(sol_y))
                    ax2.pcolormesh(tt,xx,sol_y.reshape(plotting_ts.shape[0],plotting_xs.shape[0]),vmin=0,vmax=jnp.amax(sol_y))
                    ax2.plot(training_ts,training_xs,'kx')
                    ax1.set_xlabel("t")
                    ax1.set_ylabel("x")
                    ax1.set_title(f"PINN, step = {step}")
                    ax2.set_title("Truth/Data")
                    fig.tight_layout()
                    fig.savefig(f"./Plots/PINN_{str(count).zfill(4)}.png")
                    plt.close(fig)

    return model

extent = 4.0

plotting_ts = jnp.linspace(0.5,1.5,50)
plotting_xs = jnp.linspace(-extent,extent,100)

data_seed = 404
key = jrandom.PRNGKey(data_seed)
Ntrain = 100

training_xs = jax.random.uniform(key,shape=(Ntrain,),minval=-extent,maxval=extent)
__, key = jrandom.split(key)
training_ts = jax.random.uniform(key,shape=(Ntrain,),minval=0.5,maxval=1.5)

training_ys = diffusion_solution(training_ts,training_xs)

NDE_model = train_PINN(training_ts,training_xs,training_ys,plotting_ts,plotting_xs,plot=True)
