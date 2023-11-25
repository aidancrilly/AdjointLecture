import diffrax
import optax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.nn as jnn
import jax.random as jrandom
import matplotlib.pyplot as plt
import time

def RLC_equations(t,y,args):
    I,Idot = y
    R,L,C,dVdt = args['R'],args['L'],args['C'],args['dVdt']

    dIdt   = Idot
    d2Idt2 = (-R*Idot-I/C+dVdt(t))/L

    return jnp.array([dIdt,d2Idt2])

def V(t,omega0,tmax):
    omega = omega0*(2*t/tmax)
    return jnp.sin(omega*t)

omega0 = 4.0
tmax   = 10.0

# Truth solution
args = {'R' : 0.5, 'L' : 1.0, 'C' : 1/omega0**2, 'dVdt' : jax.grad(lambda t : V(t,omega0,tmax))}
tsave    = jnp.linspace(0.0,tmax,500)
term     = diffrax.ODETerm(RLC_equations)
solver   = diffrax.Heun()
adjoint  = diffrax.RecursiveCheckpointAdjoint()
saveat   = diffrax.SaveAt(ts=tsave)
y0       = jnp.array([0.0,0.0])
solution_true = diffrax.diffeqsolve(term, solver, t0=0, t1=tsave[-1], dt0=0.5*(tsave[1]-tsave[0]), y0=y0, saveat=saveat, args = args, adjoint=adjoint)

I_true = solution_true.ys[:,0]

# NDE solution
# Following https://docs.kidger.site/diffrax/examples/neural_ode/
class Func(eqx.Module):
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

    def __call__(self, t, y, args):
        circuit_response = self.mlp(y)
        circuit_response = circuit_response.at[1].set(circuit_response[1]+args['dVdt'](t))
        return circuit_response
    
class NeuralODE(eqx.Module):
    func: Func

    def __init__(self, in_size, out_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.func = Func(in_size, out_size, width_size, depth, key=key)

    def __call__(self, ts, y0, args):
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            diffrax.Heun(),
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=y0,
            args=args,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
            saveat=diffrax.SaveAt(ts=ts),
            max_steps=int(1e6)
        )
        return solution.ys
    
def train_NDE(
    ts,ys,args,
    lr_strategy=(1e-2,1e-2),
    steps_strategy=(200, 500),
    length_strategy=(0.5, 1),
    width_size=32,
    depth=3,
    seed=5678,
    plot=True,
    print_every=10,
):
    key = jrandom.PRNGKey(seed)
    __, model_key = jrandom.split(key)

    length_size = ts.shape[0]

    model = NeuralODE(2, 2, width_size, depth, key=model_key)

    # Training loop like normal.
    #
    # Only thing to notice is that up until step 200 we train on only the first 50% of
    # each time series. This is a standard trick to avoid getting caught in a local
    # minimum.

    @eqx.filter_value_and_grad
    def grad_loss(model, ti, yi):
        y_pred = model(ti, yi[0, :],args)
        return jnp.mean((yi[:,0] - y_pred[:,0]) ** 2)

    @eqx.filter_jit
    def make_step(ti, yi, model, opt_state):
        loss, grads = grad_loss(model, ti, yi)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    count = 0
    for lr, steps, length in zip(lr_strategy, steps_strategy, length_strategy):
        optim = optax.adabelief(lr)
        opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
        _ts = ts[: int(length_size * length)]
        _ys = ys[: int(length_size * length),:]
        for step in range(steps):
            count += 1
            start = time.time()
            loss, model, opt_state = make_step(_ts, _ys, model, opt_state)
            end = time.time()
            if (step % print_every) == 0 or step == steps - 1:
                print(f"Step: {step}, Loss: {loss}, Computation time: {end - start}")

                if plot:
                    fig = plt.figure(dpi=200)
                    ax1 = fig.add_subplot(121)
                    ax2 = fig.add_subplot(122)
                    ax1.plot(ts, ys[:, 0], c="dodgerblue", label="Real")
                    model_y = model(ts, ys[0, :],args)
                    ax1.plot(ts, model_y[:, 0], c="crimson", label="Model")
                    ax1.set_xlabel("t")
                    ax1.set_ylabel("I")
                    ax1.legend()
                    ax2.plot(ts, ys[:, 1], c="dodgerblue", ls = '--')
                    ax2.plot(ts, model_y[:, 1], c="crimson", ls = '--')
                    ax2.set_xlabel("t")
                    ax2.set_ylabel("dI/dt")
                    fig.tight_layout()
                    fig.savefig(f"./Plots/neural_ode_{str(count).zfill(4)}.png")
                    plt.close(fig)

    return model

NDE_model = train_NDE(tsave,solution_true.ys,args,plot=False)

def V2(t,omega0,tmax):
    omega = 2*omega0
    return jnp.sin(omega*t)

args = {'R' : 0.5, 'L' : 1.0, 'C' : 1/omega0**2, 'dVdt' : jax.grad(lambda t : V2(t,omega0,tmax))}

# New test 
ts = tsave
solution = diffrax.diffeqsolve(term, solver, t0=0, t1=tsave[-1], dt0=0.5*(tsave[1]-tsave[0]), y0=y0, saveat=saveat, args = args, adjoint=adjoint)
ys = solution.ys
fig = plt.figure(dpi=200)
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.plot(ts, ys[:, 0], c="dodgerblue", label="Real")
model_y = NDE_model(ts, ys[0, :],args)
ax1.plot(ts, model_y[:, 0], c="crimson", label="Model")
ax1.set_xlabel("t")
ax1.set_ylabel("I")
ax1.legend()
ax2.plot(ts, ys[:, 1], c="dodgerblue", ls = '--')
ax2.plot(ts, model_y[:, 1], c="crimson", ls = '--')
ax2.set_xlabel("t")
ax2.set_ylabel("dI/dt")
fig.tight_layout()
fig.savefig(f"./Plots/test_neural_ode.png")
plt.close(fig)