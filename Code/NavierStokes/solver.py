import jax
import jax.numpy as jnp

"""

uv_predictor, poisson_solver, uv_corrector
Following: https://www.montana.edu/mowkes/research/source-codes/GuideToCFD_2020_02_28_v2.pdf

Note rho=dx=dy=1

All periodic boundary conditions

"""

@jax.jit
def periodic_x(x):
    x_extend = jnp.insert(jnp.append(x,x[0:1,:],axis=0),0,x[-1:,:],axis=0)
    return x_extend

@jax.jit
def periodic_y(x):
    x_extend = jnp.insert(jnp.append(x,x[:,0:1],axis=1),jnp.array([0]),x[:,-1:],axis=1)
    return x_extend

@jax.jit
def u_periodic_x(u):
    u_extend = jnp.insert(jnp.append(u,u[1:2,:],axis=0),0,u[-2:-1,:],axis=0)
    return u_extend

@jax.jit
def v_periodic_y(v):
    v_extend = jnp.insert(jnp.append(v,v[:,1:2],axis=1),jnp.array([0]),v[:,-2:-1],axis=1)
    return v_extend

@jax.jit
def GS_step(params,it):
    x,b = params
    x_extend = periodic_x(x)
    x_ghost  = periodic_y(x_extend)
    x = (x_ghost[2:,1:-1]+x_ghost[1:-1,2:]+x_ghost[:-2,1:-1]+x_ghost[1:-1,:-2]+b)/4.0
    return (x,b),it

@jax.jit
def uv_predictor(dt,u,v,nu):
    ux_ghost = u_periodic_x(u)
    dudx = (ux_ghost[2:,:]-ux_ghost[:-2,:])/2.0
    d2udx2 = ux_ghost[2:,:]+ux_ghost[:-2,:]-2*u

    vy_ghost = v_periodic_y(v)
    dvdy = (vy_ghost[:,2:]-vy_ghost[:,:-2])/2.0
    d2vdy2 = vy_ghost[:,2:]+vy_ghost[:,:-2]-2*v

    vx_ghost = periodic_x(v)
    dvdx = (vx_ghost[2:,:]-vx_ghost[:-2,:])/2.0
    d2vdx2 = vx_ghost[2:,:]+vx_ghost[:-2,:]-2*v

    uy_ghost = periodic_y(u)
    dudy = (uy_ghost[:,2:]-uy_ghost[:,:-2])/2.0
    d2udy2 = uy_ghost[:,2:]+uy_ghost[:,:-2]-2*u

    u_extend = uy_ghost
    u_avg = (u_extend[:-1,:-1]+u_extend[1:,:-1]+u_extend[:-1,1:]+u_extend[1:,1:])/4.0

    v_extend = vx_ghost
    v_avg = (v_extend[:-1,:-1]+v_extend[1:,:-1]+v_extend[:-1,1:]+v_extend[1:,1:])/4.0

    ustar = u+dt*(-u*dudx-v_avg*dudy)+dt*(nu*(d2udx2+d2udy2))
    vstar = v+dt*(-u_avg*dvdx-v*dvdy)+dt*(nu*(d2vdx2+d2vdy2))
    return ustar,vstar

@jax.jit
def uv_corrector(dt,ustar,vstar,p):
    p_ghost = periodic_x(p)
    u = ustar-dt*(p_ghost[1:,:]-p_ghost[:-1,:])
    p_ghost = periodic_y(p)
    v = vstar-dt*(p_ghost[:,1:]-p_ghost[:,:-1])
    return u,v

@jax.jit
def advect(dt, f, u, v):
    # Donor cell
    f_ghost = periodic_x(f)
    flux_Fx = jnp.where(u > 0, u*f_ghost[:-1,:],u*f_ghost[1:,:])
    f_ghost = periodic_y(f)
    flux_Fy = jnp.where(v > 0, v*f_ghost[:,:-1],v*f_ghost[:,1:])

    delf_x = (flux_Fx[1:,:]-flux_Fx[:-1,:])
    delf_y = (flux_Fy[:,1:]-flux_Fy[:,:-1])

    return f-dt*(delf_x+delf_y)

def FluidSolver(Nsub,NGS):

    @jax.jit
    def GS_solve(x,b):
        input = (x,b)
        input,count = jax.lax.scan(GS_step,input,jnp.arange(NGS))
        x,b = input
        return x
    
    @jax.jit
    def poisson_solver(p,ustar,vstar):

        del_uv = (ustar[1:,:]-ustar[:-1,:])+(vstar[:,1:]-vstar[:,:-1])

        # Gauss-Seidal solution
        p = GS_solve(p,-del_uv)

        return p

    @jax.jit
    def NS_subcycle(input,it):
        u,v,p = input['u'],input['v'],input['p']
        nu,rho,NGS = input['nu'],input['rho'],input['NGS']
        dt = 1.0/Nsub

        # Ensure periodicity
        u = u.at[-1,:].set(u[0,:])
        v = v.at[:,-1].set(v[:,0])

        ustar,vstar = uv_predictor(dt,u,v,nu)
        p = poisson_solver(p,ustar,vstar)
        u,v = uv_corrector(dt,ustar,vstar,p)

        # Ensure periodicity
        u = u.at[-1,:].set(u[0,:])
        v = v.at[:,-1].set(v[:,0])
        rho = advect(dt,rho,u,v)

        input['u'],input['v'],input['p'],input['rho'] = u,v,p,rho
        return input,it
    
    @jax.jit
    def NS_time_step(input,it):
        input,count = jax.lax.scan(NS_subcycle,input,jnp.arange(Nsub))
        return input,it
    
    return NS_time_step,poisson_solver