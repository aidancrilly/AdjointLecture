import numpy as np
import matplotlib.pyplot as plt

def f(x,y):
    return np.exp(-x**2-y**2)

def g(x,y):
    return y-(x-1)**2

def dfdx(x,y):
    return -2*x*f(x,y)

def dfdy(x,y):
    return -2*y*f(x,y)

def dgdx(x,y):
    return -2*(x-1)

def dgdy(x,y):
    return 1

x = np.linspace(-2.0,2.0,200)
y = x.copy()

xx,yy = np.meshgrid(x,y)

fig = plt.figure(dpi=200,figsize=(4,4))
ax1 = fig.add_subplot(111)

ax1.contour(xx,yy,f(xx,yy),colors='b',levels=[0.01,0.1,0.5,0.75,0.99])
ax1.contour(xx,yy,g(xx,yy),colors='r',levels=[-5.0,-2.0,-1.0,-0.5,0.5,1.0])
ax1.contour(xx,yy,g(xx,yy),colors='k',levels=[0.0])

xopt = 0.41025
yopt = (xopt-1)**2
ax1.plot(xopt,yopt,'ko')
grad_scale = 1.5e-1
ax1.arrow(xopt,yopt,grad_scale*dgdx(xopt,yopt),grad_scale*dgdy(xopt,yopt),width=1e-2,color='k')
grad_scale = 2e-1
ax1.arrow(xopt,yopt,grad_scale*dfdx(xopt,yopt),grad_scale*dfdy(xopt,yopt),width=1e-2,color='b')

fig.tight_layout()
plt.show()