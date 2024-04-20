#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# -*- coding: utf-8 -*-
"""
Simulation code used for the numerical illustration of the article 
"Boundary stabilization of a class of coupled reaction-diffusion system with one control"
by Gonzalo Arias and Hugo Parada.
Code written by Gonzalo Arias.
We refer to the article for details of the system being simulated here.
"""


# In[ ]:


import numpy as np
import scipy.linalg as la
import scipy.special as sp
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'notebook')

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from scipy.interpolate import RectBivariateSpline
from scipy.integrate import simpson


# In[ ]:


plt.close("all")
plt.rcParams.update({
  "mathtext.fontset": "stix"
})


# In[ ]:


CASE = 1
# CASE = 1 Means that the reduced order system is unstable, therefore the 2x2 heat equation is unstable.
#          We then apply a feedback controller to stabilize both systems. 

# CASE = 2 Means that we have an exponentially stable reduced order system
#          therefore yielding the exponential stability of the 2x2 heat equation.



# In[ ]:


def backstepping(x,Nx,nu):
  """
  Simulation to compute the kernel of the backstepping transformation proposed in the article.

  Parameters
  ----------
  x  : numpy.ndarray, vector of shape (Nx,). Space nodes.
  Nx : positive integer. Number of spaces steps.
  nu : positive number. Is given by $\lambda - \tilde{\lambda}$. \tilde{\lambda} is a design parameter.
  
  Returns
  -------
  K  : numpy.ndarray, square matrix with shape (Nx-2, Nx-2). 
       Discretized kernel of the backstepping transformation.
       
  L  : numpy.ndarray, square matrix with shape (Nx-2, Nx-2). 
       Discretized kernel of the inverse backstepping transformation.
  """
  K=np.tri(Nx-2,Nx-2)
  L=np.tri(Nx-2,Nx-2)
  for i in range(0,Nx-2):
        for j in range(0,Nx-2):
            if i==j and i >= j:
                K[i,i]=(nu/2)*x[i]
                L[i,i]=(nu/2)*x[i]
            if i!=j and i>j:
                z=np.sqrt(nu*(x[j]**2-x[i]**2))
                K[i,j]=nu*x[j]*(sp.iv(1,z)/z)
                L[i,j]=nu*x[j]*(sp.jv(1,z)/z)
  return K,L


# In[ ]:


def error(xi,zeta,T,Nt,Nx):
  """ 
  Simulation of the approximation error given by Tikhonov's type theorem.
  
  Parameters
  ----------
  xi   : numpy.ndarray, shape (Nt, Nx, 1). Represent the slow error dynamic.
  zeta : numpy.ndarray, shape (Nt, Nx, 1). Represent the fast error dynamic.
  T    : positive number. Time horizon.
  Nt   : positive integer. Number of time steps.
  Nx   : positive integer. Number of spaces steps.
  
  
  Returns
  -------
  E    : numpy.ndarray, shape (Nt,). 
         Error of approximating the slow dynamic and fast dynamic throughout the reduced models.
  """
  dx=1/Nx
  E=np.empty(Nt)
  for i in range(0,Nt):
        E[i] =((np.dot(zeta[i,:],zeta[i,:]))+np.dot(xi[i,:],xi[i,:]))*dx
  return E


# In[ ]:


def simul_BLS(gamma, v0_bls, T_bls, Nt_bls, CFL):  
  """
  Simulation of the boundary layer system.

  Parameters
  ----------
  gamma  : real number, should be greater than $-\pi^2$
  v0_bls : numpy.ndarray, vector with shape (Nx,). Initial condition.
  T_bls  : positive number. Time horizon.
  Nt     : positive integer. Number of time steps.
  Nx     : positive integer. Number of spaces steps.
  Returns
  -------
  
  v_bls : numpy.ndarray, shape (Nt, Nx, 1). Values of the boundary-layer system.
  t     : numpy.ndarray, vector of shape (Nt,). Time nodes.
  x     : numpy.ndarray, vector of shape (Nx,). Space nodes.
  """
  t = np.linspace(0, T_bls, Nt_bls)
  dt = t[1] - t[0]
  dx=np.sqrt(dt/CFL)
  Nx=int(np.floor(1/dx))+1
  x = np.linspace(0, 1, Nx)
  dx = x[1]-x[0]
  CFL=dt/(dx**2)
  v_bls = np.empty((Nt_bls, Nx, 1))
  v_bls[0,:] = v0_bls(x.reshape(-1,1))
  A=np.eye(Nx-2,Nx-2,k=-1)*(dt/(dx**2)) + np.eye(Nx-2,Nx-2)*(1-2*dt/(dx**2)-dt*(gamma)) + np.eye(Nx-2,Nx-2,k=1)*(dt/(dx**2))
  for n in range(Nt_bls-1):
      v_bls[n+1,0]=0
      v_bls[n+1,Nx-1]=0
      v_bls[n+1,:][1:Nx-1] = A.dot(v_bls[n,:][1:Nx-1])
  return t, x, v_bls


# In[ ]:


def simul_ROS(alpha, beta, gamma, Lambda, u0_ros, T, Nt,CFL): 
  """
  Simulation of the reduced order system.

  Parameters
  ----------
  gamma  : real number. Should be greater than $-\pi^2$.
  Lambda : real number. Reaction term.
  alpha  : real number. Coupling term.
  beta   : real number. Coupling term.
  u0_ros : numpy.ndarray, vector with shape (Nx,). Initial condition.
  T      : positive number. Time horizon.
  Nt     : positive integer. Number of time steps.
  CFL    : positive number. Condition that assures the convergence of the numerical method.
  
  Returns
  -------
  
  u_ros : numpy.ndarray, shape (Nt, Nx, 1). Values of the reduced order system.
  v_ros : numpy.ndarray, shape (Nt, Nx, 1). Values of the quasi static solution.
  t     : numpy.ndarray, vector of shape (Nt,). Time nodes.
  x     : numpy.ndarray, vector of shape (Nx,). Space nodes.
  """   
    
  t = np.linspace(0, T, Nt)
  dt = t[1] - t[0]
  dx=np.sqrt(dt/CFL)
  Nx=int(np.floor(1/dx))+1
  x = np.linspace(0, 1, Nx)
  dx = x[1]-x[0] 
  
  A=np.eye(Nx-2,Nx-2,k=-1)*(dt/(dx**2)) + np.eye(Nx-2,Nx-2)*(1-2*dt/(dx**2)-dt*(Lambda)) + np.eye(Nx-2,Nx-2,k=1)*(dt/(dx**2))
  B=np.diag(np.full(Nx-2,dt*alpha))
  C=np.eye(Nx-2,Nx-2,k=-1)*(-1/(dx**2)) + np.eye(Nx-2,Nx-2)*(2/(dx**2)+(gamma)) + np.eye(Nx-2,Nx-2,k=1)*(-1/(dx**2)) 
  G=A+beta*B.dot(la.inv(C))

  u_ros = np.empty((Nt, Nx, 1))
  v_ros = np.empty((Nt, Nx, 1))
    
  u_ros[0,:]=u0_ros(x.reshape(-1,1))
  v_ros[0,:]=u0_ros(x.reshape(-1,1))

  v_ros[0,:][1:Nx-1]=beta*((la.inv(C)).dot(u_ros[0,:][1:Nx-1]))


  for n in range(Nt-1):
        u_ros[n+1,0],u_ros[n+1,Nx-1],v_ros[n+1,0],v_ros[n+1,Nx-1]=0,0,0,0
        u_ros[n+1,:][1:Nx-1] = G.dot(u_ros[n,:][1:Nx-1])
        v_ros[n+1,:][1:Nx-1] = beta*((la.inv(C)).dot(u_ros[n+1,:][1:Nx-1]))
  return t,x,u_ros,v_ros
  


# In[ ]:


def simul_ROS_controlled(alpha, beta, gamma, tilde_Lam, Lambda, u0_ros, T, Nt,CFL):
  """
  Simulation of the stabilized reduced order system.

  Parameters
  ----------
  alpha     : real number. Coupling term.
  beta      : real number. Coupling term.
  gamma     : real number. Should be greater than $-\pi^2$.
  tilde_Lam : real number. Design parameter of the backstepping kernel. Shouldn't be too large.
  Lambda    : real number. Reaction term.
  u0_ros    : numpy.ndarray, vector with shape (Nx,). Initial condition.
  T         : positive number. Time horizon.
  Nt        : positive integer. Number of time steps.
  CFL       : positive number. Condition that assures the convergence of the numerical method.
  
  Returns
  -------
  
  u_ros : numpy.ndarray, shape (Nt, Nx, 1). Values of the stabilized reduced order system.
  v_ros : numpy.ndarray, shape (Nt, Nx, 1). Values of the stabilized quasi static solution.
  t     : numpy.ndarray, vector of shape (Nt,). Time nodes.
  x     : numpy.ndarray, vector of shape (Nx,). Space nodes.
  """ 
  t = np.linspace(0, T, Nt)
  dt = t[1] - t[0]
  dx=np.sqrt(dt/CFL)
  Nx=int(np.floor(1/dx))+1
  x = np.linspace(0, 1, Nx)
  dx = x[1]-x[0]
  CFL=dt/(dx**2) 
  nu=Lambda-tilde_Lam
  
  K,L=backstepping(x,Nx,nu)
  TK=np.eye(Nx-2,Nx-2)+dx*K
  TL=np.eye(Nx-2,Nx-2)-dx*L
    
  A=np.eye(Nx-2,Nx-2,k=-1)*CFL + np.eye(Nx-2,Nx-2)*(1-2*CFL-dt*(tilde_Lam)) + np.eye(Nx-2,Nx-2,k=1)*CFL
  B=np.diag(np.full(Nx-2,dt*alpha))
  C=np.eye(Nx-2,Nx-2,k=-1)*(-1/(dx**2)) + np.eye(Nx-2,Nx-2)*(2/(dx**2)+gamma) + np.eye(Nx-2,Nx-2,k=1)*(-1/(dx**2))
  D=la.inv(C)
  G=A+beta*B.dot(TK).dot(D).dot(TL)
  
  u_ros = np.empty((Nt, Nx, 1))
  v_ros = np.empty((Nt, Nx, 1))
  w_ros = np.empty((Nt, Nx, 1))

  u_ros[0,:]=u0_ros(x.reshape(-1,1))
  v_ros[0,:]=u0_ros(x.reshape(-1,1))
  w_ros[0,:]=u0_ros(x.reshape(-1,1))
    
  w0=u0(x.reshape(-1,1))
  w0[1:Nx-1]=w0[1:Nx-1]+dx*(K.dot(w0[1:Nx-1]))
  v_ros[0,:][1:Nx-1]=beta*(D.dot(u_ros[0,:][1:Nx-1]))

  for n in range(Nt-1):
      u_ros[n+1,0],w_ros[n+1,0],w_ros[n+1,Nx-1],v_ros[n+1,0],v_ros[n+1,Nx-1]=0,0,0,0,0 #Boundary conditions
        
      w_ros[n+1,:][1:Nx-1] = G.dot(w_ros[n,:][1:Nx-1])
      v_ros[n+1,:][1:Nx-1] = beta*D.dot(TL).dot(w_ros[n,:][1:Nx-1])
      u_ros[n+1,:][1:Nx-1]=TL.dot(w_ros[n+1,:][1:Nx-1])
      u_ros[n+1,Nx-1]= -dx*np.dot(L[Nx-3,:],w_ros[n+1,:][1:Nx-1])        # We actuate in the u_ros variable
  return t,x,u_ros,v_ros
  


# In[ ]:


def simul_full(alpha,beta,gamma,Lambda, u0, v0, epsilon, T, Nt, CFL):
  """
  Simulation of the full system.

  Parameters
  ----------
  alpha     : real number. Coupling term.
  beta      : real number. Coupling term.
  gamma     : real number. Should be greater than $-\pi^2$.
  Lambda    : real number. Reaction term.
  u0        : function: x numpy.ndarray (Nx,) |--> y0(x) numpy.ndarray (Nx,). Initial condition of the slow dynamic.
  v0        : function: x numpy.ndarray (Nx,) |--> y0(x) numpy.ndarray (Nx,). Initial condition of the fast dynamic.
                    *** Both IC should accept as inputs vectors of any size Nx. ***
                    
  epsilon   : Positive number. Parameter of the singular perturbation.
  T         : Positive number. Time horizon.
  Nt        : Positive integer. Number of time steps.
  CFL       : Number in the interval (0, 1/2). Desired CFL condition. The true CFL condition will be smaller than or equal
              to this value.

  Returns
  -------
  t  : numpy.ndarray, vector of shape (Nt,). Times.
  x  : numpy.ndarray, vector of shape (Nx,). Space variables.
  u  : numpy.ndarray, shape (Nt, Nx, 1). Values of the slow variable.
  v  : numpy.ndarray, shape (Nt, Nx, 1). Values of the fast variable.
  Nx : integer with the number of space steps
  Nt : integer with the number of time steps

  """

  e=1/epsilon
  t = np.linspace(0, T, Nt)
  dt = t[1] - t[0]
  dx=np.sqrt(dt*e/CFL)
  Nx=int(np.floor(1/dx))+1
  x = np.linspace(0, 1, Nx)
  dx = x[1]-x[0]
  u = np.empty((Nt, Nx, 1))
  v = np.empty((Nt, Nx, 1))
  CFL=e*dt/(dx**2)  

  u[0, :] = u0(x.reshape(-1,1))
  v[0, :] = v0(x.reshape(-1,1))
  
  B1=np.diag(np.full(Nx-2,dt*alpha))
  B2=np.diag(np.full(Nx-2,e*dt*beta))
  A1=np.eye(Nx-2,Nx-2,k=-1)*(dt/(dx**2)) + np.eye(Nx-2,Nx-2)*(1-2*dt/(dx**2)-dt*Lambda) + np.eye(Nx-2,Nx-2,k=1)*(dt/(dx**2))
  A2=np.eye(Nx-2,Nx-2,k=-1)*(CFL) + np.eye(Nx-2,Nx-2)*e*(epsilon-2*dt/(dx**2)-dt*gamma) + np.eye(Nx-2,Nx-2,k=1)*(e*dt/(dx**2))
  
  for n in range(Nt - 1):
      u[n+1,0],u[n+1,Nx-1],v[n+1,0],v[n+1,Nx-1]=0,0,0,0       #Boundary conditions 

      u[n+1,:][1:Nx-1]=A1.dot(u[n,:][1:Nx-1])+B1.dot(v[n,:][1:Nx-1])
      v[n+1,:][1:Nx-1]=A2.dot(v[n,:][1:Nx-1])+B2.dot(u[n,:][1:Nx-1])
  return t, x, u, v, 


# In[ ]:


def target_controlled(alpha,beta,gamma,tilde_Lam,Lambda, u0, v0, epsilon, T, Nt, CFL):
  """
  Simulation of the stabilized full system. 

  Parameters
  ----------
  alpha     : real number. Coupling term.
  beta      : real number. Coupling term.
  gamma     : real number. Should be greater than $-\pi^2$.
  tilde_Lam : real number. Design parameter of the backstepping kernel. Shouldn't be too large.
  Lambda    : real number. Reaction term.
  u0        : function: x numpy.ndarray (Nx,) |--> y0(x) numpy.ndarray (Nx,). Initial condition of the slow dynamic.
  v0        : function: x numpy.ndarray (Nx,) |--> y0(x) numpy.ndarray (Nx,). Initial condition of the fast dynamic.
                    *** Both IC should accept as inputs vectors of any size Nx. ***
                    
  epsilon   : Positive number. Parameter of the singular perturbation.
  T         : Positive number. Time horizon.
  Nt        : Positive integer. Number of time steps.
  CFL       : Number in the interval (0, 1/2). Desired CFL condition. The true CFL condition will be smaller than or equal
              to this value.

  Returns
  -------
  t  : numpy.ndarray, vector of shape (Nt,). Times.
  x  : numpy.ndarray, vector of shape (Nx,). Space variables.
  u  : numpy.ndarray, shape (Nt, Nx, 1). Values of the slow variable.
  v  : numpy.ndarray, shape (Nt, Nx, 1). Values of the fast variable.
  Nx : integer with the number of space steps
  Nt : integer with the number of time steps

  """
  nu=Lambda-tilde_Lam
  e=1/epsilon
  t = np.linspace(0, T, Nt)
  dt = t[1] - t[0]
  dx=np.sqrt(dt*e/CFL)
  Nx=int(np.floor(1/dx))+1
  x = np.linspace(0, 1, Nx)
  dx = x[1]-x[0]
  u = np.empty((Nt, Nx, 1))
  w = np.empty((Nt, Nx, 1))
  v = np.empty((Nt, Nx, 1))
  CFL=dt*e/(dx**2)  

  K,L=backstepping(x,Nx,nu)
    
  B1=np.diag(np.full(Nx-2,dt*alpha))
  B2=np.diag(np.full(Nx-2,e*dt*beta))
  C1=alpha*dx*dt*K
  C2=-1*e*beta*dx*dt*L
  A1=np.eye(Nx-2,Nx-2,k=-1)*(dt/(dx**2)) + np.eye(Nx-2,Nx-2)*(1-2*dt/(dx**2)-dt*(tilde_Lam)) + np.eye(Nx-2,Nx-2,k=1)*(dt/(dx**2))
  A2=np.eye(Nx-2,Nx-2,k=-1)*(CFL) + np.eye(Nx-2,Nx-2)*e*(epsilon-2*dt/(dx**2)-dt*gamma) + np.eye(Nx-2,Nx-2,k=1)*(e*dt/(dx**2))
  
  w0=u0(x.reshape(-1,1))
  w0[1:Nx-1]=w0[1:Nx-1]+dx*(K.dot(w0[1:Nx-1]))
  w[0, :] = w0
  u[0, :] = u0(x.reshape(-1,1))
  v[0, :] = v0(x.reshape(-1,1))
  
  for n in range(Nt - 1):
      u[n+1,0],v[n+1,0],v[n+1,Nx-1],w[n+1,0],w[n+1,Nx-1]=0,0,0,0,0                 #Boundary conditions 
      w[n+1,:][1:Nx-1]=A1.dot(w[n,:][1:Nx-1])+B1.dot(v[n,:][1:Nx-1])+C1.dot(v[n,:][1:Nx-1])
      v[n+1,:][1:Nx-1]=A2.dot(v[n,:][1:Nx-1])+B2.dot(v[n,:][1:Nx-1])+C2.dot(w[n,:][1:Nx-1])
      u[n+1,:][1:Nx-1]=w[n+1,:][1:Nx-1]- dx*(L.dot(w[n+1,:][1:Nx-1]))
      u[n+1,Nx-1]= -dx*np.dot(L[Nx-3,:],w[n+1,:][1:Nx-1])
  return t, x, u, v, Nx,Nt


# In[ ]:


# =============================================================================
# Parameters of the simulations
# =============================================================================
if CASE==1:
    alpha=5
    beta=6
    gamma=0.1
    Lambda=-10
    tilde_Lam=60
else:
    alpha=2
    beta=2.5
    gamma=0.1
    Lambda=3

u0 = lambda x: x*(1-x)
v0 = lambda x: -10*np.sin(x)*(1-x)

T = 5
T_bls = 20

CFL = 0.5

epsilons = np.array([0.03,0.05, 0.1, 0.125])
Nt_ros = 1000
Nt_epsilon = 6 # Number of time steps in each interval of size epsilon
Nt_bls = T_bls * Nt_epsilon
Nts = ((Nt_epsilon * T / epsilons + 1)**2).astype(int)

# When epsilon is small, we need very small time steps in order to have enough
# space steps in the discretization of the PDE while ensuring a reasonable CFL
# condition for stability


# In[ ]:


# =============================================================================
# Simulations
# =============================================================================

# First we have to compute the solutions to the reduced order system, boundary-layer system and the full system
# If in addition the reduced order system is unstable (therefore, the full system is unstable), we compute the solution
# to the reduced order system and full system when applying the same backstepping based control.

print("Computing solution of the boundary layer system in the time interval [0, {}] with {} time steps...".format(T_bls, Nt_bls), end = " ")
t, x, v_bls=simul_BLS(gamma, v0, T_bls, Nt_bls, CFL)
print("ok!")

print("Computing the solution of the reduced order system in the time interval [0, {}] with {} time steps...".format(T, Nt_ros), end = " ")
t_ros,x_ros,u_ros,v_ros=simul_ROS(alpha, beta, gamma, Lambda, u0, T, Nt_ros,CFL)
print("ok!")

h=RectBivariateSpline(t_ros,x_ros,u_ros[:,:,0]) #Function that serves to interpolate u_rosC
g=RectBivariateSpline(t_ros,x_ros,v_ros[:,:,0]) #Function that serves to interpolate v_rosC

res_full = []
for i in range(epsilons.size):
    print("Computing the solution of the full system with epsilon = {} in the time interval [0, {}] with {} time steps...".format(epsilons[i], T, Nts[i]), end = " ")
    res_full.append(simul_full(alpha,beta,gamma,Lambda, u0, v0, epsilons[i], T, Nts[i], CFL))
    print("ok!")


if CASE == 1:

    print("Computing the solution of the reduced order system in the time interval [0, {}] with {} time steps...".format(T, Nt_ros), end = " ")
    t_ros,x_ros,u_rosC,v_rosC=simul_ROS_controlled(alpha, beta, gamma, tilde_Lam, Lambda, u0, T, Nt_ros,CFL)
    print("ok!")
    
    h_c=RectBivariateSpline(t_ros,x_ros,u_rosC[:,:,0]) #Function that serves to interpolate u_rosC
    g_c=RectBivariateSpline(t_ros,x_ros,v_rosC[:,:,0]) #Function that serves to interpolate v_rosC
    res_full_controlled = []
    for i in range(epsilons.size):
        print("Computing the solution of the full system with epsilon = {} in the time interval [0, {}] with {} time steps...".format(epsilons[i], T, Nts[i]), end = " ")
        res_full_controlled.append(target_controlled(alpha,beta,gamma,tilde_Lam,Lambda, u0, v0, epsilons[i], T, Nts[i], CFL))
        print("ok!")
else:
    CASE=2


# In[ ]:


i = 1

fig, ax = plt.subplots(1, 2, subplot_kw={"projection": "3d"}, figsize = (9, 9))
ax[0].set_position([0, 0.5, 0.45, 0.45])
ax[1].set_position([0.5, 0.5, 0.45, 0.45])
fig.suptitle(r"Solution of the PDE with $\varepsilon = " + "{:3g}$".format(epsilons[i]))
t, x, u, v = res_full[i]
T1, X = np.meshgrid(t, x)
ax[0].set_zlabel("$u(\cdot, \cdot)$")
ax[1].set_zlabel("$v(\cdot, \cdot)$")
for j in range(2):
    ax[j].set_xlabel("$t$")
    ax[j].set_ylabel("$x$")
ax[0].plot_surface(T1, X, u[:, :, 0].T, rcount = 100, ccount = 100, cmap = "plasma", antialiased=False, linewidth = 0)
ax[1].plot_surface(T1, X, v[:, :, 0].T, rcount = 100, ccount = 100, cmap = "plasma", antialiased=False, linewidth = 0)

if CASE==1:
  fig, ax = plt.subplots(2, 1, subplot_kw={"projection": "3d"}, figsize = (9, 9))
  ax[0].set_position([0, 0.5, 0.45, 0.45])
  ax[1].set_position([0.5, 0.5, 0.45, 0.45])
  fig.suptitle(r"Solution of the controlled PDE with $\varepsilon = " + "{:3g}$".format(epsilons[i]))
  t, x, u, v, NX,NT = res_full_controlled[i]
  T1, X = np.meshgrid(t, x)
  ax[0].set_zlabel("$u(\cdot, \cdot)$")
  ax[1].set_zlabel("$v(\cdot, \cdot)$")
  for j in range(2):
      ax[j].set_xlabel("$t$")
      ax[j].set_ylabel("$x$")
  ax[0].plot_surface(T1, X, u[:, :, 0].T, rcount = 1000, ccount = 1000, cmap = "plasma", antialiased=False, linewidth = 0)
  ax[1].plot_surface(T1, X, v[:, :, 0].T, rcount = 1700, ccount = 1700, cmap = "plasma", antialiased=False, linewidth = 0)
  #ticks = np.arange(0, 10, 2)
  #ax[0, 0].set_xticks(ticks, ["${:1g}".format(t) + r"\varepsilon$" if t != 0 else "$0$" for t in ticks])
  #ax[1, 0].set_xticks(ticks, ["${:1g}".format(t) + r"\varepsilon$" if t != 0 else "$0$" for t in ticks])
  # rcount and ccount should be increased to get a better quality of the graphs. In the article, we took rcount = ccount = 1000 for the plot of y_fast and rcount = ccount = 5000 for the plot of y. With these values, the graph may take a long time to be shown.

else:
    print("your system is stable, enjoy")
    
## COMPUTING ERROR OF APPROXIMATION

linestyles = [(0, (5, 1)), (0, (4, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1)), (0, (2, 1, 1, 1, 1, 1, 1, 1)), (0, (1, 1))]

if CASE==1:
    fig, ax = plt.subplots(figsize = (5, 5))
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.set_xlabel("Time $t$")
    ax.set_ylabel("$E(t)$")
    ax.set_title("Error of Tikhonov approximation")
    for i in range(epsilons.size):
        t, x, u, v, NX,NT=res_full_controlled[i]
        u_new=h_c(t,x)
        v_new=g_c(t,x)
        Error=error(u[:,:,0]-u_new,v[:,:,0]-v_new,T,NT,NX)
        Error=Error[t <= 0.04]
        t_short = t[t <= 0.04]
        ax.plot(t_short, Error, linestyle = linestyles[i], label = r"$\varepsilon = " + "{:.3g}".format(epsilons[i]) + "$")
    ax.legend()
else:
    fig, ax = plt.subplots(figsize = (5, 5))
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.set_xlabel("Time $t$")
    ax.set_ylabel("$E(t)$")
    ax.set_title("Error of Tikhonov approximation")
    for i in range(epsilons.size):
        t, x, u, v, NX,NT=res_full[i]
        u_new=h(t,x)
        v_new=g(t,x)
        Error=error(u[:,:,0]-u_new,v[:,:,0]-v_new,T,NT,NX)
        Error=Error[t <= 0.04]
        t_short = t[t <= 0.04]
        ax.plot(t_short, Error, linestyle = linestyles[i], label = r"$\varepsilon = " + "{:.3g}".format(epsilons[i]) + "$")
    ax.legend()


# In[ ]:




