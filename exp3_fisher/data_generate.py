#import model_pde_selection as md
from scipy.io import savemat
import torch
from scipy.integrate import odeint
from scipy.sparse import diags

# In[0] ode system data
import numpy as np

device = 'cpu'

D = 0.01 #diffusion
r = 1.0 #reaction rate
  
X = 1.0
T = 20

dx = 0.04
dt = T/10

x = torch.tensor(list(range(26))).float()/25
t = T*torch.tensor(list(range(11))).float()/10
Nx = len(x)
Nt = len(t)
  
#initial conditions
Amp = 1.0
Delta = 0.2
#IC-1
rho0 = Amp*(torch.tanh((x - (0.5 - Delta/2))/(Delta/10)) - torch.tanh((x - (0.5 + Delta/2))/(Delta/10)))/2    

# define function
def fisher_kpp(u):  
    u = r*u*(1-u)
    return u   

# generate the tri-diagonal matrix
k = 1/(dx*dx)*np.array([-2*np.ones(Nx),np.ones(Nx-1),np.ones(Nx-1)])
offset = [0,1,-1]
lap = diags(k,offset).toarray()
lap[0,-1] = 1/(dx*dx)
lap[-1,0] = 1/(dx*dx)

def rc_ode(rho,t):
    #finite difference
    rho = torch.tensor(rho)
    rho = rho.clone().detach().squeeze().unsqueeze(1).float()
    drho_dt = D * torch.mm(torch.tensor(lap).float(), rho) + fisher_kpp(rho)  #第一项中，lap相当于是对x取二阶的操作； 而lap*rho才相当于是rho对x的二阶
#        drho_dt = fisher_kpp(rho)  #第一项中，lap相当于是对x取二阶的操作； 而lap*rho才相当于是rho对x的二阶
    drho_dt = drho_dt.squeeze()
    return drho_dt

# time span
t = np.linspace(0, T, Nt)
#get ode solution
sol = odeint(rc_ode, rho0, t)

initial_condition = rho0.to(device)
time_series = torch.tensor(t).to(device)    
target = torch.tensor(sol).to(device)


du = []
for i in range(len(target)):
    dudt = rc_ode(target[i],t)
    du.append(dudt.unsqueeze(0))
    
du = torch.tensor(torch.cat(du,0))


savemat("data/FisherKPP.mat",
            {
                'Output': du.cpu().detach().numpy(),
                'Input': target.cpu().numpy(),
            })    

