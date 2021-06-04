#import model_pde_selection as md
from scipy.io import savemat
import torch
from scipy.integrate import odeint
from scipy.sparse import diags

# In[0] ode system data
import numpy as np

device = 'cpu'

p = [1.3, 0.9, 0.8, 1.8]
α, β, γ, δ = p

def lotka(u, t):
    du1 = α*u[0] - β*u[1]*u[0]
    du2 = γ*u[0]*u[1]  - δ*u[1]
    du = [du1,du2]
    return du

# Define the experimental parameter
# time span
tspan = np.linspace(0, 30, 300)


#    tspan = (0.0f0,3.0f0)a
u0 = [0.44249296,4.6280594]
sol = odeint(lotka, u0, tspan)  
target = torch.tensor(sol).to(device)
initial_condition = torch.tensor(u0).squeeze().to(device)
time_series = torch.tensor(tspan).to(device)    

du = []
for i in range(len(target)):
    du1, du2 = lotka(target[i],tspan)
    du.append(torch.tensor([du1.item(), du2.item()]).unsqueeze(0))
    
du = torch.tensor(torch.cat(du,0))

savemat("data/lotka_volterra.mat",
            {
                'Output': du.cpu().detach().numpy(),
                'Input': target.cpu().numpy(),
            })   

