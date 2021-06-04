"""
this dataset is with the selection for convlutional stencil 
"""

import algorithm_lz_depend_v4 as Algorithm
import model_node as md
from scipy.io import savemat
import scipy.io as scio
import torch
import numpy as np

#import datetime
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from scipy.integrate import odeint
from scipy.sparse import diags
if __name__ == '__main__':  # multi-processing protection
    folder_save_data = './' +'data/'  # the folder to save result            	 
    sigma = 10  # Lorenz's parameters (chaotic)
    beta = 8/3
    rho = 28
    noise = 0.01

    def lorenz(u,t):
        y1,y2,y3 = u
        du1 = sigma*(y2-y1);
        du2 =  y1*(rho-y3)-y2;
        du3 =  y1*y2-beta*y3;
        du = [du1,du2,du3]
        return du

    u0=[-8.0, 8.0, 27.0]
    tspan = np.linspace(0.001, 5, 100)

    sol = odeint(lorenz, u0, tspan)  
    target = torch.tensor(sol).to(device)
    initial_condition = torch.tensor(u0).squeeze().to(device)
    time_series = torch.tensor(tspan).to(device)    
    
    du = []
    for i in range(len(target)):
        du1, du2, du3 = lorenz(target[i],tspan)
        du.append(torch.tensor([du1.item(), du2.item(), du3.item()]).unsqueeze(0))
        
    du = torch.tensor(torch.cat(du,0))+noise*torch.rand(du.size())
    
    savemat(folder_save_data+'lorenz_noise'+ str(noise)  + '_best_model.mat',
            {
                'x': target.cpu().numpy(),
                'dx': torch.tensor(
                    du).cpu().detach().numpy(),
            })    
