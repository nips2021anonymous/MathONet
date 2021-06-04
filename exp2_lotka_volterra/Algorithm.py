"""consider dependency between both predecessors and successors;
consider both weight magnitude and weight uncertainty"""

import torch.nn.functional as F
#import numpy as np
import torch
import math

class MLP_Algorithm:
    def __init__(self, model, device, lambda_base):
        self.device = device
        self.model = model.to(self.device)
        self.lambda_base = lambda_base       
        #omage, gamma, alpha for weight
        self.gamma = {}
        self.omega = {}
        self.alpha = {}
        self.sum_gamma = {} 
        
        for name, value in model.state_dict().items():   
            if len(value.size())<=2:						
                self.gamma[name + '_group'] = torch.ones(value.size(), dtype=torch.float32, device=self.device)
                self.omega[name + '_group'] = torch.ones(value.size(), dtype=torch.float32, device=self.device)
                self.alpha[name + '_group'] = torch.ones(value.size(), dtype=torch.float32, device=self.device)
                self.gamma[name + '_l1'] = torch.ones(value.size(), dtype=torch.float32, device=self.device)
                self.omega[name + '_l1'] = torch.ones(value.size(), dtype=torch.float32, device=self.device)
                self.alpha[name + '_l1'] = torch.ones(value.size(), dtype=torch.float32, device=self.device)
                self.sum_gamma[name] = torch.ones(value.size(), dtype=torch.float32, device=self.device)             

    def update(self, hessian_auto, onoff_regularization):
        with torch.no_grad():
            #rename the Hessian
            for name in hessian_auto.keys():
                    #for 2D-weight update         
                sign_inner_in = False
                sign_inner_group = False
                sign_inner_l1 = False
                sign_inner_l2 = False
                gamma_inv = 0
                hessian = 0                
                hessian_layer = hessian_auto[name]               
                weight_layer = self.model.state_dict()[name]
                if len(weight_layer.size())<=3:
                    # update gamma for lambda group
                    if onoff_regularization['group'] == True:
                        group_temp_update_gamma = torch.zeros(self.gamma[name + '_group'].size()).to(self.device)
                        if float(self.lambda_base['lambda_group'][name])!=0:
                            sign_inner_group = True
                            for j in range(self.gamma[name + '_group'].size(1)):
                                group_gamma_row = self.gamma[name + '_group'].t()[j]
                                group_omega_row = self.omega[name + '_group'].t()[j]
                                group_length_row = len(group_gamma_row)
                                group_weight_row = weight_layer.t()[j]
                                group_weight_norm2 = torch.norm(group_weight_row.clone().detach(), 2).to(self.device)
                                group_omega_row_value = torch.sum(group_omega_row) / group_length_row
                                group_gamma_row_value = group_weight_norm2 / group_omega_row_value
                                group_gamma_row_value = torch.clamp(group_gamma_row_value, min=1e-6, max=1e6)
                                group_temp_update_gamma.t()[j] = torch.ones(group_length_row, dtype=torch.float32, device=self.device)* group_gamma_row_value                            
                                if math.isnan(group_gamma_row_value):
                                    print('algorithm check 2:', group_gamma_row_value, group_omega_row_value, group_weight_norm2)
                                    raise Exception("A NAN number appears")                                                        
                            self.gamma[name + '_group'] = group_temp_update_gamma
                            gamma_inv += 1 / self.gamma[name + '_group']
                            hessian += 1 / float(float(self.lambda_base['lambda_group'][name])) * hessian_layer
    
                    # update gamma for lambda l1
                    if onoff_regularization['l1'] == True:
                        if float(self.lambda_base['lambda_l1'][name])!=0:
                            sign_inner_l1= True
                            l1_omega_layer = self.omega[name + '_l1']            
                            l1_temp_update_gamma = torch.abs(weight_layer / l1_omega_layer).to(self.device)
                            l1_temp_update_gamma = torch.clamp(l1_temp_update_gamma, min=1e-6, max=1e6).to(self.device)
                            self.gamma[name + '_l1'] = l1_temp_update_gamma                    
                            gamma_inv += 1 / self.gamma[name + '_l1']
                            hessian += 1 / float(float(self.lambda_base['lambda_l1'][name])) * hessian_layer
 
                   # update for C
                    if sign_inner_in== True or sign_inner_group== True or sign_inner_l1== True or sign_inner_l2== True:
                        self.sum_gamma[name] = 1/gamma_inv
                        hessian_auto[name] = hessian
             
            
            for name in hessian_auto.keys():
                C1 = 1/self.sum_gamma[name] + hessian_auto[name]
                C2 = torch.reciprocal(C1)          
                C = torch.clamp(C2, min=1e-6, max=1e6).to(self.device)                  
        
                # group_group
                if onoff_regularization['group'] == True:
                    if float(self.lambda_base['lambda_group'][name]) != 0:
                        group_alpha = -C / (self.gamma[name + '_group'].pow(2)) + 1 / self.gamma[name + '_group']
                        group_alpha_reshape = group_alpha
                        self.alpha[name + '_group'] = group_alpha_reshape
                        group_update_omega_temp = torch.ones(self.gamma[name + '_group'].size(), dtype=torch.float32, device=self.device)
                        for i in range(self.gamma[name + '_group'].size(1)):
                            group_alpha_i = group_alpha_reshape.t()[i]
                            group_alpha_i_value = torch.sqrt(torch.abs(group_alpha_i).sum()/len(group_alpha_i))
                            group_alpha_i_value = torch.clamp(group_alpha_i_value, min=1e-6, max=1e6)
                            group_update_omega_temp.t()[i] = torch.ones(self.gamma[name + '_group'].size(0), dtype=torch.float32, device=self.device)* group_alpha_i_value                            
                            if math.isnan(group_alpha_i_value):
                                print('algorithm check 4:', group_alpha_i_value, group_alpha_i)
                                raise Exception("A NAN number appears")                            
                        self.omega[name + '_group'] = group_update_omega_temp
                        
                if onoff_regularization['l1'] == True:
                    if float(self.lambda_base['lambda_l1'][name]) != 0:
                        l1_alpha = -C / (self.gamma[name + '_l1'].pow(2)) + 1 / self.gamma[name + '_l1']
                        l1_alpha_reshape = l1_alpha
                        self.alpha[name + '_l1'] = l1_alpha_reshape
                        l1_update_omega_temp = torch.sqrt(torch.abs(l1_alpha_reshape))
                        l1_update_omega_temp  = torch.clamp(l1_update_omega_temp , min=1e-6, max=1e6)
                        self.omega[name + '_l1'] = l1_update_omega_temp
                        
        return


#
    def loss_cal(self, prediction, target,  onoff_regularization):
        target = target.float()
        # prediction loss
        loss_prediction = F.mse_loss(prediction.reshape(target.size()), target)		
        loss_reg = torch.tensor([0.]).to(self.device)

        for name, tmp_weight in self.model.state_dict().items():
            if len(tmp_weight.size())<4:            
                if onoff_regularization['group'] == True:
                    if self.lambda_base["lambda_group"][name] != 0:
                        loss_reg += self.lambda_base["lambda_group"][name].to(self.device)*torch.abs(torch.norm(torch.mul(tmp_weight,self.omega[name + '_group']), 2,0)).sum()
                        
                if onoff_regularization['l1'] == True:
                    if self.lambda_base["lambda_l1"][name] != 0:
                        loss_reg += self.lambda_base["lambda_l1"][name].to(self.device)*torch.abs(torch.norm(torch.mul(tmp_weight,self.omega[name + '_l1']), 1)).sum()
        
        loss = loss_prediction + loss_reg
        return loss_prediction, loss
     
    
    def extract_info(self):
        gamma_value = {}
        lambda_value = {}	
              #dict
        for name, param in self.model.state_dict().items():         
            if len(param.size())<4:                                                        
                gamma_value[name+'_group'] = self.gamma[name + '_group'].cpu().numpy()
                gamma_value[name+'_l1'] = self.gamma[name + '_l1'].cpu().numpy()
                gamma_value[name+'_sum'] = self.sum_gamma[name].cpu().numpy()
    			
    			
                lambda_value[name+'_group'] = self.lambda_base['lambda_group'][name].cpu().numpy()
                lambda_value[name+'_l1'] = self.lambda_base['lambda_l1'][name].cpu().numpy()
               
        return gamma_value, lambda_value


    def extract_hessian_info(self, input_hessian):
        hessian_value = {}		
        #dict
        for name, param in self.model.state_dict().items(): 
            if len(param.size())<4:                                                        
                hessian_value[name] = input_hessian[name].cpu().numpy()
        return hessian_value


