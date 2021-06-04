"""
this dataset is with the selection for convlutional stencil 
"""
import Algorithm
import model as md
from scipy.io import savemat
import scipy.io as scio
import torch
import copy
import lib
import os
import pickle as pkl
import math
#import time
#import matplotlib.pyplot as plt
#import numpy as np
#import datetime
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if __name__ == '__main__':  # multi-processing protection
    folder_save_result = './' +'save_result/'  # the folder to save result            
    if not os.path.exists(folder_save_result):
        os.makedirs(folder_save_result)
    experiment_index = 'e1'    
    
    # training hyper-parameters
    lr = 1e-2    
    cycles = 2
    epochs = 4
    lambda_decay_epochs = 800  # labmda will decay every 'lambda_decay_epochs' epochs
    amount_repeat = 10 # number of repeated experiments
    val_batch_size = 1 #batchsize for evaluation
    train_batch_size = 1 #batchsize for train

    # Data processing parameter
    percentage_train_test = 0.9   # percentage to devide train dataset and test dataset
    data_norm_sign_list = ['original'] #'norm: normalization',original: original data'    
    onoff_random_seed_list = [False] #'True'
    index_models = [1] # the index for the model to be trained 

    # MathONet initilization	
    MathONet_hidden_neurons = [3] #hidden layers and setting for prior-network

    # Network purning hyperparameters
    onoff_regularization = {'group': True, 'l1': True}
    lambda_org = {
                  'lambda_group': torch.tensor([1e-2]),
                  'lambda_l1': torch.tensor([1e-2]),
                  }     #lambda_group: regulazation parameter for group pruning;  lambda_l1:regulazation parameter for connection pruning;
    gamma_threshold = 0.058  # threshold for gamma pruning if used
    weight_threshold = 1e-3  # threshold for weight pruning if used 
    alpha_threshold = 0.01  # threshold for alpha pruning if used    
    for index_model in index_models:
        for rep_index in range(amount_repeat):
            NN_dataset_name = 'data/FisherKPP.mat'       
            for data_norm_sign in data_norm_sign_list:
                load_data = scio.loadmat(NN_dataset_name)                          
                estimation_input_org = torch.tensor(load_data['Input']).float()        
                estimation_output_org = torch.tensor(load_data['Output']).float()               
                estimation_input_pro, estimation_output_pro, validation_input_pro, validation_output_pro = lib.divide_data(
                    estimation_input_org, estimation_output_org, percentage_train_test)                              
                if data_norm_sign == 'norm':
                    onoff_normalize_data = True								
                    estimation_input_pro, estimation_output_pro, mean_X_train, std_X_train, mean_y_train, std_y_train  = \
                        lib.trainData_normalize(estimation_input_pro, estimation_output_pro,
                                       onoff_normalize_data)
                    validation_input_pro  = \
                        lib.testData_normalize(validation_input_pro, mean_X_train, std_X_train)           
                if data_norm_sign == 'original':
                    onoff_normalize_data = False
                    estimation_input_pro, estimation_output_pro, validation_input_pro, std_y_train, mean_y_train, mean_X_train, std_X_train = \
                        lib.clean_data(estimation_input_pro, estimation_output_pro, validation_input_pro,
                                       onoff_normalize_data)   														
                input_features = 1										
                output_features = 1
                             

                unary_names = ['sin','cos','ident'] #['sin','cos','ident','log','exp','square','sqrt','cube']
                MathONet_layers = [input_features] + MathONet_hidden_neurons + [output_features]
                            
                seed = 0 # decide if random initialization										
                for onoff_random_seed in onoff_random_seed_list:                                               
                    if onoff_random_seed == True:
                        torch.manual_seed(seed)
                    seed += 1
                model = md.MathONet(MathONet_layers, unary_names, torch.device(device))
                lambda_base = {}
                lambda_base['lambda_group'] = {}
                lambda_base['lambda_l1'] = {}

                # for current lambda
                for name, value in model.named_parameters():
                    if len(value.size())>2:
                        pass
                    else:
                        sign_size_1 = 1. 
                        if int(torch.max(torch.tensor(value.size()))) == 1:
                            sign_size_1 = 0. # if the size of one weight matrix equals 1, then only one lambda option would be retained
                        lambda_base['lambda_group'][name] = copy.deepcopy(lambda_org['lambda_group'])*sign_size_1
                        lambda_base['lambda_l1'][name] = copy.deepcopy(lambda_org['lambda_l1'])

                #initialize model, optimizer and learning rate schedule for every independent experiment
                filename = folder_save_result + experiment_index + '_model_'+str(index_model) + '_repeat_' + str(rep_index)
                minimal_loss = 1e10  # for comparison about the validation error. Maybe the bigger the better
                last_cycle_loss = 1e10
                algorithm = Algorithm.MLP_Algorithm(model, model.save_device, lambda_base)
                optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)  
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * cycles)

                # define the mask   
                masks = {}                                                          
                for name, value in model.named_parameters():                                        
                    mask = torch.abs(value) == 0
                    masks[name] = mask                                
                                                                                                                      
                
                # save the initial weight
                epoch_record = 0                                                    
                state = {'cycle_index': 0, 'lambda': lambda_base, 'state_dict': model.state_dict()}
                torch.save(state, filename + '_cycle_' + str(0) + '_model.pt')

                # train
                count_epoch = 0
                train_sign_nan_inf = 0
                train_sign_valid_snn_cnn = 0
                for cycle in range(cycles):
                    for epoch in range(epochs):
                        count_epoch = epoch + cycle*epochs
                        onoff_hessian = False
                        if epoch == epochs - 1:
                            onoff_hessian = True
                        epoch_record += 1
                        # train with mask
                        pred, pred_SNN, pred_CNN, target, train_loss, hessian_auto = lib.fine_train_node(model, algorithm, optimizer, estimation_input_pro,
                                                                    estimation_output_pro,
                                                                    train_batch_size,
                                                                    onoff_regularization,
                                                                    onoff_hessian, masks, count_epoch)       
                                                                
                        #jude on the SNN output, if equals to zero, then break
                        if torch.sum(torch.abs(pred_SNN))==0 or torch.sum(torch.abs(pred_CNN))== 0:
                            print("here")
                            print('sign 1: DISCONNECTED SNN OR CNN')
                            print('sign 1 epoch:', count_epoch, torch.sum(pred_SNN), torch.sum(pred_CNN))
                            train_sign_valid_snn_cnn = 1
                            break
                        #jude on the train_loss, if train_loss equals to inf/nan, then break                                                                    
                        if math.isnan(train_loss) or math.isinf(train_loss):
                            print('sign 4: TRAIN BREAK: NON/INF APPEARS')
                            print('sign 4 epoch:', count_epoch, train_sign_nan_inf)
                            train_sign_nan_inf = 1															
                            break                                                           
                        # lambda decay
                        if epoch_record % lambda_decay_epochs == 0:
                            for key, value in lambda_base.items():
                                for sub_key, sub_value in lambda_base[key].items():
                                    lambda_base[key][sub_key] *= 0.1
                            algorithm.lambda_base = lambda_base
                                                                    
                        scheduler.step()
                        
                        # Compute Hessian & update parameter
                        if cycle < cycles - 1:
                                algorithm.update(hessian_auto, onoff_regularization)        																										
#														         
                        del gamma           
                        del hessian_auto
                        
                        #dynamic pruning
                        for name, value in model.named_parameters():                                                 
                            if len(value.size())<4:
                                mask = torch.abs(algorithm.alpha[name+'_group']) > alpha_threshold
                                value.data = value.masked_fill(mask, 0)                                       
                                mask = torch.abs(algorithm.alpha[name + '_l1']) > alpha_threshold
                                value.data = value.masked_fill(mask, 0)                                       
                                mask = torch.abs(value) == 0
                                masks[name] = mask 
        
                
                        val_loss, val_pred, val_pred_diff = lib.validate(model, algorithm, validation_input_pro,                                                                                                        
                                                          validation_output_pro.to(
                                                              algorithm.device), std_y_train,
                                                          mean_y_train)        
                      # save each cycle
                        savemat(filename + '_cycle_' + str(cycle + 1) + '.mat',
                                {
                                    'train_pred_cycle_' + str(cycle + 1): pred.cpu().detach().numpy(),
                                    'train_Eva_cycle_' + str(cycle + 1): target.cpu().numpy(),
                                    'train_loss': torch.tensor(train_loss).cpu().numpy(),
                                    'val_loss': torch.tensor(val_loss).cpu().numpy() 
                                })        
                                                  
                        state = {'cycle_index': cycle + 1, 'train_acc': train_loss, 'val_acc': val_loss,
                                 'state_dict': model.state_dict()}

                        torch.save(state, filename + '_cycle_' + str(cycle + 1) + '_model.pt')        

                        if abs(val_loss-last_cycle_loss) < 1e-6: #if the val_loss of adjacent cycles are almost the same, we belive the Net has realized the convergence point
                            print('BREAK: CONVERGENCE')
                            break
                            
                        last_cycle_loss = val_loss

                        if math.isnan(val_loss) or math.isinf(val_loss) or train_sign_nan_inf == 1:
                            print('sign 2: BREAK: NON/INF APPEARS')
                            break
 

                        if train_sign_valid_snn_cnn == 1:
                            print('sign 1: DISCONNECTED SNN OR CNN')
                            break
   
                        # save the model.pt with smallest validation error
                        if val_loss < minimal_loss:
                            print('Best model saved.')
                            minimal_loss = val_loss
                            state = {'cycle_index': cycle + 1, 'train_acc': train_loss, 'val_acc': val_loss,
                                     'state_dict': algorithm.model.state_dict()}
                            torch.save(state, filename + '_best_model.pt')
                            
                            save_masks = {name.replace(".","_"): masks[name].cpu().detach().numpy() for name in masks}
                                                                            
                            savemat(filename + '_best_model.mat',
                                    {
                                        'train_pred': pred.cpu().detach().numpy(),
                                        'train_true': estimation_output_pro.cpu().numpy(),
                                        'minimal_val_loss': torch.tensor(
                                            minimal_loss).cpu().detach().numpy(),
                                        'masks':save_masks
                                    })
                                                                    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
