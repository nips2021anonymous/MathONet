import torch.nn.functional as F
import numpy as np
import torch
import copy
import torch.autograd as autograd
import math
import model as md


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def clean_data(estimation_input, estimation_output, validation_input, onoff_normalize_data):
    """this part is used to normalize data"""
    std_y_train, mean_y_train = 1, 0
    #normalize estimation_data
    # train_number = int(estimation_input.shape[0])
    X_train = estimation_input.to(DEVICE)
    y_train = estimation_output.to(DEVICE)
    X_test = validation_input.to(DEVICE)
    mean_X_train = torch.zeros(X_train.size(1),device=DEVICE)
    std_X_train = torch.ones(X_train.size(1),device=DEVICE)
    if onoff_normalize_data == True:
        std_X_train=np.std(X_train.cpu().numpy(),0,ddof=1)
#        std_X_train[std_X_train==0]=1
        std_X_train=std_X_train.reshape(X_train.shape[1],1)
        mean_X_train=np.mean(X_train.cpu().numpy(),0)   
#        print('libcheck2:',mean_X_train) 
#        print('libcheck aaaaaaa  111:',mean_X_train)	
#        print('libcheck aaaaaaa  222:',std_X_train)	
    
        mean_X_train = torch.tensor(mean_X_train,device=DEVICE)
        std_X_train = torch.tensor(std_X_train,device=DEVICE)
		
        for i in range(X_train.size(1)):
            X_train.t()[i]=(X_train.t()[i]-mean_X_train[i])/std_X_train[i]
        for i in range(X_test.size(1)):
            X_test.t()[i] = (X_test.t()[i] - mean_X_train[i]) / std_X_train[i]

        std_y_train=np.std(y_train.cpu().numpy(),ddof=1) #ddof=1 used to make the result is the same as MATLAB
        mean_y_train=np.mean(y_train.cpu().numpy())
        mean_y_train = torch.tensor(mean_y_train,device=DEVICE)

        std_y_train = torch.tensor(std_y_train,device=DEVICE)
		
		
#        print('libcheck aaaaaaa  333:',mean_y_train)	
#        print('libcheck aaaaaaa  444:',std_y_train)  
		
		
        y_train_normalized=(y_train-mean_y_train)/std_y_train
        y_train=y_train_normalized.reshape(y_train.size())
    return X_train.clone().detach(), y_train.clone().detach(), X_test.clone().detach(), std_y_train, mean_y_train, mean_X_train, std_X_train


def trainData_normalize(estimation_input, estimation_output, onoff_normalize_data):
    
    if onoff_normalize_data:
        mean_X_train = torch.mean(estimation_input).cpu()
#        print('libcheck ddddddd  111:',mean_X_train)      
    
        std_X_train = torch.std(estimation_input, unbiased=True).cpu()
    	
#        print('libcheck ddddddd  222:',std_X_train)  
    	
        mean_y_train = torch.mean(estimation_output).cpu()
        std_y_train = torch.std(estimation_output, unbiased=True).cpu()
        
#        print('libcheck ddddddd  333:',mean_y_train)	
#        print('libcheck ddddddd  444:',std_y_train)  
    	
        X_train = (estimation_input.clone()-mean_X_train)/std_X_train
        y_train = (estimation_output.clone()-mean_y_train)/std_y_train
    else:
        X_train = estimation_input
        y_train = estimation_output
        mean_X_train, std_X_train = 0,1
        mean_y_train, std_y_train = 0,1

    return X_train, y_train, mean_X_train, std_X_train, mean_y_train, std_y_train

def testData_normalize(validation_input, mean_X_train, std_X_train):
    X_test = (validation_input-mean_X_train)/std_X_train
#    y_test = (validation_output-mean_y_train)/std_y_train
    return X_test




def divide_data(input_feature,target, percentage):
    """this part is used to normalize data"""
    train_number = int(np.round(input_feature.shape[0] * percentage))
    X_train = input_feature[:train_number]
    y_train = target[:train_number]              
    X_test=input_feature[train_number:]
    y_test=target[train_number:]            
    return X_train,y_train,X_test,y_test

def select_train_data(input_feature, target, percentage):
    """this part is used to normalize data"""
    train_number = int(np.round(input_feature.shape[0] * percentage))
    X_train = input_feature[:train_number]
    y_train = target[:train_number]                   
    return X_train,y_train

def generate_prediction_data(estimation_input, estimation_output, validation_input, validation_output, lag_input, lag_output):
    
    #decide new dimension for validation data
    if lag_input < lag_output: # the following function is under the assumption that lag_input is bigger than lag_output
        raise ValueError("lag_input should be bigger than lag_output")
        
#    print('lib_check:',validation_output.size())
    new_val_sample_numbers = validation_input.size(0) - lag_input
    new_val_feature_numbers = lag_output*validation_output.size(1)+(lag_input+1)*validation_input.size(1)
	
    print("check lib:", new_val_sample_numbers, new_val_feature_numbers)	
    target_test_input= torch.ones((new_val_sample_numbers, new_val_feature_numbers),device=DEVICE)
    target_test_output = torch.ones((new_val_sample_numbers, validation_output.size(1)),device=DEVICE)
          
    for i in range(target_test_input.size(0)):
#        output_index = lag_output+i   
        input_index = lag_input + i                  

        target_test_input[i] = torch.cat((validation_input[i:input_index+1].reshape(-1), validation_output[input_index-lag_output:input_index].reshape(-1)),0).view(target_test_input[i].size())
        target_test_output[i] = validation_output[input_index]  
        
#   for training data:
    new_train_sample_numbers = estimation_input.size(0) - lag_input
    new_train_feature_numbers = lag_output*estimation_output.size(1)+(lag_input+1)*estimation_input.size(1)
    target_train_input= torch.ones((new_train_sample_numbers, new_train_feature_numbers),device=DEVICE)
    target_train_output = torch.ones((new_train_sample_numbers, estimation_output.size(1)),device=DEVICE)      
    for i in range(target_train_input.size(0)):
#        output_index = lag_output+i  
        input_index = lag_input + i                  
        target_train_input[i] = torch.cat((estimation_input[i:input_index+1].reshape(-1), estimation_output[input_index-lag_output:input_index].reshape(-1)),0).view(target_train_input[i].size())
#        print(estimation_output.size())
        target_train_output[i] = estimation_output[input_index]      
    return target_train_input,target_train_output,target_test_input,target_test_output


def fine_train(model, algorithm, optimizer, estimation_input_pro, estimation_output_pro, \
          train_batch_size, onoff_regularization, \
          onoff_hessian, masks,count_epoch):
	
    global pred, y
    hessian_auto = {}
    start=0
    end=train_batch_size-1
    if estimation_input_pro.shape[0]>=end:  		
        while end<estimation_input_pro.shape[0]:
            x = estimation_input_pro[start:end]
            y = estimation_output_pro[start:end]
            x, y = x.to(algorithm.device), y.to(algorithm.device)
#            with autograd.detect_anomaly():                
            pred = model(x)        
            
#            print('fuck train?', torch.max(pred))

#            for name, value in model.named_parameters():   
#                print('before:', name, value, value.grad)    

            
            loss_prediction, loss = algorithm.loss_cal(pred, y, onoff_regularization)
            
            if math.isnan(loss_prediction) or math.isinf(loss_prediction):
                print('sign 4: TRAIN BREAK: NON/INF APPEARS')
                start += train_batch_size
                end += train_batch_size
                pass  
            else:         
                optimizer.zero_grad()    
                loss.backward(create_graph=True)            
#                print('hessian start sign:', onoff_hessian)
                if onoff_hessian == True:			
#                    print('hessian start:')
                    second_order_derivative(model)
                    for name,param in model.named_parameters():
                        hessian_auto[name] = param.grad2
                optimizer.step()                     
                for name, value in model.named_parameters():                                           
                    value.data = value.masked_fill(masks[name], 0)             
                start += train_batch_size
                end += train_batch_size
                if end < estimation_input_pro.shape[0] and end+train_batch_size > estimation_input_pro.shape[0]:
                    end = estimation_input_pro.shape[0]-1
					
#            for name, value in model.named_parameters():   
#                print('after:', name, value, value.grad)    
					
					
    else:
        x = estimation_input_pro
        y = estimation_output_pro
        x, y = x.to(algorithm.device), y.to(algorithm.device)
#        with autograd.detect_anomaly():                
        pred = model(x)
        loss_prediction, loss = algorithm.loss_cal(pred, y, onoff_regularization)
		
#        for name, value in model.named_parameters():   
#            print('before:', name, value, value.grad)    
#		

        if math.isnan(loss_prediction) or math.isinf(loss_prediction):
            print('sign 3: TRAIN BREAK: NON/INF APPEARS')
            pass  
        else:  	
    #        loss = 10*loss
            optimizer.zero_grad()
            loss.backward(create_graph=True)		
            
            if onoff_hessian == True:			
                
#                print('hessian start:')

                
                second_order_derivative(model)
                for name,param in model.named_parameters():
                    hessian_auto[name] = param.grad2				
            
            optimizer.step()
    				
            for name, value in model.named_parameters():                                             
                value.data = value.masked_fill(masks[name], 0)  


#            for name, value in model.named_parameters():   
#                print('after:', name, value, value.grad)    


    if count_epoch%200 == 0:
		   print('Epoch:', count_epoch, 'training loss',loss_prediction.item())
    return pred, y, loss_prediction, hessian_auto
#
#def second_order_derivative(model):
#    '''
#    optimizer is in accordance to named_parameters
#    '''
#    memory={}
#    for name,param in model.named_parameters():      
#        print('hessian calculation:',name, param)
#        if param.grad.requires_grad:
#            
#            torch.autograd.set_detect_anomaly(True)
#            grad=autograd.grad(param.grad.sum(),param,retain_graph=True)[0].to(model.save_device)
#        else:
#            grad=torch.zeros_like(param).to(model.save_device)
#        memory[name]=grad
#        
#    for name,param in model.named_parameters():
#        param.grad2=memory[name]
#    return 


def second_order_derivative(model):
    '''
    optimizer is in accordance to named_parameters
    '''
    memory={}
    for name,param in model.named_parameters():      
        if param.grad is not None and param.grad.requires_grad:
            torch.autograd.set_detect_anomaly(True)
            grad=autograd.grad(param.grad.sum(),param,retain_graph=True)[0].to(model.save_device)
        else:
            grad=torch.zeros_like(param).to(model.save_device)
        memory[name]=grad
    
    for name,param in model.named_parameters():
        param.grad2=memory[name]
    return 



def grad_hessian_calculate(model, algorithm, optimizer, estimation_input_pro, estimation_output_pro, \
          train_batch_size, onoff_regularization, \
          masks,count_epoch):
	
    global pred, y
    hessian_auto = {}
#    grad_auto = {}
    start=0
    end=train_batch_size-1
    if estimation_input_pro.shape[0]>=end:  		
        while end<estimation_input_pro.shape[0]:
            x = estimation_input_pro[start:end]
            y = estimation_output_pro[start:end]
            x, y = x.to(algorithm.device), y.to(algorithm.device)
            pred = model(x)          
            loss_prediction, loss = algorithm.loss_cal(pred, y, onoff_regularization)
            optimizer.zero_grad()    
            loss.backward(create_graph=True)            
#            if onoff_hessian == True:			
            second_order_derivative(model)
            for name,param in model.named_parameters():
                hessian_auto[name] = param.grad2
#                grad_auto[name] = param.grad				

            for name, value in model.named_parameters(): 
#                print("check:", name, hessian_auto[name].is_cuda, masks[name].is_cuda, hessian_auto[name].size(),masks[name].size(), value.size())                                          
                hessian_auto[name].data = torch.tensor(hessian_auto[name]).masked_fill(masks[name], 0)  
#                grad_auto[name].data = grad_auto[name].masked_fill(masks[name], 0)  

            optimizer.step()                     


            start += train_batch_size
            end += train_batch_size
            if end < estimation_input_pro.shape[0] and end+train_batch_size > estimation_input_pro.shape[0]:
                end = estimation_input_pro.shape[0]-1
    else:
        x = estimation_input_pro
        y = estimation_output_pro
        x, y = x.to(algorithm.device), y.to(algorithm.device)
        pred = model(x)
        loss_prediction, loss = algorithm.loss_cal(pred, y, onoff_regularization)
#        loss = 10*loss
        optimizer.zero_grad()
        loss.backward(create_graph=True)	

#        if onoff_hessian == True:			
        second_order_derivative(model)
        for name,param in model.named_parameters():
            hessian_auto[name] = param.grad2
#            grad_auto[name] = param.grad				
				
        for name, _ in model.named_parameters():                                             
            hessian_auto[name].data = hessian_auto[name].masked_fill(masks[name], 0)  
#            grad_auto[name].data = grad_auto[name].masked_fill(masks[name], 0)  
#
        optimizer.step()


    if count_epoch%200 == 0:
		   print('Epoch:', count_epoch, 'training loss',loss_prediction.item())
    return hessian_auto
#    return grad_auto, hessian_auto


def norm_restore(inputs, std_y_train, mean_y_train):
    """this code is used for testing,only exeacuting the forward process"""
    # change the input and output as columns vector
    
#    print(std_)
    
    PreOutputs = inputs * std_y_train + mean_y_train
    return PreOutputs

def validate(model, algorithm, validation_input, validation_output, std_y_train, mean_y_train):
    with torch.no_grad():
        x = validation_input
        y = validation_output
        x, y = x.to(algorithm.device), y.to(algorithm.device)
        
#        print("111",x[0:1])
        
        val_pred_diff = model(x)
        val_pred =  norm_restore(val_pred_diff,std_y_train,mean_y_train)
        
#        print('222', val_pred)
        
        val_loss = F.mse_loss(val_pred, validation_output.reshape(val_pred.size()).float()).item()
        print('validation loss',val_loss)    
    return val_loss, val_pred, val_pred_diff


def validate_check_v3(model, algorithm, validation_input, validation_output, std_y_train, mean_y_train):
    with torch.no_grad():
        x = validation_input
        y = validation_output
        x, y = x.to(algorithm.device), y.to(algorithm.device)
        val_pred_diff, mean_pn, mean_fc = model(x)
        val_pred = norm_restore(val_pred_diff,std_y_train,mean_y_train)
        val_loss = F.mse_loss(val_pred, validation_output.reshape(val_pred.size()).float()).item()
        print('validation loss',val_loss)    
    return val_loss, val_pred, val_pred_diff, mean_pn, mean_fc

#def weight_overall_sparsity(weight):
#    row, column = weight.size()
#    amount_elements = row * column
#    amount_zero = int((weight == 0).sum())
#    print('weight sparsity', 1 - float(amount_zero / amount_elements))
#    return 1 - float(amount_zero / amount_elements)

#def structure_sparsity(weight):
#    weight_in = weight
#    weight_out = weight.t()
#    row, column = weight.size()
#    amount_input = row
#    amount_output = column
#    weight_norm_in = torch.norm(weight_in, 2, 1)
#    weight_norm_out = torch.norm(weight_out, 2, 1)
#    amount_zero_in = torch.nonzero(weight_norm_in.data).size(0)
#    amount_zero_out = torch.nonzero(weight_norm_out.data).size(0)
#    return float(amount_zero_in / amount_input), float(amount_zero_out / amount_output)

def model_overall_sparsity(model):
    amount_zero = 0
    amount_total = 0
    for name, param in model.named_parameters():
#        if 'weight' in name:
        if len(param.size())>1:            
            row, column = param.size()
            tmp_amount_elements = row * column
            tmp_amount_zero = (param == 0).sum()
            amount_total += tmp_amount_elements
            amount_zero += int(tmp_amount_zero)
        else:
            row = param.size(0)
            tmp_amount_elements = int(row)
            tmp_amount_zero = (param == 0).sum()
#            print('fuck',tmp_amount_elements)
            amount_total += tmp_amount_elements
            amount_zero += int(tmp_amount_zero)            
    return 1 - float(amount_zero / amount_total)

def sim_validate(model, algorithm, validation_input, validation_output, std_y_train, mean_y_train, onoff_diff):
    with torch.no_grad():
        x = validation_input.unsqueeze(0)
        y = validation_output
        x, y = x.to(algorithm.device), y.to(algorithm.device)
        
#        print('sim validate1:',x.size())

        
        val_pred_diff = model(x)

#        print('sim validate2:',val_pred_diff.size())
        
        if onoff_diff == True:
            delay_distance = validation_output.size(1)-val_pred_diff.size(0)
            val_pred = val_pred_diff + validation_output.t()[delay_distance-1:validation_output.size(1)-1]
        else:          
            val_pred = val_pred_diff
        
#        print('sim validate3:',val_pred_diff.size(), val_pred.size())
        val_pred = norm_restore(val_pred, std_y_train, mean_y_train)
        val_loss = F.mse_loss(val_pred, validation_output.reshape(val_pred.size()).float()).item()
        # print('sim cur loss:', val_loss, validation_output, val_pred)

        # val_loss = F.mse_loss(val_pred, validation_output.float()).item()
    return val_pred,val_loss





def update_next_input_original(current_input, next_input, last_prediction, lag):
    #update the input data
    current_input[0:lag+1] = next_input[0:lag+1]
#    print('fine check1:', current_input)
    current_input[lag+1:-1] = current_input.clone()[lag+2:]
#    print('fine check2:', current_input)
#    print('fine check3:', current_input[-1], last_prediction)
    current_input[-1] = last_prediction
#    print('fine check4:', current_input)
    return current_input



def predict_distribution(model, algorithm, inputs, outputs, masks, gamma, hessian,
                         std_y_train, mean_y_train, repeat=10000):
    """ Predictive distribution mode calculation by MC integration """
    model.eval()
    with torch.no_grad():
        weights = sample_posterior_weights(model, hessian, gamma, repeat)

        # Implement the validation process repeatly
        uncertainty_outputs = []
        uncertainty_loss = []
        for rep in range(repeat):
            # Set the sampled weights
            for name,param in model.named_parameters():
#                param = model.weights[name]
#                if 'linear' in name:
                param.data = torch.tensor(weights[name][:, :, rep].reshape(param.size())).to(DEVICE)
                param.data = param.masked_fill(masks[name], 0)  
                
#                hessian_auto[name].data = hessian_auto[name].masked_fill(masks[name], 0)  


            val_loss, val_pred,_ = validate(model, algorithm, inputs, outputs, std_y_train, mean_y_train)

            uncertainty_outputs.append(val_pred)
            uncertainty_loss.append(val_loss)

        # Calcule the mean/variance of the predicted outputs
        uncertainty_outputs_reshape = torch.cat(uncertainty_outputs, 1)
        val_pred_mean = torch.mean(uncertainty_outputs_reshape, 1)
        val_pred_std = torch.std(uncertainty_outputs_reshape, 1)

        # val_pred_mode, _ = torch.mode(uncertainty_outputs_reshape, 1)

    return val_pred_mean, val_pred_std

def simulate_distribution(model, inputs, outputs, masks, gamma, hessian,
                         std_y_train, mean_y_train, lag_in, lag_out,
                          repeat=10000):
    """ Predictive distribution mode calculation by MC integration """
    model.eval()
    with torch.no_grad():
        weights = sample_posterior_weights(model, hessian, gamma, repeat)

        # Implement the validation process repeatly
        uncertainty_outputs = []
        uncertainty_loss = []
        for rep in range(repeat):
            # Set the sampled weights
            for name in model.names:
                param = model.weights[name]
#                if 'linear' in name:
                param.data = weights[name][:, :, rep].reshape(param.size())
                param.data = param.masked_fill(~masks[name], 0)

            val_loss, val_sim = validate_sim(model, inputs, outputs,
                                              std_y_train, mean_y_train,
                                              lag_in, lag_out)

            if np.isnan(val_loss) or np.isinf(val_loss) or val_loss>100:
                pass
            else:
                uncertainty_loss.append(val_loss)
                uncertainty_outputs.append(val_sim)

        # Calcule the mean/variance of the predicted outputs
        uncertainty_outputs_reshape = torch.cat(uncertainty_outputs, 1)
        val_pred_mean = torch.mean(uncertainty_outputs_reshape, 1)
        val_pred_std = torch.std(uncertainty_outputs_reshape, 1)
    return val_pred_mean, val_pred_std


def sample_posterior_weights(model, hessian, gamma, repeat):
    means = {}
    variances = {}
    weights = {}
    with torch.no_grad():
        # Sample Connection weights from posterior
        for name,param in model.named_parameters():
#            param = model.weights[name]
#            if 'linear' in name:
            
            if int(torch.max(torch.tensor(param.size()))) == 1:
                second_size = param.size(0)
            else:
                second_size = param.size(1)
            
            variances[name] = 1 / (torch.abs(hessian[name])+ torch.tensor(1 / gamma[name+'_sum']).to(DEVICE))
            means[name] = param
            weights[name] = torch.zeros((param.size(0), second_size, repeat))

            # Sample weights from the posterior distribution
            nonzero_index_list = torch.nonzero(param)
            
            if int(torch.max(torch.tensor(param.size()))) == 1:
        #        print('1',name,param.size())
                mean =  means[name]
                std =  variances[name]     		
                weights[name] = torch.normal(mean.item(), std.item(), size = (1,1,repeat))

            
            else:
                for index in nonzero_index_list:
                    mean = means[name][index[0], index[1]].item()
                    std = variances[name][index[0], index[1]].item()
                    if std==0 or mean <=1e-4  : std = 1e-10
                    weights[name][index[0], index[1]] = torch.normal(
                        mean, std, size=(1, repeat))
    return weights

def validate_sim(model, inputs, outputs, std_y_train, mean_y_train,
                 lag_in, lag_out):
    model.eval()
    with torch.no_grad():
        layer_n = len(model.layers)-2
        lag = max([lag_in,lag_out])
        simloop_num = outputs.size(0)
        sim_inputs = torch.zeros(1,lag_in+lag_out+1, device=model.save_device)
        sim_outputs = outputs[lag - lag_out:lag].clone()
        val_sim = torch.zeros(simloop_num, outputs.shape[1],
                              device=model.save_device)
        val_sim[0:lag] = restore_data(outputs[0:lag], std_y_train,
                                      mean_y_train)
        if isinstance(model, md.RNNNetwork):
            hidden = torch.zeros(layer_n,  1, model.hiddenUnits,
                                 device=model.save_device, requires_grad=False)
            c = torch.zeros(layer_n, 1, model.hiddenUnits,
                            device=model.save_device, requires_grad=False)
        for sim_iter in range(lag, simloop_num):
            sim_inputs[:,0:lag+1] = inputs[sim_iter - lag:sim_iter + 1,:].t()
            sim_inputs[:,-lag_out:] = sim_outputs.t()
            if isinstance(model, md.MLPNetwork):
                sim = model(sim_inputs)
            elif isinstance(model, md.RNNNetwork):
                sim, hidden[:,0,:], c[:,0,:] = model(sim_inputs, hidden, c)
            val_sim[sim_iter] = restore_data(sim, std_y_train, mean_y_train)
            sim_outputs[0:lag_out-1] = sim_outputs[1:lag_out].clone()
            sim_outputs[-1] = sim
        sim_real = restore_data(outputs, std_y_train, mean_y_train)
        sim_loss = torch.sqrt(torch.nn.functional.mse_loss(sim_real, val_sim))
    return sim_loss.item(), val_sim

def restore_data(inputs, std_y_train, mean_y_train):
    """this code is used for testing,only exeacuting the forward process"""
    with torch.no_grad():
        pred_outputs = inputs * std_y_train + mean_y_train
    return pred_outputs


def lay_down_model_params(dic):
    """this code is used for testing,only exeacuting the forward process"""
    var_param = []
    act_param = []
    constant_param = []
    final_linear_param = []
    for key,value in dic.items():
        value = torch.abs(torch.tensor(value))
        if 'VarNet' in key:
            var_param.append(value[:])
        if 'ActNet' in key:
            act_param.append(value[:])
        if 'constant' in key:
            constant_param.append(value[:])
        if 'linear_P' in key:
            final_linear_param.append(value[:])      
#    print("check here  ggg",var_param[:])

    var_param = torch.cat((var_param[:]),1)
    act_param = torch.cat((act_param[:]),0)
    if len(constant_param) > 0:
        constant_param = torch.cat((constant_param),1)
    final_linear_param = torch.cat((final_linear_param),1)
#    lay_down_params = var_param + act_param + constant_param + final_linear_param    
#    lay_down_params = torch.cat((var_param,act_param,constant_param,final_linear_param),0)

#    print("check here",var_param.size(), act_param.size(), final_linear_param.size())

    lay_down_params = torch.cat((var_param,act_param.unsqueeze(0),final_linear_param),1)

#    print("check here",type(lay_down_params), lay_down_params)
    return lay_down_params
