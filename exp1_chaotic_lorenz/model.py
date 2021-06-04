import torch.nn as nn
import torch
import copy
import torch.nn.functional as F

class PolyNet(nn.Module):
    """"""
    def __init__(self, input_channels, save_device=torch.device('cuda')):
        super(PolyNet, self).__init__()
        self.save_device = save_device
        self.input_channels = input_channels
        self.linear_0 = torch.nn.Linear(in_features=self.input_channels + 1, out_features=1, bias=False)

    def forward(self, x):
        x_new = torch.cat((x, torch.ones(x.size(0), 1).to(self.save_device)), 1)
        x = x_new.float().to(self.save_device)
        x = self.linear_0(x)		
        del x_new
        
        return x

 
class OperNet(nn.Module):
    """"""
    def __init__(self, unary_names, save_device=torch.device('cpu')):
        super(OperNet, self).__init__()
        self.unary_names = unary_names
        for op_name in unary_names:
           if 'sqrt' in op_name:
               self.unary_sqrt = nn.Parameter(torch.ones(1).to(save_device))            
           if 'square' in op_name:
               self.unary_square = nn.Parameter(torch.ones(1).to(save_device))            
           if 'cube' in op_name:
               self.unary_cube = nn.Parameter(torch.ones(1).to(save_device)) 
           if 'ident' in op_name:
               self.unary_ident = nn.Parameter(torch.ones(1).to(save_device))
           if 'sin' in op_name:
               self.unary_sin = nn.Parameter(torch.ones(1).to(save_device))
           if 'cos' in op_name:
               self.unary_cos = nn.Parameter(torch.ones(1).to(save_device))
           if 'exp' in op_name:
               self.unary_exp = nn.Parameter(torch.ones(1).to(save_device))
           if 'log' in op_name:
               self.unary_log = nn.Parameter(torch.ones(1).to(save_device))

    def forward(self, x):
        x_ident = 0
        x_sin = 0
        x_cos = 0
        x_exp = 0
        x_log = 0
        x_sqrt = 0
        x_square = 0
        x_cube = 0
       
        for op_name in self.unary_names:       
            if  op_name == 'sqrt':
                x_sqrt = self.unary_sqrt*torch.sqrt(torch.abs(x))

            if op_name == 'square':
                x_square = self.unary_square*torch.pow(torch.abs(x),2)  
                
            if op_name =='cube':
                x_cube = self.unary_cube*torch.pow(x,3)

            if op_name == 'ident':
                x_ident = self.unary_ident * x
                
            if op_name == 'sin':
                if self.unary_sin != 0:
                    x_sin = x
                    x_sin = self.unary_sin * torch.sin(x_sin)

            if op_name == 'cos':
                if self.unary_cos != 0:
                    x_cos = x
                    x_cos = self.unary_cos * torch.cos(x_cos)

            if op_name == 'exp':
                #in case the exp make the result become inf
                if self.unary_exp != 0:

                    if torch.max(x)>88:
                        self.unary_exp.data = 0*self.unary_exp
                        x_exp =  x                    
                    else:
                        x_exp = x
                        x_exp = self.unary_exp * torch.exp(x_exp)
                
            if op_name == 'log':
                if self.unary_log != 0:
                    x_log = x
                    x_log = torch.clamp(torch.abs(x_log), min=1e-10, max=1e10)
                    x_log = self.unary_log * torch.log(x_log)
        x = x_ident + x_sin + x_cos + x_exp + x_log + x_sqrt + x_square + x_cube      
        del x_ident, x_sin, x_cos, x_exp, x_log, x_sqrt, x_square, x_cube        
        return x


class MathONet(nn.Module):
    def __init__(self, MathONet_layers, unary_names_lists, save_device=torch.device('cuda')):
        super(MathONet, self).__init__()
        self.save_device = save_device
        self.MathONet_layers = MathONet_layers

        if len(self.MathONet_layers) > 1:        
            self.unary_names = unary_names_lists
            self.input_features = MathONet_layers[0]
            self.amount_layers = len(MathONet_layers)-2 
                    
            self.length_PolyNets = sum([MathONet_layers[i]*MathONet_layers[i+1] for i in range(len(MathONet_layers)-2)])                          
            self.PolyNets = nn.ModuleList([PolyNet(self.input_features, save_device=save_device) for i in range(self.length_PolyNets)])
    
            self.length_OperNets = sum([MathONet_layers[i+1] for i in range(len(MathONet_layers)-2)])                          
            self.OperNets = nn.ModuleList([OperNet(self.unary_names, save_device=save_device) for i in range(self.length_OperNets)])
			
            self.linear_P = torch.nn.Linear(in_features=self.MathONet_layers[-2], out_features=self.MathONet_layers[-1],bias=False)        
            self.constant = torch.nn.Linear(in_features=1, out_features=self.MathONet_layers[1], bias=False)
    def forward(self, x):        
   
        x1 = torch.tensor([0.]).to(self.save_device)
      
        if len(self.MathONet_layers) > 1:
            x1 = copy.deepcopy(x)                      
            x_constant = torch.ones(x.size(0), 1).to(self.save_device)        
            #update parameter in PolyNets
            PolyNets = []
            for m in self.PolyNets:
                PolyNets.append(m(x1))            
            #divide PolyNets into different layers
            counter = self.length_PolyNets
            start_act_index = 0
            while counter>0:
                for index_layer in range(self.amount_layers):
                    #find how many tmp_PolyNets are there in current layer
                    tmp_PolyNets = PolyNets[:self.MathONet_layers[index_layer]*self.MathONet_layers[index_layer+1]]
                    counter -= len(tmp_PolyNets)
                    tmp_PolyNets = torch.cat(tmp_PolyNets,1)                   
                    start_index = 0
                    tmp_layer_output = []
                    for index_neuron in range(self.MathONet_layers[index_layer+1]): 
                        short_weight_tmp = tmp_PolyNets[:,start_index:start_index+self.MathONet_layers[index_layer]]
                        start_index += self.MathONet_layers[index_layer]                  
                        tmp_layer_output.append(torch.sum(x1*short_weight_tmp,1).unsqueeze(1))              
                    sum_tmp_layer = x1 = torch.cat(tmp_layer_output,1)      
                    if index_layer == 0:
                        constant = self.constant(x_constant)
                        sum_tmp_layer = x1 = x1+constant    
                   
                    #find how many tmp_OperNets are there in current layer
                    tmp_OperNets = self.OperNets[start_act_index:start_act_index+self.MathONet_layers[index_layer+1]]
                    start_act_index += self.MathONet_layers[index_layer+1]
                    
                    index_n = 0
                    tmp_act_layer = []
                    for tmp_act in tmp_OperNets:
                        tmp_act_layer.append(tmp_act(sum_tmp_layer[:,index_n]).unsqueeze(1))
                        index_n += 1                                      
                    x1 = torch.cat(tmp_act_layer,1)                          
                    del PolyNets[:self.MathONet_layers[index_layer]*self.MathONet_layers[index_layer+1]]                                    
            x1 = self.linear_P(x1)

        result = x1.to(self.save_device)           
        return result
     
            
                
       

