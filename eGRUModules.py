# Author: Justice Amoh
# Date: 09/29/2018
# Description: eGRU PyTorch Implementation

# Basic Imports
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


###################################
##    HELPER FUNCTIONS            #
###################################

def quantize(x,f=2):
    y = th.round(x*(2.**f))
    y = th.clamp(y,min=-f,max=f)
    y = y/(2.**f)
    return y    

def expquantize(x,n=2):
    lb = 2.**(-n)
    y = th.abs(x.clamp(min=-1,max=+1))
    y = th.log(y)/np.log(2.)                # Base 2 log
    y = th.round(y)                         # Integer exponents
    y = th.sign(x)*(2.**y)
    y[th.abs(y)<lb] = 0                     # Set weights < 2^(-n) to zero
    return y  

# Quantized LinearFxn
class QLinearFxn(Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        ctx.save_for_backward(input, weight, bias)
        wq = expquantize(weight)
        output  = input.mm(wq.t())

        if bias is not None:
            bq = expquantize(bias)
            output += bq.unsqueeze(0).expand_as(output)
        return output
       

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_variables
        grad_input = grad_weight = grad_bias = None

        # Propagate gradient as if no quantization
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias



###################################
##    eGRU & RNN MODULES          #
###################################

# Quantized Linear Module
class QLinear(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super(QLinear, self).__init__()
        self.input_features  = input_features
        self.output_features = output_features

        self.weight  = nn.Parameter(th.Tensor(output_features, input_features))
        nn.init.xavier_normal_(self.weight.data) # Initialize with Glorot Normal

        if bias:
            self.bias = nn.Parameter(th.Tensor(output_features))
            nn.init.constant_(self.bias.data, 0) # Glorot Init of bias
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        return QLinearFxn.apply(input, self.weight, self.bias)

    def getQweights(self):
        return expquantize(self.weight.data)


# Embedded GRU Cell
class eGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size,use_quant=True):
        super(eGRUCell, self).__init__()

        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.u_size      = input_size + hidden_size # concat x & h size
 
        # eGRU weights (quantized or not)
        if use_quant:
            self.weight_zx = QLinear(self.u_size, hidden_size) 
            self.weight_hx = QLinear(self.u_size, hidden_size)
        else:
            self.weight_zx = nn.Linear(self.u_size, hidden_size) 
            self.weight_hx = nn.Linear(self.u_size, hidden_size)

    def forward(self,x,state):
        u = th.cat((x,state),1)             # Concatenation of input & previous state
        za= F.softsign(self.weight_zx(u))
        z = (za + 1)/2
        g = F.softsign(self.weight_hx(u))   # candidate cell state
        h = (1 - z)*state + z*g
        return h



# Custom GRU Cell
class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUCell, self).__init__()

        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.u_size      = input_size + hidden_size # concat x & h size
 
        # GRU weights
        self.weight_rx = nn.Linear(self.u_size, hidden_size) 
        self.weight_zx = nn.Linear(self.u_size, hidden_size) 
        self.weight_hx = nn.Linear(self.u_size, hidden_size)

    def forward(self,x,state):
        u = th.cat((x,state),1)              # Concatenation of input & previous state 
        r = th.sigmoid(self.weight_rx(u))    # update gate
        z = th.sigmoid(self.weight_zx(u))    # update gate

        k = th.cat((x,r*state),1)

        g = th.tanh(self.weight_hx(k))       # candidate cell state
        h = (1 - z)*state + z*g
        return h



# Custom GRU Cell
class eRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size,use_quant=True):
        super(eRNNCell, self).__init__()

        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.u_size      = input_size + hidden_size # concat x & h size
 
        # RNN weights (quantized or not)
        if use_quant:
            self.weight_hx = QLinear(self.u_size, hidden_size)
        else: 
            self.weight_hx = nn.Linear(self.u_size, hidden_size)

    def forward(self,x,state):
        u = th.cat((x,state),1)             # Concatenation of input & previous state
        h = F.softsign(self.weight_hx(u))   # next state
        return h




