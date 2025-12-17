# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from torch import nn
from torchdiffeq import odeint_adjoint as odeint

class WODE(nn.Module):
    def __init__(self, odefunc, odeint_rtol = 1e-3, odeint_atol = 1e-4, method = 'rk4', weight_method = 'softmax'):
        super().__init__()
        
        self.odefunc = odefunc
        self.odeint_rtol = odeint_rtol # relative tolerance
        self.odeint_atol = odeint_atol # absolute tolerance
        self.method = method # integrator method 
        self.weight_method = weight_method
    
    def compute_normalized_exp_matrix(self, a, b):
        """
        Compute an L x P matrix where each element (i, j) is:
            exp(1 / abs(a_i - b_j)) / sum_j(exp(1 / abs(a_i - b_j)))
        
        Args:
            a (torch.Tensor): Tensor of shape (L,)
            b (torch.Tensor): Tensor of shape (P,)
        
        Returns:
            torch.Tensor: Tensor of shape (L, P)
        """
        # Ensure a and b are 1D tensors
        a = a.view(-1, 1)  # shape: (L, 1)
        b = b.view(1, -1)  # shape: (1, P)
    
        # Compute absolute difference
        diff = torch.abs(a - b)  # shape: (L, P)
    
        # Avoid division by zero by adding a small epsilon
        epsilon = 1e-8
        inv_diff = 1.0 / (diff + epsilon)
    
        # Apply exponential
        exp_matrix = torch.exp(inv_diff)  # shape: (L, P)
        
        exp_norm = exp_matrix.sum(dim=0, keepdim=False) # (P)
        exp_norm = exp_norm.unsqueeze(dim = 0).expand(a.shape[0], b.shape[1])
    
        # Normalize along each row (i.e., across columns j)
        normalized_matrix = exp_matrix / exp_norm
    
        return normalized_matrix.float()


    def forward(self, x, his_timeslot, pre_timeslot):
        """
            x: hidden state vector for q or k with shape [B, L, E]
            his_timeslot: historical time slot with shape [L]
            pre_timeslot: predicted time slot with shape [P]

            Return: predicted output at pre_timeslot [B, P, E]
        """
        
        # for example x = Q = [q1, q2, ..., qL]       
        B, L, E = x.shape
        P = pre_timeslot.shape[0]
                      
        y = torch.zeros(B, L, P, E, device = x.device, dtype = x.dtype)
        
        for idx in range(L):
            current_t = his_timeslot[idx]
                                  
            current_state = x[:, idx, :] # (B, E)
                      
            # only predict future           
            z0 = odeint(self.odefunc, current_state, torch.cat([current_t.unsqueeze(dim = 0), pre_timeslot]), rtol = self.odeint_rtol, atol = self.odeint_atol, 
                                  method = self.method)[1 :] # (P, B, E)                  
            
            z0 = z0.permute(1, 0, 2) # (B, P, E)
            y[:, idx, :, :] = z0
        
        if self.weight_method == 'softmax':
            weight = self.compute_normalized_exp_matrix(a = his_timeslot, b = pre_timeslot) # (L, P)
            weight = weight.unsqueeze(0).unsqueeze(-1)
            weight = weight.expand(B, weight.shape[1], weight.shape[2], E) # (B, L, P, E)   
        elif self.weight_method == 'uniform': 
            weight = 1. / L * torch.ones(B, L, P, E, device = x.device, dtype = x.dtype)
            weight = weight.float()
                
        y = weight * y # (B, L, P, E)
            
        y = y.sum(dim = 1, keepdim = False) # (B, P, E)
               
        return y

