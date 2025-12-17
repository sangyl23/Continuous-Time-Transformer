# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from torch import nn
from torchdiffeq import odeint_adjoint as odeint

class InterpLinear(nn.Module):
    def __init__(self):
        super().__init__()

        self.gauss_weight = {
            1: torch.tensor([2]),
            2: torch.tensor([1, 1]),
            3: torch.tensor([0.55555, 0.88888, 0.55555]),
            4: torch.tensor([0.34785, 0.65214, 0.65214, 0.34785]),
            5: torch.tensor([0.23692, 0.47863, 0.56888, 0.47863, 0.23692])
        }
        self.gauss_legendre = {
            1: torch.tensor([0]),
            2: torch.tensor([-0.57735, 0.57735]),
            3: torch.tensor([-0.77459, 0, 0.77459]),
            4: torch.tensor([-0.86113, -0.33998, 0.33998, 0.86113]),
            5: torch.tensor([-0.90618, -0.53846, 0, 0.53846, 0.90618])
        }


    def forward(self, x, timeslot):
        """
            x: hidden state vector for q or k or v with shape [B, L, H, E]

            Return: 1/2 * q(t) or 1/2 * q(t) or 1/2 * v(t), [B, L, L, H, E, 2]
        """
        
        # for example x = Q = [q1, q2, ..., qL]
        
        # x0 form at dims [L, L]
        # [q1, q1, q1, ...]
        # [q2, q2, q2, ...]
        # ...
        # [qL, qL, qL, ...]
        x0 = x.unsqueeze(dim = 2).repeat(1, 1, x.shape[1], 1, 1).unsqueeze(dim = -1) # (B, L, L, H, E, 1)
        
        # x1 form at dims [L, L]
        # [q1, q2, ..., qL]
        # [q1, q2, ..., qL]
        # ...
        # [q1, q2, ..., qL]
        x1 = x.unsqueeze(dim = 1).repeat(1, x.shape[1], 1, 1, 1).unsqueeze(dim = -1) # (B, L, L, H, E, 1)
        
        # x form at dims [L, L, 2]
        # [(q1, q1), (q1, q2), ..., (q1, qL)]
        # [(q2, q1), (q2, q2), ..., (q2, qL)]
        # ...
        # [(qL, q1), (qL, q2), ..., (qL, qL)]
        x = torch.cat([x0, x1], dim = -1) # (B, L, L, H, E, 2)
        
        return x


class ODELinear(nn.Module):
    def __init__(self, odefunc, odeint_rtol = 1e-3, odeint_atol = 1e-4, method = 'rk4'):
        super().__init__()
        
        self.gauss_weight = {
            1: torch.tensor([2]),
            2: torch.tensor([1, 1]),
            3: torch.tensor([0.55555, 0.88888, 0.55555]),
            4: torch.tensor([0.34785, 0.65214, 0.65214, 0.34785]),
            5: torch.tensor([0.23692, 0.47863, 0.56888, 0.47863, 0.23692])
        }
        
        self.odefunc = odefunc
        self.odeint_rtol = odeint_rtol # relative tolerance
        self.odeint_atol = odeint_atol # absolute tolerance
        self.method = method # integrator method 


    def forward(self, x, timeslot):
        """
            x: hidden state vector for q or k with shape [B, L, H, E]
            timeslot: time slot with shape [L]

            Return: q(t) or k(t) or v(t), [B, L, L, H, E, 2]
        """
        
        # for example x = Q = [q1, q2, ..., qL]       
        B, L, H, E = x.shape
               
        # x0 form at dims [L, L]
        # [q1, q1, q1, ...]
        # [q2, q2, q2, ...]
        # ...
        # [qL, qL, qL, ...]
        x0 = x.unsqueeze(dim = 2).repeat(1, 1, L, 1, 1).unsqueeze(dim = -1) # (B, L, L, H, E, 1)
        
        # form for goal of x1 at dims [L, L]
        # [q1(t1), q1(t2), ..., q1(tL)]
        # [q2(t1), q2(t2), ..., q2(tL)]
        # ...
        # [qL(t1), qL(t2), ..., qL(tL)]
        x1 = torch.zeros_like(x0) # (B, L, L, H, E, 1)
        
        for idx in range(L):
            current_t = timeslot[idx]
            
            smaller_idx = (timeslot < current_t).nonzero(as_tuple=True)[0]
            past_timeslot = torch.cat([timeslot[smaller_idx], current_t.unsqueeze(dim = 0)])
            past_timeslot = reversed(past_timeslot)

            larger_idx = (timeslot > current_t).nonzero(as_tuple=True)[0]
            future_timeslot = torch.cat([current_t.unsqueeze(dim = 0), timeslot[larger_idx]])
                       
            current_state = x[:, idx, :, :] # (B, H, E)
            current_state = current_state.reshape(-1, E) # (B * H, E)
                        
            if len(past_timeslot) == 1:
                # only predict future           
                z0 = odeint(self.odefunc, current_state, future_timeslot, rtol = self.odeint_rtol, atol = self.odeint_atol, 
                                      method = self.method)[:] # (L, B * H, E)                  
            elif len(future_timeslot) == 1:
                # only infer past
                z0 = torch.flip(odeint(self.odefunc, current_state, past_timeslot, rtol = self.odeint_rtol, atol = self.odeint_atol, 
                                      method = self.method)[:], dims = [0]) # (L, B * H, E)
            else:
                # require to infer past and predict future
                z0_past = torch.flip(odeint(self.odefunc, current_state, past_timeslot, rtol = self.odeint_rtol, atol = self.odeint_atol, 
                                      method = self.method)[:], dims = [0]) # (L1, B * H, E)
                z0_future = odeint(self.odefunc, current_state, future_timeslot, rtol = self.odeint_rtol, atol = self.odeint_atol, 
                                      method = self.method)[:] # (L2, B * H, E)
                z0 = torch.cat([z0_past, z0_future[1:, :, :]], dim = 0) # (L, B * H, E)
            
            z0 = z0.reshape(L, B, H, E) # (L, B, H, E)
            z0 = z0.permute(1, 0, 2, 3) # (B, L, H, E)
            x1[:, idx, :, :, :, 0] = z0
        
        # x form at dims [L, L, 2]
        # [(q1, q1(t1)), (q1, q1(t2)), ..., (q1, q1(tL))]
        # [(q2, q2(t1)), (q2, q2(t2)), ..., (q2, q2(tL))]
        # ...
        # [(qL, qL(t1)), (qL, qL(t2)), ..., (qL, qL(tL))]
        x = torch.cat([x0, x1], dim = -1) # (B, L, L, H, E, 2)
        
        return x

