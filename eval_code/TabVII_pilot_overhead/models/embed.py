import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import math

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, 
                                    kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2)
        return x

class DataEmbedding_withoutPE(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', dropout=0.1):
        super(DataEmbedding_withoutPE, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        
        '''
        # channel visualization
        
        pos_enc = self.position_embedding(x) 
        
        plt.figure(figsize = (10, 6))
        plt.plot(np.arange(0, pos_enc.shape[1]), pos_enc[0, :, 0].cpu().detach().numpy(), 'o-', label = 'sin')
        plt.plot(np.arange(0, pos_enc.shape[1]), pos_enc[0, :, 1].cpu().detach().numpy(), 'x-', label = 'cos')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        
        plt.savefig('DTPE Visualization.png')
        
        print("Have saved DTPE Visualization.png!")
        '''
        
        x = self.value_embedding(x)
        # x = self.value_embedding(x)
        
        return self.dropout(x)

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        
        '''
        # channel visualization
        
        pos_enc = self.position_embedding(x) 
        
        plt.figure(figsize = (10, 6))
        plt.plot(np.arange(0, pos_enc.shape[1]), pos_enc[0, :, 0].cpu().detach().numpy(), 'o-', label = 'sin')
        plt.plot(np.arange(0, pos_enc.shape[1]), pos_enc[0, :, 1].cpu().detach().numpy(), 'x-', label = 'cos')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        
        plt.savefig('DTPE Visualization.png')
        
        print("Have saved DTPE Visualization.png!")
        '''
        
        x = self.value_embedding(x) + self.position_embedding(x) 
        # x = self.value_embedding(x)
        
        return self.dropout(x)

class Continuous_time_DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type = 'fixed', dropout = 0.1, omega_0 = 30.):
        super(Continuous_time_DataEmbedding, self).__init__()
        
        self.value_embedding = TokenEmbedding(c_in = c_in, d_model = d_model)
        
        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)])
        
        self.position_embedding = PositionalEmbedding(d_model = d_model)
        
        self.dropout = nn.Dropout(p = dropout)
        self.omega_0 = omega_0

    def forward(self, x, t):
        
        '''
        Args:
            x : input feature vector, (b * M, his_len or label_len + pre_len, F)
            t : time, (his_len or label_len + pre_len,)
        
        Returns:
            continuous-time encoder ouput: (b * M, his_len or label_len + pre_len, F)
        '''
        
        t = t.unsqueeze(dim = 0).repeat(x.shape[0], 1) # (b * M, his_len or label_len + pre_len)
        pos_enc = t.unsqueeze(-1) / self.position_vec.to(t.device) # (b * M, his_len or label_len + pre_len, F)
        pos_enc[:, :, 0::2] = torch.sin(self.omega_0 * pos_enc[:, :, 0::2])
        pos_enc[:, :, 1::2] = torch.cos(self.omega_0 * pos_enc[:, :, 1::2])
        
        '''
        # channel visualization
        
        plt.figure(figsize = (10, 6))
        plt.plot(np.arange(0, pos_enc.shape[1]), pos_enc[0, :, 0].cpu().detach().numpy(), 'o-', label = 'sin')
        plt.plot(np.arange(0, pos_enc.shape[1]), pos_enc[0, :, 1].cpu().detach().numpy(), 'x-', label = 'cos')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        
        plt.savefig('CTPE Visualization.png')
        
        print("Have saved CTPE Visualization.png!")
        '''
        
        x = self.value_embedding(x) + pos_enc
        
        return self.dropout(x.float())