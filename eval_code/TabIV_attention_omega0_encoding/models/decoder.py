import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu", time_emb_kind = 'HFTE', omega_0 = 30.):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        
        self.time_emb_kind = time_emb_kind
        if time_emb_kind == 'HFTE':
            self.position_vec = torch.tensor(
                [math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)])
            self.omega_0 = omega_0
        
    def forward(self, x, cross, his_timeslot, pre_timeslot, x_mask=None, cross_mask=None):
        
        x = x + self.dropout(self.self_attention(
            x, x, x, his_timeslot = pre_timeslot, label_pre_timeslot = pre_timeslot,
            attn_mask=x_mask
        )[0])
        x = self.norm1(x)
        
        if self.time_emb_kind == 'HFTE':
            
            t = pre_timeslot.unsqueeze(dim = 0).repeat(x.shape[0], 1) # (b * M, his_len or label_len + pre_len)
            pos_dec = t.unsqueeze(-1) / self.position_vec.to(t.device) # (b * M, his_len or label_len + pre_len, F)
            pos_dec[:, :, 0::2] = torch.sin(self.omega_0 * pos_dec[:, :, 0::2])
            pos_dec[:, :, 1::2] = torch.cos(self.omega_0 * pos_dec[:, :, 1::2])
            
            x += pos_dec
        
        x = x + self.dropout(self.cross_attention(
            x, cross, cross, his_timeslot = his_timeslot, label_pre_timeslot = pre_timeslot,
            attn_mask=cross_mask
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))

        return self.norm3(x+y)

class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross, his_timeslot, pre_timeslot, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, his_timeslot, pre_timeslot, x_mask = x_mask, cross_mask = cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x