import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import collections
from torchdiffeq import odeint
from utils import TriangularCausalMask, ProbMask
from models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from models.decoder import Decoder, DecoderLayer
from models.attn import FullAttention, ProbAttention, AttentionLayer, CTAttention, Prob_CTAttention
from models.embed import DataEmbedding_withoutPE, DataEmbedding, Continuous_time_DataEmbedding
from models.weighted_ode_output import WODE
import matplotlib
import matplotlib.pyplot as plt

class RNNUnit(nn.Module):
    def __init__(self, features, input_size, hidden_size, num_layers = 2):
        
        super(RNNUnit, self).__init__()
        
        self.encoder = nn.Sequential(nn.Linear(features, input_size))
        self.rnn = nn.RNN(input_size, hidden_size, num_layers)
        self.decoder = nn.Sequential(nn.Linear(hidden_size, features))
        # self.out = nn.Linear(hidden_size, features)
        
    def forward(self, x, prev_hidden):
        # if len(x.shape) > 3:
            # print('x shape must be 3')
        
        L, B, F = x.shape
        output = x.reshape(L * B, -1) 
        output = self.encoder(output)
        # print(1)
        output = output.reshape(L, B, -1)       
        output, cur_hidden = self.rnn(output, prev_hidden)
        # print(2)
        output = output.reshape(L * B, -1)
        # print(3)
        output = self.decoder(output)
        # output = self.out(torch.cos(output))
        output = output.reshape(L, B, -1) 
        
        return output, cur_hidden

class Vanilla_RNN(nn.Module):
    def __init__(self, features, input_size, hidden_size, num_layers = 2):
        
        super(Vanilla_RNN, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_size 
        self.hidden_size = hidden_size
        self.features = features
        self.model = RNNUnit(features, input_size, hidden_size, num_layers = self.num_layers)
        
    def train_data(self, x,device):
        
        BATCH_SIZE, seq_len, _ = x.shape
        prev_hidden = torch.zeros(self.num_layers, BATCH_SIZE,  self.hidden_size).to(device) 
        outputs = []
        for idx in range(seq_len):
            output, prev_hidden= self.model(x[:,idx:idx+1,...].permute(1,0,2).contiguous(), prev_hidden)
            outputs.append(output)
        outputs = torch.cat(outputs, dim = 0).permute(1,0,2).contiguous()


        return outputs
    
    def test_data(self, x, pred_len,device):
        
        BATCH_SIZE, seq_len, _ = x.shape
        prev_hidden = torch.zeros(self.num_layers, BATCH_SIZE,  self.hidden_size).to(device) 
        outputs = []
        for idx in range(seq_len + pred_len - 1):
            if idx < seq_len:
                output, prev_hidden= self.model(x[:,idx:idx+1,...].permute(1,0,2).contiguous(), prev_hidden)
            else:
                output, prev_hidden= self.model(output,  prev_hidden) 
            if idx >= seq_len - 1:
                outputs.append(output)
        outputs = torch.cat(outputs, dim = 0).permute(1,0,2).contiguous()

        return outputs
    
class LSTMUnit(nn.Module):
    def __init__(self, features, input_size, hidden_size, num_layers = 2):
        
        super(LSTMUnit, self).__init__()
        
        self.encoder = nn.Sequential(nn.Linear(features, input_size))
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.decoder = nn.Sequential(nn.Linear(hidden_size, features))
        # self.out = nn.Linear(hidden_size, features)
        
    def forward(self, x, prev_hidden, prev_cell):
        # if len(x.shape) > 3:
            # print('x shape must be 3')
        
        L, B, F = x.shape
        output = x.reshape(L * B, -1) 
        output = self.encoder(output)
        # print(1)
        output = output.reshape(L, B, -1)       
        output, (cur_hidden, cur_cell) = self.lstm(output, (prev_hidden, prev_cell))
        # print(2)
        output = output.reshape(L * B, -1)
        # print(3)
        output = self.decoder(output)
        # output = self.out(torch.cos(output))
        output = output.reshape(L, B, -1) 
        
        return output, cur_hidden, cur_cell

class Vanilla_LSTM(nn.Module):
    def __init__(self, features, input_size, hidden_size, num_layers = 2):
        
        super(Vanilla_LSTM, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_size 
        self.hidden_size = hidden_size
        self.features = features
        self.model = LSTMUnit(features, input_size, hidden_size, num_layers = self.num_layers)
    
    def train_data(self, x, device):
        
        BATCH_SIZE, seq_len, _ = x.shape
        prev_hidden = torch.zeros(self.num_layers, BATCH_SIZE,  self.hidden_size).to(device) 
        prev_cell = torch.zeros(self.num_layers, BATCH_SIZE,  self.hidden_size).to(device) 
        outputs = []
        for idx in range(seq_len):
            output, prev_hidden, prev_cell = self.model(x[:,idx:idx+1,...].permute(1,0,2).contiguous(), prev_hidden, prev_cell)
            outputs.append(output)
        outputs = torch.cat(outputs, dim = 0).permute(1,0,2).contiguous()


        return outputs

    def test_data(self, x, pred_len, device):
        
        BATCH_SIZE, seq_len, _ = x.shape
        prev_hidden = torch.zeros(self.num_layers, BATCH_SIZE,  self.hidden_size).to(device) 
        prev_cell = torch.zeros(self.num_layers, BATCH_SIZE,  self.hidden_size).to(device) 
        outputs = []
        for idx in range(seq_len + pred_len - 1):
            if idx < seq_len:
                output, prev_hidden, prev_cell = self.model(x[:,idx:idx+1,...].permute(1,0,2).contiguous(), prev_hidden, prev_cell)
            else:
                output, prev_hidden, prev_cell = self.model(output,  prev_hidden, prev_cell) 
            if idx >= seq_len - 1:
                outputs.append(output)
        outputs = torch.cat(outputs, dim = 0).permute(1,0,2).contiguous()


        return outputs

class GRUUnit(nn.Module):
    def __init__(self, features, input_size, hidden_size, num_layers = 2):
        
        super(GRUUnit, self).__init__()
        
        self.encoder = nn.Sequential(nn.Linear(features, input_size))
        self.gru = nn.GRU(input_size, hidden_size, num_layers)
        self.decoder = nn.Sequential(nn.Linear(hidden_size, features))
        # self.out = nn.Linear(hidden_size, features)
        
    def forward(self, x, prev_hidden):
        # if len(x.shape) > 3:
            # print('x shape must be 3')
        
        L, B, F = x.shape
        output = x.reshape(L * B, -1) 
        output = self.encoder(output)
        # print(1)
        output = output.reshape(L, B, -1)       
        output, cur_hidden = self.gru(output, prev_hidden)
        # print(2)
        output = output.reshape(L * B, -1)
        # print(3)
        output = self.decoder(output)
        # output = self.out(torch.cos(output))
        output = output.reshape(L, B, -1) 
        
        return output, cur_hidden

class Vanilla_GRU(nn.Module):
    def __init__(self, features, input_size, hidden_size, num_layers = 2):
        
        super(Vanilla_GRU, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_size 
        self.hidden_size = hidden_size
        self.features = features
        self.model = GRUUnit(features, input_size, hidden_size, num_layers = self.num_layers)
    

    def train_data(self, x, device):
        
        BATCH_SIZE, seq_len, _ = x.shape
        prev_hidden = torch.zeros(self.num_layers, BATCH_SIZE,  self.hidden_size).to(device) 
        outputs = []
        for idx in range(seq_len):
            output, prev_hidden= self.model(x[:,idx:idx+1,...].permute(1,0,2).contiguous(), prev_hidden)
            outputs.append(output)
        outputs = torch.cat(outputs, dim = 0).permute(1,0,2).contiguous()


        return outputs

    def test_data(self, x, pred_len,device):
        
        BATCH_SIZE, seq_len, _ = x.shape
        prev_hidden = torch.zeros(self.num_layers, BATCH_SIZE,  self.hidden_size).to(device) 
        outputs = []
        for idx in range(seq_len + pred_len - 1):
            if idx < seq_len:
                output, prev_hidden= self.model(x[:,idx:idx+1,...].permute(1,0,2).contiguous(), prev_hidden)
            else:
                output, prev_hidden= self.model(output,  prev_hidden) 
            if idx >= seq_len - 1:
                outputs.append(output)
        outputs = torch.cat(outputs, dim = 0).permute(1,0,2).contiguous()


        return outputs
    
class Vanilla_Transformer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor = 5, d_model = 512, n_heads = 8, e_layers = 3, d_layers = 2, d_ff = 512, 
                dropout = 0.0, attn = 'prob', embed = 'fixed', activation = 'gelu',
                output_attention = False, distil = True,
                device = torch.device('cuda:0')):
        super(Vanilla_Transformer, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention
        
        # PE
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, dropout)
                       
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        
        # Encoder
        stacks = list(range(e_layers, 0, -1)) # you can customize here
        encoders = [
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                    d_model, n_heads),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ) for l in range(el)
                ],
                [
                    ConvLayer(
                        d_model
                    ) for l in range(el-1)
                ] if distil else None,
                norm_layer = torch.nn.LayerNorm(d_model)
            ) for el in stacks]
        self.encoder = EncoderStack(encoders)
        
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(                    
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                    time_emb_kind = 'PE'
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)
        
    def forward(self, x_enc, x_dec, his_timeslot = None, pre_timeslot = None,
                enc_self_mask = None, dec_self_mask = None, dec_enc_mask = None):
        
        # x_enc (b * M, his_len, F)
        # x_dec (b * M, pre_len + label_len, F)
        # his_timeslot: time slot of historical channel sequence, size of (his_len,)
        # pre_timeslot: time slot of predicted channel sequence, size of (pre_len + label_len,)
               
        enc_out = self.enc_embedding(x_enc)
            
        enc_out, attns = self.encoder(enc_out, his_timeslot = his_timeslot, attn_mask = enc_self_mask)
                
        dec_out = self.dec_embedding(x_dec)
            
        dec_out = self.decoder(dec_out, enc_out, his_timeslot = his_timeslot, pre_timeslot = pre_timeslot, x_mask = dec_self_mask, cross_mask = dec_enc_mask)
        dec_out = self.projection(dec_out)
        
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:,-self.pred_len:,:], attns
        else:
            return dec_out[:,-self.pred_len:,:] # (b * M, pre_len, F)
    
# define derivative function
class ODEfunc(nn.Module):
    def __init__(self, ode_i = 256, ode_h = 256, ode_o = 256, hidden_layers = 2):
        super(ODEfunc, self).__init__()
        
        self.net = []
        
        self.net.append(nn.Linear(ode_i, ode_h))
        self.net.append(nn.Tanh())
        
        for i in range(hidden_layers):
            self.net.append(nn.Linear(ode_h, ode_h))
            self.net.append(nn.Tanh())
        
        self.net.append(nn.Linear(ode_h, ode_o))
        
        self.net = nn.Sequential(*self.net)

    def forward(self, t, h):        
        '''
        Args:
            t : time, (1)
            h : rnn hidden state, (b, ode_i)
        
        Returns:
            dhdt: dh/dt, (b, ode_o)
        '''
        
        dhdt = self.net(h)      
        return dhdt

class ConcatLinear_v2(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(1, dim_out, bias = False)
        self._hyper_bias.weight.data.fill_(0.0)

    def forward(self, t, x):
        return self._layer(x) + self._hyper_bias(t.view(-1, 1))


class ConcatLinearNorm(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(1, dim_out, bias = False)
        self._hyper_bias.weight.data.fill_(0.0)
        self.norm = nn.LayerNorm(dim_out, eps = 1e-6)

    def forward(self, t, x):
        return self.norm(self._layer(x) + self._hyper_bias(t.view(-1, 1)))


class ConcatSquashLinear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(1, dim_out, bias = False)
        self._hyper_gate = nn.Linear(1, dim_out)

    def forward(self, t, x):
        return self._layer(x) * torch.sigmoid(self._hyper_gate(t.view(-1, 1))) \
               + self._hyper_bias(t.view(-1, 1))

LAYERTYPES = {
    'ConcatLinear_v2': ConcatLinear_v2,
    'ConcatLinearNorm': ConcatLinearNorm,
    'ConcatSquashLinear': ConcatSquashLinear
}
               
# define derivative function
class ODEfunc_witht(nn.Module):
    def __init__(self, ode_i = 256, ode_h = 256, ode_o = 256, hidden_layers = 2, derivative_function_type = 'ConcatLinear_v2'):
        super(ODEfunc_witht, self).__init__()
        
        self.net = []
        
        self.net.append(LAYERTYPES[derivative_function_type](dim_in = ode_i, dim_out = ode_h))
        
        for i in range(hidden_layers):
            self.net.append(LAYERTYPES[derivative_function_type](dim_in = ode_h, dim_out = ode_h))
        
        self.net.append(LAYERTYPES[derivative_function_type](dim_in = ode_h, dim_out = ode_o))
        
        self.net = nn.Sequential(*self.net)
        
        self.actfn = nn.Tanh()

    def forward(self, t, h):        
        '''
        Args:
            t : time, (1)
            h : rnn hidden state, (b, ode_i)
        
        Returns:
            dhdt: dh/dt, (b, ode_o)
        '''
        
        for layer_idx in range(len(self.net) - 1):        
            h = self.net[layer_idx](t, h)  
            h = self.actfn(h)
        
        dhdt = self.net[-1](t, h)
            
        return dhdt

class Neural_ODE(nn.Module):
    def __init__(self,
                 features = 2,
                 rnn_i = 128,
                 rnn_h = 128,
                 rnn_layers = 2,
                 rnn_type = 'LSTM',
                 ode_h = 128,
                 ode_hidden_layers = 2,
                 odeint_rtol = 1e-3, 
                 odeint_atol = 1e-4, 
                 method = 'dopri5',
                 device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super(Neural_ODE, self).__init__()

        # initialize parameters
        self.device = device
        self.rnn_h = rnn_h
        # ode parameters
        self.odeint_rtol = odeint_rtol # relative tolerance
        self.odeint_atol = odeint_atol # absolute tolerance
        self.method = method # integrator method   
        self.rnn_layers = rnn_layers
        self.rnn_h = rnn_h
        self.rnn_type = rnn_type
        
        self.embedding_fc = nn.Linear(features, rnn_i)
        # self.ln = nn.LayerNorm(rnn_i)
        if rnn_type == 'RNN':
            self.rnn = nn.RNN(input_size = rnn_i, hidden_size = rnn_h, num_layers = rnn_layers)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size = rnn_i, hidden_size = rnn_h, num_layers = rnn_layers)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size = rnn_i, hidden_size = rnn_h, num_layers = rnn_layers)
        else:  
            raise NotImplementedError
        
        self.odefunc = ODEfunc(ode_i = rnn_h, ode_h = ode_h, ode_o = rnn_h, hidden_layers = ode_hidden_layers)
           
        # self.drop_predict = nn.Dropout(0.02)
        self.fc_predict = nn.Linear(rnn_h, features)
               
    def forward(self, x, pre_len, pre_timeslot, device):        
        '''
        Args:
            x: historical c-band channel sequence, size of (b * M, F, his_len)
            pre_len: length of predicted channel sequence 
            pre_timeslot: time slot of predicted channel sequence, size of (pre_len + 1)
        
        Returns:
            y: predicted channel sequence, size of (b * M, F, pre_len)    
        '''
                
        b, _, his_len = x.shape
                
        x = x.permute(2, 0, 1) # (his_len, b * M, F)
        
        x = self.embedding_fc(x) # (his_len, b * M, F)
        
        # define lstm hidden state/cell state
        if self.rnn_type == 'LSTM':
            rnn_hidden = torch.zeros(self.rnn_layers, b, self.rnn_h).to(device) 
            rnn_cell = torch.zeros(self.rnn_layers, b, self.rnn_h).to(device) 
        else:
            rnn_hidden = torch.zeros(self.rnn_layers, b, self.rnn_h).to(device) 
                
        # LN
        # x = self.ln(x)
                          
        # encoder rnn learn historical sequence        
        for idx in range(his_len):
            if self.rnn_type == 'LSTM':
                output, (rnn_hidden, rnn_cell) = self.rnn(x[idx : idx + 1, :, :].contiguous(), (rnn_hidden, rnn_cell))
            else:
                output, rnn_hidden = self.rnn(x[idx : idx + 1, :, :].contiguous(), rnn_hidden)
        
        z0 = output.permute(1, 0, 2) # (b * M, 1, hidden_size)
        z0 = z0.reshape(b, -1) # (b * M, 1 * hidden_size)
        
        # ode decoding
        y = odeint(self.odefunc, z0, pre_timeslot, rtol = self.odeint_rtol, atol = self.odeint_atol, method = self.method)[1 : pre_len + 1] # (pre_len, b * M, hidden_size)   
        y = y.permute(1, 0, 2) # (b * M, pre_len, hidden_size)  
                             
        # output FC
        # y = self.drop_predict(y)
        y = self.fc_predict(y) # (b * M, pre_len, F)
        
        y = y.permute(0, 2, 1) # (b * M, F, pre_len)  
        
        return y.squeeze()
    
class Latent_ODE(nn.Module):
    def __init__(self,
                 features = 2,
                 rnn_i = 128,
                 rnn_h = 128,
                 rnn_layers = 2,
                 rnn_type = 'LSTM',
                 enc_dec_odefunc_ifshared = True,
                 enc_ode_h = 128,
                 enc_ode_hidden_layers = 2,
                 enc_odeint_rtol = 1e-3, 
                 enc_odeint_atol = 1e-4, 
                 enc_method = 'rk4',
                 dec_ode_h = 128,
                 dec_ode_hidden_layers = 2,
                 dec_odeint_rtol = 1e-3, 
                 dec_odeint_atol = 1e-4, 
                 dec_method = 'rk4',
                 device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super(Latent_ODE, self).__init__()

        # initialize parameters
        self.device = device
        self.rnn_h = rnn_h
        # ode parameters
        self.enc_dec_odefunc_ifshared = enc_dec_odefunc_ifshared
        self.enc_odeint_rtol = enc_odeint_rtol # relative tolerance
        self.enc_odeint_atol = enc_odeint_atol # absolute tolerance
        self.enc_method = enc_method # integrator method           
        self.dec_odeint_rtol = dec_odeint_rtol # relative tolerance
        self.dec_odeint_atol = dec_odeint_atol # absolute tolerance
        self.dec_method = dec_method # integrator method 
        
        self.rnn_layers = rnn_layers
        self.rnn_h = rnn_h
        self.rnn_type = rnn_type
        
        self.embedding_fc = nn.Linear(features, rnn_i)
        if rnn_type == 'RNN':
            self.rnn = nn.RNN(input_size = rnn_i, hidden_size = rnn_h, num_layers = rnn_layers)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size = rnn_i, hidden_size = rnn_h, num_layers = rnn_layers)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size = rnn_i, hidden_size = rnn_h, num_layers = rnn_layers)
        else:  
            raise NotImplementedError
        
        self.dec_odefunc = ODEfunc(ode_i = rnn_h, ode_h = enc_ode_h, ode_o = rnn_h, hidden_layers = enc_ode_hidden_layers)
        if enc_dec_odefunc_ifshared == False:
            self.enc_odefunc = ODEfunc(ode_i = rnn_h, ode_h = dec_ode_h, ode_o = rnn_h, hidden_layers = dec_ode_hidden_layers)
                       
        # self.drop_predict = nn.Dropout(0.02)
        self.fc_predict = nn.Linear(rnn_h, features)
               
    def forward(self, x, pre_len, his_timeslot, pre_timeslot, device):        
        '''
        Args:
            x: historical c-band channel sequence, size of (b * M, F, his_len)
            pre_len: length of predicted channel sequence 
            his_timeslot: time slot of historical channel sequence, size of (his_len,)
            pre_timeslot: time slot of predicted channel sequence, size of (pre_len + 1,)
        
        Returns:
            y: predicted channel sequence, size of (b * M, F, pre_len)    
        '''
                
        b, _, his_len = x.shape
                
        x = x.permute(2, 0, 1) # (his_len, b * M, F)
        
        x = self.embedding_fc(x) # (his_len, b * M, F)
        
        # define historical sampling time                                 
        his_ts = torch.tensor([0., 0.], device = device) 
        his_ts[1] = his_timeslot[0]
        his_ts_idx = 1
        
        # define lstm hidden state/cell state
        if self.rnn_type == 'LSTM':
            rnn_hidden = torch.zeros(self.rnn_layers, b, self.rnn_h).to(device) 
            rnn_cell = torch.zeros(self.rnn_layers, b, self.rnn_h).to(device) 
        else:
            rnn_hidden = torch.zeros(self.rnn_layers, b, self.rnn_h).to(device) 
                
        for idx in range(his_len):
            # update algorithm
            # 1. \bar{hi-1} = odesolver(hi-1, (ti-1, ti))
            # 2. (hi, ci) = lstm(xi, (\bar{hi-1}, ci-1)) 
            if idx == 0:
                # print('state0')
                if self.rnn_type == 'LSTM':
                    output, (rnn_hidden, rnn_cell) = self.rnn(x[idx : idx + 1, :, :].contiguous(), (rnn_hidden, rnn_cell))
                else:
                    output, rnn_hidden = self.rnn(x[idx : idx + 1, :, :].contiguous(), rnn_hidden)
            else:      
                # print('state1')
                # update sampling time
                his_ts[0] = his_ts[1]
                his_ts[1] = his_timeslot[his_ts_idx]
                his_ts_idx += 1
                # print(his_ts)
                # ode update in [his_ts[0], his_ts[1]]
                output = output.permute(1, 0, 2) # (b * M, 1, hidden_size)
                output = output.reshape(b, -1) # (b * M, 1 * hidden_size)
                
                if self.enc_dec_odefunc_ifshared:
                    output = odeint(self.dec_odefunc, output, his_ts, rtol = self.dec_odeint_rtol, atol = self.dec_odeint_atol, method = self.dec_method)[1]  
                else:
                    output = odeint(self.enc_odefunc, output, his_ts, rtol = self.enc_odeint_rtol, atol = self.enc_odeint_atol, method = self.enc_method)[1]  
                    
                output = output.reshape(b, 1, -1) # (b * M, 1, hidden_size)
                output = output.permute(1, 0, 2) # (1, b * M, hidden_size)  
                # basic rnn update
                if self.rnn_type == 'LSTM':
                    output, (rnn_hidden, rnn_cell) = self.rnn(x[idx : idx + 1, :, :].contiguous(), (torch.cat([rnn_hidden[0 : 1, :, :], output], dim = 0).contiguous(), rnn_cell))
                else:
                    output, rnn_hidden = self.rnn(x[idx : idx + 1, :, :].contiguous(), torch.cat([rnn_hidden[0 : 1, :, :], output], dim = 0).contiguous())
        
        z0 = output.permute(1, 0, 2) # (b * M, 1, hidden_size)
        z0 = z0.reshape(b, -1) # (b * M, 1 * hidden_size)
                 
        # ode decoding
        y = odeint(self.dec_odefunc, z0, pre_timeslot, rtol = self.dec_odeint_rtol, atol = self.dec_odeint_atol, method = self.dec_method)[1 : pre_len + 1] # (pre_len, b * M, rnn_h)   
        y = y.permute(1, 0, 2) # (b * M, pre_len, rnn_h)           
            
        # output FC
        # y = self.drop_predict(y)
        y = self.fc_predict(y) # (b * M, pre_len, F)
        
        y = y.permute(0, 2, 1) # (b * M, F, pre_len)  
        
        return y.squeeze()
    
class CT_Transformer(nn.Module):
    def __init__(self, time_emb_kind, omega_0, enc_ctsa, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor = 5, d_model = 512, n_heads = 8, e_layers = 3, d_layers = 2, d_ff = 512, 
                dropout = 0.0, attn = 'prob', ct_attn = 'ct_full', embed = 'fixed', activation = 'gelu',
                output_attention = False, distil = True,
                continuous_qkv_method = ['interp', 'interp', 'ode'],
                derivative_function_type = 'ConcatLinear_v2',
                ode_h = 128,
                ode_hidden_layers = 2,
                odeint_rtol = 1e-3, 
                odeint_atol = 1e-4, 
                method = 'dopri5',
                device = torch.device('cuda:0')):
        super(CT_Transformer, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention
        self.time_emb_kind = time_emb_kind
        
        # Encoding
        if time_emb_kind == 'HFTE':
            self.enc_embedding = Continuous_time_DataEmbedding(enc_in, d_model, embed, dropout, omega_0)
            self.dec_embedding = Continuous_time_DataEmbedding(dec_in, d_model, embed, dropout, omega_0)
        elif time_emb_kind == 'PE':
            self.enc_embedding = DataEmbedding(enc_in, d_model, embed, dropout)
            self.dec_embedding = DataEmbedding(dec_in, d_model, embed, dropout)
        elif time_emb_kind == 'without':
            self.enc_embedding = DataEmbedding_withoutPE(enc_in, d_model, embed, dropout)
            self.dec_embedding = DataEmbedding_withoutPE(dec_in, d_model, embed, dropout)
        else:
            raise ValueError(f"Unknown time_emb_kind: {time_emb_kind}")
                       
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        CTAttn = Prob_CTAttention if ct_attn == 'ct_prob' else CTAttention
        
        # print(CTAttn)
        
        # Encoder
        stacks = list(range(e_layers, 0, -1)) # you can customize here
        encoders = [
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(CTAttention(False, factor, attention_dropout=dropout, output_attention=output_attention,
                                    odefunc_q = None if continuous_qkv_method[0] == 'interp' else ODEfunc_witht(ode_i = d_model // n_heads, ode_h = ode_h, ode_o = d_model // n_heads, hidden_layers = ode_hidden_layers, derivative_function_type = derivative_function_type),
                                    odefunc_k = None if continuous_qkv_method[1] == 'interp' else ODEfunc_witht(ode_i = d_model // n_heads, ode_h = ode_h, ode_o = d_model // n_heads, hidden_layers = ode_hidden_layers, derivative_function_type = derivative_function_type),
                                    odefunc_v = None if continuous_qkv_method[2] == 'interp' else ODEfunc_witht(ode_i = d_model // n_heads, ode_h = ode_h, ode_o = d_model // n_heads, hidden_layers = ode_hidden_layers, derivative_function_type = derivative_function_type),
                                    odeint_rtol = odeint_rtol, 
                                    odeint_atol = odeint_atol, 
                                    method = method), d_model, n_heads) if (el == stacks[0] and l == 0 and enc_ctsa) else AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                    d_model, n_heads),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ) for l in range(el)
                ],
                [
                    ConvLayer(
                        d_model
                    ) for l in range(el-1)
                ] if distil else None,
                norm_layer=torch.nn.LayerNorm(d_model)
            ) for el in stacks]
        self.encoder = EncoderStack(encoders)
        
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(CTAttn(True, factor, attention_dropout=dropout, output_attention=False,
                                               odefunc_q = None if continuous_qkv_method[0] == 'interp' else ODEfunc_witht(ode_i = d_model // n_heads, ode_h = ode_h, ode_o = d_model // n_heads, hidden_layers = ode_hidden_layers, derivative_function_type = derivative_function_type),
                                               odefunc_k = None if continuous_qkv_method[1] == 'interp' else ODEfunc_witht(ode_i = d_model // n_heads, ode_h = ode_h, ode_o = d_model // n_heads, hidden_layers = ode_hidden_layers, derivative_function_type = derivative_function_type),
                                               odefunc_v = None if continuous_qkv_method[2] == 'interp' else ODEfunc_witht(ode_i = d_model // n_heads, ode_h = ode_h, ode_o = d_model // n_heads, hidden_layers = ode_hidden_layers, derivative_function_type = derivative_function_type),
                                               odeint_rtol = odeint_rtol, 
                                               odeint_atol = odeint_atol, 
                                               method = method), 
                                d_model, n_heads),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                    time_emb_kind = time_emb_kind,
                    omega_0 = omega_0
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)
        
    def forward(self, x_enc, x_dec, his_timeslot, pre_timeslot,
                enc_self_mask = None, dec_self_mask = None, dec_enc_mask = None):
        
        # x_enc (b * M, his_len, F)
        # x_dec (b * M, pre_len + label_len, F)
        # his_timeslot: time slot of historical channel sequence, size of (his_len,)
        # pre_timeslot: time slot of predicted channel sequence, size of (pre_len + label_len,)
               
        if self.time_emb_kind == 'HFTE':
            enc_out = self.enc_embedding(x_enc, his_timeslot)
        elif self.time_emb_kind == 'PE' or self.time_emb_kind == 'without':
            enc_out = self.enc_embedding(x_enc)
            
        enc_out, attns = self.encoder(enc_out, his_timeslot = his_timeslot, attn_mask = enc_self_mask)
                
        if self.time_emb_kind == 'HFTE':
            dec_out = self.dec_embedding(x_dec, pre_timeslot)
        elif self.time_emb_kind == 'PE' or self.time_emb_kind == 'without':
            dec_out = self.dec_embedding(x_dec)
            
        dec_out = self.decoder(dec_out, enc_out, his_timeslot = his_timeslot, pre_timeslot = pre_timeslot, x_mask = dec_self_mask, cross_mask = dec_enc_mask)
        dec_out = self.projection(dec_out)
        
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:,-self.pred_len:,:], attns
        else:
            return dec_out[:,-self.pred_len:,:] # (b * M, pre_len, F)

class CT_Transformer_encoderonly_v0(nn.Module):
    def __init__(self, time_emb_kind, omega_0, enc_ctsa, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor = 5, d_model = 512, n_heads = 8, e_layers = 3, d_layers = 2, d_ff = 512, 
                dropout = 0.0, attn = 'prob', embed = 'fixed', activation = 'gelu',
                output_attention = False, distil = True,
                continuous_qkv_method = ['interp', 'interp', 'ode'],
                derivative_function_type = 'ConcatLinear_v2',
                ode_h = 128,
                ode_hidden_layers = 2,
                odeint_rtol = 1e-3, 
                odeint_atol = 1e-4, 
                method = 'dopri5',
                device = torch.device('cuda:0')):
        super(CT_Transformer_encoderonly_v0, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention
        self.time_emb_kind = time_emb_kind
        
        # Encoding
        if time_emb_kind == 'HFTE':
            self.enc_embedding = Continuous_time_DataEmbedding(enc_in, d_model, embed, dropout, omega_0)
        elif time_emb_kind == 'PE':
            self.enc_embedding = DataEmbedding(enc_in, d_model, embed, dropout)
        elif time_emb_kind == 'without':
            self.enc_embedding = DataEmbedding_withoutPE(enc_in, d_model, embed, dropout)
        else:
            raise ValueError(f"Unknown time_emb_kind: {time_emb_kind}")
                       
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        
        # Encoder
        stacks = list(range(e_layers, 0, -1)) # you can customize here
        encoders = [
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(CTAttention(False, factor, attention_dropout=dropout, output_attention=output_attention,
                                    odefunc_q = None if continuous_qkv_method[0] == 'interp' else ODEfunc_witht(ode_i = d_model // n_heads, ode_h = ode_h, ode_o = d_model // n_heads, hidden_layers = ode_hidden_layers, derivative_function_type = derivative_function_type),
                                    odefunc_k = None if continuous_qkv_method[1] == 'interp' else ODEfunc_witht(ode_i = d_model // n_heads, ode_h = ode_h, ode_o = d_model // n_heads, hidden_layers = ode_hidden_layers, derivative_function_type = derivative_function_type),
                                    odefunc_v = None if continuous_qkv_method[2] == 'interp' else ODEfunc_witht(ode_i = d_model // n_heads, ode_h = ode_h, ode_o = d_model // n_heads, hidden_layers = ode_hidden_layers, derivative_function_type = derivative_function_type),
                                    odeint_rtol = odeint_rtol, 
                                    odeint_atol = odeint_atol, 
                                    method = method), d_model, n_heads) if (el == stacks[0] and l == 0 and enc_ctsa) else AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                    d_model, n_heads),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ) for l in range(el)
                ],
                [
                    ConvLayer(
                        d_model
                    ) for l in range(el-1)
                ] if distil else None,
                norm_layer=torch.nn.LayerNorm(d_model)
            ) for el in stacks]
        self.encoder = EncoderStack(encoders)
        
        self.end_conv1 = nn.Conv1d(in_channels = 38, out_channels = out_len, kernel_size = 1, bias = True) # 'in_channels' is not <seq_len + out_len> because multi-scale encoder
        self.end_conv2 = nn.Conv1d(in_channels = d_model, out_channels = c_out, kernel_size = 1, bias = True)
        # self.projection = nn.Linear(d_model, c_out, bias=True)
        
    def forward(self, x_enc, timeslot, enc_self_mask = None):
        
        # x_enc (b * M, his_len + pre_len, F)
        # timeslot: time slot of historical and predicted channel sequence, size of (his_len + pre_len,)
               
        if self.time_emb_kind == 'HFTE':
            enc_out = self.enc_embedding(x_enc, timeslot)
        elif self.time_emb_kind == 'PE' or self.time_emb_kind == 'without':
            enc_out = self.enc_embedding(x_enc)
            
        enc_out, attns = self.encoder(enc_out, his_timeslot = timeslot, attn_mask = enc_self_mask) # (b * M, 38, F)
                        
        enc_out = self.end_conv1(enc_out)
        enc_out = self.end_conv2(enc_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return enc_out[:, : , :], attns
        else:
            return enc_out[:, : , :] # (b * M, pre_len, F)

class CT_Transformer_encoderonly_v1(nn.Module):
    def __init__(self, time_emb_kind, omega_0, enc_ctsa, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor = 5, d_model = 512, n_heads = 8, e_layers = 3, d_layers = 2, d_ff = 512, 
                dropout = 0.0, attn = 'prob', embed = 'fixed', activation = 'gelu',
                output_attention = False, distil = True,
                continuous_qkv_method = ['interp', 'interp', 'ode'],
                derivative_function_type = 'ConcatLinear_v2',
                ode_h = 128,
                ode_hidden_layers = 2,
                odeint_rtol = 1e-3, 
                odeint_atol = 1e-4, 
                method = 'dopri5',
                weight_method = 'softmax',
                device = torch.device('cuda:0')):
        super(CT_Transformer_encoderonly_v1, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention
        self.time_emb_kind = time_emb_kind
        
        # Encoding
        if time_emb_kind == 'HFTE':
            self.enc_embedding = Continuous_time_DataEmbedding(enc_in, d_model, embed, dropout, omega_0)
        elif time_emb_kind == 'PE':
            self.enc_embedding = DataEmbedding(enc_in, d_model, embed, dropout)
        elif time_emb_kind == 'without':
            self.enc_embedding = DataEmbedding_withoutPE(enc_in, d_model, embed, dropout)
        else:
            raise ValueError(f"Unknown time_emb_kind: {time_emb_kind}")
                       
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        
        # Encoder
        stacks = list(range(e_layers, 1, -1)) # you can customize here
        encoders = [
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(CTAttention(False, factor, attention_dropout=dropout, output_attention=output_attention,
                                    odefunc_q = None if continuous_qkv_method[0] == 'interp' else ODEfunc_witht(ode_i = d_model // n_heads, ode_h = ode_h, ode_o = d_model // n_heads, hidden_layers = ode_hidden_layers, derivative_function_type = derivative_function_type),
                                    odefunc_k = None if continuous_qkv_method[1] == 'interp' else ODEfunc_witht(ode_i = d_model // n_heads, ode_h = ode_h, ode_o = d_model // n_heads, hidden_layers = ode_hidden_layers, derivative_function_type = derivative_function_type),
                                    odefunc_v = None if continuous_qkv_method[2] == 'interp' else ODEfunc_witht(ode_i = d_model // n_heads, ode_h = ode_h, ode_o = d_model // n_heads, hidden_layers = ode_hidden_layers, derivative_function_type = derivative_function_type),
                                    odeint_rtol = odeint_rtol, 
                                    odeint_atol = odeint_atol, 
                                    method = method), d_model, n_heads) if (el == stacks[0] and l == 0 and enc_ctsa) else AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                    d_model, n_heads),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ) for l in range(el)
                ],
                [
                    ConvLayer(
                        d_model
                    ) for l in range(el-1)
                ] if distil else None,
                norm_layer=torch.nn.LayerNorm(d_model)
            ) for el in stacks]
        self.encoder = EncoderStack(encoders)
        
        self.decoder = WODE(ODEfunc_witht(ode_i = d_model, ode_h = ode_h, ode_o = d_model, hidden_layers = ode_hidden_layers, derivative_function_type = derivative_function_type),
                            odeint_rtol = odeint_rtol, 
                            odeint_atol = odeint_atol, 
                            method = method,
                            weight_method = weight_method)
        
        #self.end_conv1 = nn.Conv1d(in_channels = 38, out_channels = out_len, kernel_size = 1, bias = True) # 'in_channels' is not <seq_len + out_len> because multi-scale encoder
        #self.end_conv2 = nn.Conv1d(in_channels = d_model, out_channels = c_out, kernel_size = 1, bias = True)
        self.projection = nn.Linear(d_model, c_out, bias=True)
        
    def forward(self, x_enc, his_timeslot, pre_timeslot, enc_self_mask = None):
        
        # x_enc (b * M, his_len, F)
        # his_timeslot: time slot of historical channel sequence, size of (his_len,)
        # pre_timeslot: time slot of predicted channel sequence, size of (pre_len,)
               
        if self.time_emb_kind == 'HFTE':
            enc_out = self.enc_embedding(x_enc, his_timeslot)
        elif self.time_emb_kind == 'PE' or self.time_emb_kind == 'without':
            enc_out = self.enc_embedding(x_enc)
            
        enc_out, attns = self.encoder(enc_out, his_timeslot = his_timeslot, attn_mask = enc_self_mask) # (b * M, his_len, F)
        
        enc_out = self.decoder(x = enc_out, his_timeslot = his_timeslot, pre_timeslot = pre_timeslot) # (b * M, pre_len, F)
        enc_out = self.projection(enc_out)
                        
        if self.output_attention:
            return enc_out[:, : , :], attns
        else:
            return enc_out[:, : , :] # (b * M, pre_len, F)