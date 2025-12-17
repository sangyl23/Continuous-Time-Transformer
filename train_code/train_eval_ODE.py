import torch
import numpy as np
import scipy.io as sio
from dataloader_ODE import Dataloader
from model_ODE import Vanilla_RNN, Vanilla_LSTM, Vanilla_GRU, Vanilla_Transformer, Neural_ODE, Latent_ODE, CT_Transformer, CT_Transformer_encoderonly_v0, CT_Transformer_encoderonly_v1
from utils import *
import time
import random
import math
import argparse
import logging

logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler("logfile.txt", mode = 'w'),
    logging.StreamHandler()
])
logger = logging.getLogger()

model_parser = argparse.ArgumentParser(add_help=False)
model_parser.add_argument('--model_name', type=str, default = 'CT_Transformer',
                          choices=['Vanilla_RNN', 'Vanilla_LSTM', 'Vanilla_GRU',
                                   'Vanilla_Transformer', 'Neural_ODE', 'Latent_ODE', 'CT_Transformer', 'CT_Transformer_encoderonly_v0', 'CT_Transformer_encoderonly_v1'])
known_args, remaining_argv = model_parser.parse_known_args()
model_name = known_args.model_name
print(f"Selected model: {model_name}")

parser = argparse.ArgumentParser(parents=[model_parser])

from configs.channel_system_parameters_setup import add_channel_system_args
parser = add_channel_system_args(parser)

if model_name == 'Vanilla_RNN':
    from configs.vanilla_rnn_parameters_setup import add_model_args
    parser = add_model_args(parser)
elif model_name == 'Vanilla_LSTM':
    from configs.vanilla_lstm_parameters_setup import add_model_args
    parser = add_model_args(parser)
elif model_name == 'Vanilla_GRU':
    from configs.vanilla_gru_parameters_setup import add_model_args
    parser = add_model_args(parser)
elif model_name == 'Vanilla_Transformer':
    from configs.vanilla_transformer_parameters_setup import add_model_args
    parser = add_model_args(parser)
elif model_name == 'Neural_ODE':
    from configs.neural_ode_parameters_setup import add_model_args
    parser = add_model_args(parser)
elif model_name == 'Latent_ODE':
    from configs.latent_ode_parameters_setup import add_model_args
    parser = add_model_args(parser)
elif model_name == 'CT_Transformer':
    from configs.ct_transformer_parameters_setup import add_model_args
    parser = add_model_args(parser)
elif model_name == 'CT_Transformer_encoderonly_v0':
    from configs.ct_transformer_encoderonly_v0_parameters_setup import add_model_args
    parser = add_model_args(parser)
elif model_name == 'CT_Transformer_encoderonly_v1':
    from configs.ct_transformer_encoderonly_v1_parameters_setup import add_model_args
    parser = add_model_args(parser)
else:
    raise ValueError(f"Unknown model name: {model_name}")    
    
from configs.training_parameters_setup import add_training_args
parser = add_training_args(parser)

args = parser.parse_args()

def channel_NMSE(h_gt, h_pre):
    # (b, 2, M, pre_len)
    h_gt = torch.complex(h_gt[:, 0, :, :], h_gt[:, 1, :, :]) # (b, M, pre_len)
    h_pre = torch.complex(h_pre[:, 0, :, :], h_pre[:, 1, :, :]) # (b, M, pre_len)    
    h_gt = h_gt.permute(0, 2, 1) # (b, pre_len, M)
    h_pre = h_pre.permute(0, 2, 1) # (b, pre_len, M)
        
    return torch.mean(torch.linalg.norm((h_gt - h_pre), dim = (2)) ** 2 / 
                      torch.linalg.norm(h_gt, dim = (2)) ** 2, dim = (0)), torch.mean(torch.linalg.norm(h_pre, dim = (2)), dim = (0)) # (pre_len,)

def chebyshev_points(a, b, n):
    # calculate n chebyshev interpolation point in (a, b) 
    chebyshev_points = np.cos((2 * np.arange(1, n + 1) - 1) / (2 * n) * np.pi)
    mapped_points = 0.5 * (a + b) + 0.5 * (b - a) * chebyshev_points 
    midpoint = a + (a + b) / 2
    rounded_points = np.where(mapped_points <= midpoint, np.floor(mapped_points), np.ceil(mapped_points))
    return rounded_points

# evaluation function
def eval(model, loader, M, his_len, pre_len, batch_size, device, criterion, his_timeslot, T_his):
    # reset dataloader
    loader.reset()
    # judge whether dataset is finished
    done = False
    # running loss
    running_NMSE = torch.zeros([pre_len, 1])
    running_pre_norm = torch.zeros([pre_len, 1])
    # count batch number
    b = batch_size
    batch_num = 0

    with torch.no_grad():
        # evaluate validation set
        while True:            
            # read files
            if args.intepolate_pilot_method == 'random':
                his_channel, pre_channel, pre_timeslot, his_timeslot, done = loader.next_batch()                    
            else:
                his_channel, pre_channel, pre_timeslot, _, done = loader.next_batch()
            # his_channel: historical channel sequence, size of (b, 2, M, his_len)                     
            # pre_channel: predicted channel sequence, size of (b, 2, M, pre_len)
            # pre_timeslot: predicted time slots, size of (b, pre_len)
            # his_timeslot: historical time slots, size of (b, his_len)
                           
            if done == True:
                break
            
            # normalized time instants
            pre_timeslot = pre_timeslot / T_his
            pre_timeslot.requires_grad = False
            
            if args.intepolate_pilot_method == 'random':
                his_timeslot = his_timeslot[0]
                his_timeslot = his_timeslot / T_his
                his_timeslot.requires_grad = False
            
            batch_num += 1
            
            if args.instance_norm == 'norm_2':
                his_channel_complex = torch.complex(his_channel[:, 0, :, :], his_channel[:, 1, :, :]).squeeze() # (b, M, his_len)
                norm_his_channel = torch.linalg.norm(his_channel_complex, ord = 2, dim = -1, keepdim = True) # (b, M, 1)
                his_channel_complex = his_channel_complex / norm_his_channel # (b, M, his_len)
                his_channel_real = his_channel_complex.real
                his_channel_imag = his_channel_complex.imag
                his_channel = torch.stack([his_channel_real, his_channel_imag], dim = 1) # (b, 2, M, his_len)
            elif args.instance_norm == 'norm_max':
                his_channel_complex = torch.complex(his_channel[:, 0, :, :], his_channel[:, 1, :, :]).squeeze() # (b, M, his_len)
                abs_his_channel = torch.abs(his_channel_complex)  # (b, M, his_len)
                max_abs = abs_his_channel.max(dim = -1, keepdim = True).values  # (b, M, 1)
                his_channel_complex = his_channel_complex / max_abs # (b, M, his_len)
                his_channel_real = his_channel_complex.real
                his_channel_imag = his_channel_complex.imag
                his_channel = torch.stack([his_channel_real, his_channel_imag], dim = 1) # (b, 2, M, his_len)       
            elif args.instance_norm == 'without':
                pass
            else:
                ValueError(f"Unknown instance norm: {args.instance_norm}")    
            
            if args.input_mechanism == 'Channel_independent':
                his_channel = his_channel.permute(0, 2, 1, 3) # (b, M, 2, his_len)
                his_channel = his_channel.reshape(b * M, 2, his_len) # (b * M, 2, his_len)
            elif args.input_mechanism == 'Channel_mixing':
                his_channel = his_channel.reshape(b, -1, his_len) # (b, 2 * M, his_len)
            else:
                raise NotImplementedError
                        
            # eval the network         
            if args.model_name == 'Vanilla_RNN':
                # out_tensor: predicted channel sequence
                out_tensor = model.test_data(x = his_channel.permute(0, 2, 1), pred_len = pre_channel.shape[3], device = device) 
                
                if args.input_mechanism == 'Channel_independent':
                    # (b * M, pre_len, 2)
                    out_tensor = out_tensor.reshape(b, M, pre_len, 2)
                    out_tensor = out_tensor.permute(0, 3, 1, 2) # (b, 2, M, pre_len) 
                else:
                    # (b, pre_len, 2 * M)
                    out_tensor = out_tensor.reshape(b, pre_len, 2, M)
                    out_tensor = out_tensor.permute(0, 2, 3, 1) # (b, 2, M, pre_len)
                    
            elif args.model_name == 'Vanilla_LSTM':
                # out_tensor: predicted channel sequence
                out_tensor = model.test_data(x = his_channel.permute(0, 2, 1), pred_len = pre_channel.shape[3], device = device)
                
                if args.input_mechanism == 'Channel_independent':
                    # (b * M, pre_len, 2)
                    out_tensor = out_tensor.reshape(b, M, pre_len, 2)
                    out_tensor = out_tensor.permute(0, 3, 1, 2) # (b, 2, M, pre_len) 
                else:
                    # (b, pre_len, 2 * M)
                    out_tensor = out_tensor.reshape(b, pre_len, 2, M)
                    out_tensor = out_tensor.permute(0, 2, 3, 1) # (b, 2, M, pre_len)
                    
            elif args.model_name == 'Vanilla_GRU':
                # out_tensor: predicted channel sequence
                out_tensor = model.test_data(x = his_channel.permute(0, 2, 1), pred_len = pre_channel.shape[3], device = device)
                
                if args.input_mechanism == 'Channel_independent':
                    # (b * M, pre_len, 2)
                    out_tensor = out_tensor.reshape(b, M, pre_len, 2)
                    out_tensor = out_tensor.permute(0, 3, 1, 2) # (b, 2, M, pre_len) 
                else:
                    # (b, pre_len, 2 * M)
                    out_tensor = out_tensor.reshape(b, pre_len, 2, M)
                    out_tensor = out_tensor.permute(0, 2, 3, 1) # (b, 2, M, pre_len)
                    
            elif args.model_name == 'Vanilla_Transformer':
                
                if args.input_mechanism == 'Channel_independent':
                    enc_inp = his_channel.permute(0, 2, 1) # (b * M, his_len, 2)
                    dec_inp =  torch.zeros(b * M, pre_len, 2).to(device) # (b * M, pre_len, 2)
                    dec_inp =  torch.cat([enc_inp[:, his_len - args.label_len : his_len, :], dec_inp], dim = 1) # (b * M, label_len + pre_len, 2)
                else:
                    enc_inp = his_channel.permute(0, 2, 1) # (b, his_len, 2 * M)
                    dec_inp =  torch.zeros(b, pre_len, 2 * M).to(device) # (b, pre_len, 2 * M)
                    dec_inp =  torch.cat([enc_inp[:, his_len - args.label_len : his_len, :], dec_inp], dim = 1) # (b, label_len + pre_len, 2 * M)
                
                if args.output_attention:
                    out_tensor = model(enc_inp, dec_inp)[0]
                else:
                    out_tensor = model(enc_inp, dec_inp)
                    
                if args.input_mechanism == 'Channel_independent':
                    # (b * M, pre_len, 2)
                    out_tensor = out_tensor.reshape(b, M, pre_len, 2)
                    out_tensor = out_tensor.permute(0, 3, 1, 2) # (b, 2, M, pre_len) 
                else:
                    # (b, pre_len, 2 * M)
                    out_tensor = out_tensor.reshape(b, pre_len, 2, M)
                    out_tensor = out_tensor.permute(0, 2, 3, 1) # (b, 2, M, pre_len)
                
            elif args.model_name == 'Neural_ODE':
                # out_tensor: predicted channel sequence
                out_tensor = model(x = his_channel, pre_len = pre_len, pre_timeslot = pre_timeslot[0, :], device = device) 
                
                if args.input_mechanism == 'Channel_independent':
                    # (b * M, 2, pre_len) 
                    out_tensor = out_tensor.reshape(b, M, 2, pre_len)
                    out_tensor = out_tensor.permute(0, 2, 1, 3) # (b, 2, M, pre_len)   
                else:
                    # (b, 2 * M, pre_len) 
                    out_tensor = out_tensor.reshape(b, 2, M, pre_len) 
                    
            elif args.model_name == 'Latent_ODE':
                # out_tensor: predicted channel sequence
                out_tensor = model(x = his_channel, pre_len = pre_len, his_timeslot = his_timeslot, pre_timeslot = pre_timeslot[0, :], device = device) 
                
                if args.input_mechanism == 'Channel_independent':
                    # (b * M, 2, pre_len)
                    out_tensor = out_tensor.reshape(b, M, 2, pre_len)
                    out_tensor = out_tensor.permute(0, 2, 1, 3) # (b, 2, M, pre_len)
                else:
                    # (b, 2 * M, pre_len) 
                    out_tensor = out_tensor.reshape(b, 2, M, pre_len) 
                    
            elif args.model_name == 'CT_Transformer':
                                    
                if args.input_mechanism == 'Channel_independent':
                    enc_inp = his_channel.permute(0, 2, 1) # (b * M, his_len, 2)
                    dec_inp =  torch.zeros(b * M, pre_len, 2).to(device) # (b * M, pre_len, 2)
                    dec_inp =  torch.cat([enc_inp[:, his_len - args.label_len : his_len, :], dec_inp], dim = 1) # (b * M, label_len + pre_len, 2)
                else:
                    enc_inp = his_channel.permute(0, 2, 1) # (b, his_len, 2 * M)
                    dec_inp =  torch.zeros(b, pre_len, 2 * M).to(device) # (b, pre_len, 2 * M)
                    dec_inp =  torch.cat([enc_inp[:, his_len - args.label_len : his_len, :], dec_inp], dim = 1) # (b, label_len + pre_len, 2 * M)
                                    
                pre_timeslot = pre_timeslot[0] # (pre_len + 1,)
                pre_timeslot = pre_timeslot[1 :] # (pre_len,)
                pre_timeslot = torch.cat([his_timeslot[his_len - args.label_len : his_len], pre_timeslot], dim = 0) # (label_len + pre_len)
                
                if args.output_attention:
                    out_tensor = model(enc_inp, dec_inp, his_timeslot = his_timeslot, pre_timeslot = pre_timeslot)[0]
                else:
                    out_tensor = model(enc_inp, dec_inp, his_timeslot = his_timeslot, pre_timeslot = pre_timeslot)
               
                if args.input_mechanism == 'Channel_independent':
                    # (b * M, pre_len, 2)
                    out_tensor = out_tensor.reshape(b, M, pre_len, 2)
                    out_tensor = out_tensor.permute(0, 3, 1, 2) # (b, 2, M, pre_len) 
                else:
                    # (b, pre_len, 2 * M)
                    out_tensor = out_tensor.reshape(b, pre_len, 2, M)
                    out_tensor = out_tensor.permute(0, 2, 3, 1) # (b, 2, M, pre_len) 
            
            elif args.model_name == 'CT_Transformer_encoderonly_v0':
                                    
                if args.input_mechanism == 'Channel_independent':
                    enc_inp = his_channel.permute(0, 2, 1) # (b * M, his_len, 2)
                    dec_inp =  torch.zeros(b * M, pre_len, 2).to(device) # (b * M, pre_len, 2)
                    enc_inp =  torch.cat([enc_inp[:, :, :], dec_inp], dim = 1) # (b * M, his + pre_len, 2)
                else:
                    enc_inp = his_channel.permute(0, 2, 1) # (b, his_len, 2 * M)
                    dec_inp =  torch.zeros(b, pre_len, 2 * M).to(device) # (b, pre_len, 2 * M)
                    enc_inp =  torch.cat([enc_inp[:, :, :], dec_inp], dim = 1) # (b, his + pre_len, 2 * M)
                                    
                pre_timeslot = pre_timeslot[0] # (pre_len + 1,)
                pre_timeslot = pre_timeslot[1 :] # (pre_len,)
                pre_timeslot = torch.cat([his_timeslot[:], pre_timeslot], dim = 0) # (his_len + pre_len)
                
                if args.output_attention:
                    out_tensor = model(x_enc = enc_inp, timeslot = pre_timeslot)[0]
                else:
                    out_tensor = model(x_enc = enc_inp, timeslot = pre_timeslot)
               
                if args.input_mechanism == 'Channel_independent':
                    # (b * M, pre_len, 2)
                    out_tensor = out_tensor.reshape(b, M, pre_len, 2)
                    out_tensor = out_tensor.permute(0, 3, 1, 2) # (b, 2, M, pre_len) 
                else:
                    # (b, pre_len, 2 * M)
                    out_tensor = out_tensor.reshape(b, pre_len, 2, M)
                    out_tensor = out_tensor.permute(0, 2, 3, 1) # (b, 2, M, pre_len)
            
            elif args.model_name == 'CT_Transformer_encoderonly_v1':
                                    
                if args.input_mechanism == 'Channel_independent':
                    enc_inp = his_channel.permute(0, 2, 1) # (b * M, his_len, 2)
                else:
                    enc_inp = his_channel.permute(0, 2, 1) # (b, his_len, 2 * M)
                                    
                pre_timeslot = pre_timeslot[0] # (pre_len + 1,)
                pre_timeslot = pre_timeslot[1 :] # (pre_len,)
                
                if args.output_attention:
                    out_tensor = model(x_enc = enc_inp, his_timeslot = his_timeslot, pre_timeslot = pre_timeslot)[0]
                else:
                    out_tensor = model(x_enc = enc_inp, his_timeslot = his_timeslot, pre_timeslot = pre_timeslot)
               
                if args.input_mechanism == 'Channel_independent':
                    # (b * M, pre_len, 2)
                    out_tensor = out_tensor.reshape(b, M, pre_len, 2)
                    out_tensor = out_tensor.permute(0, 3, 1, 2) # (b, 2, M, pre_len) 
                else:
                    # (b, pre_len, 2 * M)
                    out_tensor = out_tensor.reshape(b, pre_len, 2, M)
                    out_tensor = out_tensor.permute(0, 2, 3, 1) # (b, 2, M, pre_len) 
            
            if args.instance_norm == 'norm_2':
                out_tensor_complex = torch.complex(out_tensor[:, 0, :, :], out_tensor[:, 1, :, :]).squeeze() # (b, M, pre_len)
                out_tensor_complex = out_tensor_complex * norm_his_channel # (b, M, pre_len)
                out_tensor_real = out_tensor_complex.real
                out_tensor_imag = out_tensor_complex.imag
                out_tensor = torch.stack([out_tensor_real, out_tensor_imag], dim = 1) # (b, 2, M, his_len)
            elif args.instance_norm == 'norm_max':
                out_tensor_complex = torch.complex(out_tensor[:, 0, :, :], out_tensor[:, 1, :, :]).squeeze() # (b, M, pre_len)
                out_tensor_complex = out_tensor_complex * max_abs # (b, M, pre_len)
                out_tensor_real = out_tensor_complex.real
                out_tensor_imag = out_tensor_complex.imag
                out_tensor = torch.stack([out_tensor_real, out_tensor_imag], dim = 1) # (b, 2, M, his_len) 
            elif args.instance_norm == 'without':
                pass
                            
            nmse_loss, pre_norm = channel_NMSE(h_gt = pre_channel, h_pre = out_tensor)

            running_NMSE[:, 0] += nmse_loss.data.cpu()
            running_pre_norm[:, 0] += pre_norm.data.cpu()
        
    # average loss
    NMSE_losses = running_NMSE / batch_num
    pre_norm = running_pre_norm / batch_num 
    # print results
    print('Test NMSE for each time point:')
    print(NMSE_losses.T)
    print('Test prediction norm:')
    print(pre_norm.T)
    return NMSE_losses


# main function
# run this file to train and evaluate the model
# OUTPUT: MATLAB data file
def main(training_time = args.training_time, epoch_num = args.epoch_num, b = args.batch_size, lr = args.lr, minlr = 1e-12):
    
    # set seed
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # initialize parameters
    version_name = 'inregular_pilot(v' + args.UE_velocity + ',snr' + args.snr + ',lr' + str(lr) + ')'
    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")
    # print basic information
    print(version_name)
    print('device:%s' % device)
    print('batch_size:%d' % b)
    print('lr and minlr:(%e,%e)'%(lr,minlr))
    for arg in vars(args):
        logger.info(f'{arg} = {getattr(args, arg)}')

    # system parameters
    T_his = args.T_his # historical time interval 
    T_pre = args.T_pre # super-resolution time interval
    T0 = 280 # time to separate historical and future channel
    his_len = args.his_len # length of historical channel sequence
    pre_len = args.pre_len # length of predicted channel sequence
    n = args.intepolation_points_num # number of intepolation points
    M = 32 # antenna number
  
    # training set and validation set
    if args.channel_type == '3GPP_38901':
        
        if args.training_samples == '150mats':
            path_train = '../data_generation/3GPP_38901/allpath_speed' + args.UE_velocity + '_total800ms_snr' + args.snr + '_differentbatch_randomTp/train_opt2_150mats'
        elif args.training_samples == '100mats':
            path_train = '../data_generation/3GPP_38901/allpath_speed' + args.UE_velocity + '_total800ms_snr' + args.snr + '_differentbatch_randomTp/train_opt2_100mats'
        elif args.training_samples == '50mats':
            path_train = '../../../../data_generation/3GPP_38901/allpath_speed' + args.UE_velocity + '_total800ms_snr' + args.snr + '_differentbatch_randomTp/train_opt2_doppler_random'
        else:
            raise NotImplementedError
            
        path_eval = '../../../../data_generation/3GPP_38901/allpath_speed' + args.UE_velocity + '_total800ms_snr' + args.snr + '_differentbatch_randomTp/test_opt1_doppler_random'
        
    elif args.channel_type == 'CDL-B':
        path_train = '../../../data_generation/matlab_5G_toolbox_CDL_B/preprocessed_strongestpath_speed80kmh_total1000ms_cband/train'
        path_eval = '../../../data_generation/matlab_5G_toolbox_CDL_B/preprocessed_strongestpath_speed80kmh_total1000ms_cband/test'
    
    elif args.channel_type == 'DeepMIMO_O1':
        path_train = '../../../../../../data_generation/deepmimo/preprocessed_csi_prediction_dataset/train_opt2_90mats'
        path_eval = '../../../../../../data_generation/deepmimo/preprocessed_csi_prediction_dataset/test_opt1_10mats'
    
    elif args.channel_type == 'demo':
        path_train = '../demo_dataset'
        path_eval = '../demo_dataset'
    
    else:
        raise NotImplementedError
     
    if args.intepolate_pilot_method == 'chebyshev':
        intepolate_points = chebyshev_points(0, T_his, n)
        intepolate_points = intepolate_points[::-1]                
    elif args.intepolate_pilot_method == 'uniform' or args.intepolate_pilot_method == 'random':
        intepolate_points = np.arange(T_his / (n + 1), T_his, T_his / (n + 1))
    elif args.intepolate_pilot_method == 'doppler':
        intepolate_points = np.array([2])
        n = 1
    else:
        raise NotImplementedError
    
    intepolate_points = intepolate_points + T0 - T_his   
    
    intepolate_points = np.insert(intepolate_points, n, T0)      
    tmp = intepolate_points
    for his_len_count in range(his_len - 2):
        tmp1 = intepolate_points - T_his * (his_len_count + 1)
        tmp = np.concatenate((tmp1, tmp))    
    intepolate_points = tmp
    intepolate_points = np.insert(intepolate_points, 0, T0 -  T_his * (his_len - 1))
    
    # update his_len
    if args.ifintepolate_pilot:
        his_len = len(intepolate_points)
    
    intepolate_points = np.round(intepolate_points)
    intepolate_points = np.float32(intepolate_points)
    
    print('estimation time slot:')
    print(intepolate_points)
    
    loader = Dataloader(path = path_train, batch_size = b, device = device, M = M, T_his = T_his, T_pre = T_pre, his_len = his_len, pre_len = pre_len, 
                        ifintepolate_pilot = args.ifintepolate_pilot, intepolate_points = intepolate_points, intepolate_pilot_method = args.intepolate_pilot_method)
    eval_loader = Dataloader(path = path_eval, batch_size = b, device = device, M = M, T_his = T_his, T_pre = T_pre, his_len = his_len, pre_len = pre_len,
                             ifintepolate_pilot = args.ifintepolate_pilot, intepolate_points = intepolate_points, intepolate_pilot_method = args.intepolate_pilot_method)    
      
    # normalized time instants
    his_timeslot = intepolate_points / T_his
    his_timeslot = torch.from_numpy(his_timeslot)
    his_timeslot = his_timeslot.to(device)
    his_timeslot.requires_grad = False
   
    loss_train = np.zeros((training_time, epoch_num))
    loss_eval = np.zeros((pre_len, training_time, epoch_num))
    now_lr = np.zeros((training_time, epoch_num))
    
    criterion = torch.nn.MSELoss() # MSE loss

    # first loop for training times
    for tt in range(training_time):
        print('Train %d times' % (tt))
        
        # model initialization
        if args.model_name == 'Vanilla_RNN':
            model = Vanilla_RNN(features = int(args.enc_in) if args.input_mechanism == 'Channel_independent' else int(args.enc_in * M), 
                                input_size = args.rnn_i, hidden_size = args.rnn_h, num_layers = args.rnn_layers)
        elif args.model_name == 'Vanilla_LSTM':
            model = Vanilla_LSTM(features = int(args.enc_in) if args.input_mechanism == 'Channel_independent' else int(args.enc_in * M), 
                                input_size = args.rnn_i, hidden_size = args.rnn_h, num_layers = args.rnn_layers)
        elif args.model_name == 'Vanilla_GRU':
            model = Vanilla_GRU(features = int(args.enc_in) if args.input_mechanism == 'Channel_independent' else int(args.enc_in * M), 
                                input_size = args.rnn_i, hidden_size = args.rnn_h, num_layers = args.rnn_layers)
        elif args.model_name == 'Vanilla_Transformer':
            model = Vanilla_Transformer(
                enc_in = int(args.enc_in) if args.input_mechanism == 'Channel_independent' else int(args.enc_in * M),
                dec_in = int(args.dec_in) if args.input_mechanism == 'Channel_independent' else int(args.dec_in * M), 
                c_out = int(args.c_out) if args.input_mechanism == 'Channel_independent' else int(args.c_out * M), 
                seq_len = his_len, 
                label_len = args.label_len,
                out_len = pre_len, 
                factor = args.factor,
                d_model = args.d_model, 
                n_heads = args.n_heads, 
                e_layers = args.e_layers,
                d_layers = args.d_layers, 
                d_ff = args.d_ff,
                dropout = args.dropout, 
                attn = args.attn,
                embed = args.embed,
                activation = args.activation,
                output_attention = args.output_attention,
                distil = args.distil,
                device = device
            )
        elif args.model_name == 'Neural_ODE':
            model = Neural_ODE(features = int(args.enc_in) if args.input_mechanism == 'Channel_independent' else int(args.enc_in * M),
                          rnn_i = args.rnn_i,
                          rnn_h = args.rnn_h,
                          rnn_layers = args.rnn_layers,
                          rnn_type = args.ode_rnn_type,
                          ode_h = args.ode_h,
                          ode_hidden_layers = args.ode_hidden_layers,
                          odeint_rtol = args.dec_rtol, 
                          odeint_atol = args.dec_atol, 
                          method = args.dec_method,
                          device = device)
        elif args.model_name == 'Latent_ODE':
            model = Latent_ODE(features = int(args.enc_in) if args.input_mechanism == 'Channel_independent' else int(args.enc_in * M),
                          rnn_i = args.rnn_i,
                          rnn_h = args.rnn_h,
                          rnn_layers = args.rnn_layers,
                          rnn_type = args.ode_rnn_type,
                          enc_dec_odefunc_ifshared = args.enc_dec_odefunc_ifshared,
                          enc_ode_h = args.enc_ode_h,
                          enc_ode_hidden_layers = args.enc_ode_hidden_layers,                                             
                          enc_odeint_rtol = args.enc_rtol, 
                          enc_odeint_atol = args.enc_atol, 
                          enc_method = args.enc_method,
                          dec_ode_h = args.dec_ode_h,
                          dec_ode_hidden_layers = args.dec_ode_hidden_layers, 
                          dec_odeint_rtol = args.dec_rtol, 
                          dec_odeint_atol = args.dec_atol, 
                          dec_method = args.dec_method,
                          device = device)
        elif args.model_name == 'CT_Transformer':
            model = CT_Transformer(
                time_emb_kind = args.time_emb_kind,
                omega_0 = args.omega_0,
                enc_ctsa = args.enc_ctsa,
                enc_in = int(args.enc_in) if args.input_mechanism == 'Channel_independent' else int(args.enc_in * M),
                dec_in = int(args.dec_in) if args.input_mechanism == 'Channel_independent' else int(args.dec_in * M), 
                c_out = int(args.c_out) if args.input_mechanism == 'Channel_independent' else int(args.c_out * M), 
                seq_len = his_len, 
                label_len = args.label_len,
                out_len = pre_len, 
                factor = args.factor,
                d_model = args.d_model, 
                n_heads = args.n_heads, 
                e_layers = args.e_layers,
                d_layers = args.d_layers, 
                d_ff = args.d_ff,
                dropout = args.dropout, 
                attn = args.attn,
                ct_attn = args.ct_attn,
                embed = args.embed,
                activation = args.activation,
                output_attention = args.output_attention,
                distil = args.distil,
                continuous_qkv_method = args.continuous_qkv_method,
                derivative_function_type = args.derivative_function_type,
                ode_h = args.dec_ode_h,
                ode_hidden_layers = args.dec_ode_hidden_layers,
                odeint_rtol = args.dec_rtol, 
                odeint_atol = args.dec_atol, 
                method = args.dec_method,
                device = device
            )
        elif args.model_name == 'CT_Transformer_encoderonly_v0':
            model = CT_Transformer_encoderonly_v0(
                time_emb_kind = args.time_emb_kind,
                omega_0 = args.omega_0,
                enc_ctsa = args.enc_ctsa,
                enc_in = int(args.enc_in) if args.input_mechanism == 'Channel_independent' else int(args.enc_in * M),
                dec_in = int(args.dec_in) if args.input_mechanism == 'Channel_independent' else int(args.dec_in * M), 
                c_out = int(args.c_out) if args.input_mechanism == 'Channel_independent' else int(args.c_out * M), 
                seq_len = his_len, 
                label_len = args.label_len,
                out_len = pre_len, 
                factor = args.factor,
                d_model = args.d_model, 
                n_heads = args.n_heads, 
                e_layers = args.e_layers,
                d_layers = args.d_layers, 
                d_ff = args.d_ff,
                dropout = args.dropout, 
                attn = args.attn,
                embed = args.embed,
                activation = args.activation,
                output_attention = args.output_attention,
                distil = args.distil,
                continuous_qkv_method = args.continuous_qkv_method,
                derivative_function_type = args.derivative_function_type,
                ode_h = args.dec_ode_h,
                ode_hidden_layers = args.dec_ode_hidden_layers,
                odeint_rtol = args.dec_rtol, 
                odeint_atol = args.dec_atol, 
                method = args.dec_method,
                device = device
            )
        elif args.model_name == 'CT_Transformer_encoderonly_v1':
            model = CT_Transformer_encoderonly_v1(
                time_emb_kind = args.time_emb_kind,
                omega_0 = args.omega_0,
                enc_ctsa = args.enc_ctsa,
                enc_in = int(args.enc_in) if args.input_mechanism == 'Channel_independent' else int(args.enc_in * M),
                dec_in = int(args.dec_in) if args.input_mechanism == 'Channel_independent' else int(args.dec_in * M), 
                c_out = int(args.c_out) if args.input_mechanism == 'Channel_independent' else int(args.c_out * M), 
                seq_len = his_len, 
                label_len = args.label_len,
                out_len = pre_len, 
                factor = args.factor,
                d_model = args.d_model, 
                n_heads = args.n_heads, 
                e_layers = args.e_layers,
                d_layers = args.d_layers, 
                d_ff = args.d_ff,
                dropout = args.dropout, 
                attn = args.attn,
                embed = args.embed,
                activation = args.activation,
                output_attention = args.output_attention,
                distil = args.distil,
                continuous_qkv_method = args.continuous_qkv_method,
                derivative_function_type = args.derivative_function_type,
                ode_h = args.dec_ode_h,
                ode_hidden_layers = args.dec_ode_hidden_layers,
                odeint_rtol = args.dec_rtol, 
                odeint_atol = args.dec_atol, 
                method = args.dec_method,
                weight_method = args.weight_method,
                device = device
            )
        else:
            ValueError(f"Unknown model name: {args.model_name}")    

        model.to(device)
        # Adam optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr)
        min_nmse = 1e5
       
        ## print parameters
        # for name, param in model.named_parameters():
        #     print('Name:', name, 'Size:', param.size())

        # second loop for training times
        for e in range(epoch_num):
            
            start = time.time()
            
            print('Train %d epoch' % (e))
            # reset the dataloader
            loader.reset()
            eval_loader.reset()
            # judge whether data loading is done
            done = False
            # running loss
            running_loss = 0
            # count batch number
            batch_num = 0
            # running NMSE
            running_NMSE = torch.zeros([pre_len, 1])
            # running norm of prediction channel
            running_pre_norm = torch.zeros([pre_len, 1])
            
            now_lr[tt, e] = optimizer.state_dict()['param_groups'][0]['lr']
            print('This epoch lr is %.10f' % (now_lr[tt, e]))
                        
            while True:
                # read files
                if args.intepolate_pilot_method == 'random':
                    his_channel, pre_channel, pre_timeslot, his_timeslot, done = loader.next_batch()                    
                else:
                    his_channel, pre_channel, pre_timeslot, _, done = loader.next_batch()
                # his_channel: historical channel sequence, size of (b, 2, M, his_len)                     
                # pre_channel: predicted channel sequence, size of (b, 2, M, pre_len)
                # pre_timeslot: predicted time slots, size of (b, pre_len)
                # his_timeslot: historical time slots, size of (b, his_len)
                               
                if done == True:
                    break
                
                # normalized time instants
                pre_timeslot = pre_timeslot / T_his
                pre_timeslot.requires_grad = False
                
                if args.intepolate_pilot_method == 'random':
                    his_timeslot = his_timeslot[0]
                    his_timeslot = his_timeslot / T_his
                    his_timeslot.requires_grad = False
                
                batch_num += 1
                               
                if args.instance_norm == 'norm_2':
                    his_channel_complex = torch.complex(his_channel[:, 0, :, :], his_channel[:, 1, :, :]).squeeze() # (b, M, his_len)
                    norm_his_channel = torch.linalg.norm(his_channel_complex, ord = 2, dim = -1, keepdim = True) # (b, M, 1)
                    his_channel_complex = his_channel_complex / norm_his_channel # (b, M, his_len)
                    his_channel_real = his_channel_complex.real
                    his_channel_imag = his_channel_complex.imag
                    his_channel = torch.stack([his_channel_real, his_channel_imag], dim = 1) # (b, 2, M, his_len)
                elif args.instance_norm == 'norm_max':
                    his_channel_complex = torch.complex(his_channel[:, 0, :, :], his_channel[:, 1, :, :]).squeeze() # (b, M, his_len)
                    abs_his_channel = torch.abs(his_channel_complex)  # (b, M, his_len)
                    max_abs = abs_his_channel.max(dim = -1, keepdim = True).values  # (b, M, 1)
                    his_channel_complex = his_channel_complex / max_abs # (b, M, his_len)
                    his_channel_real = his_channel_complex.real
                    his_channel_imag = his_channel_complex.imag
                    his_channel = torch.stack([his_channel_real, his_channel_imag], dim = 1) # (b, 2, M, his_len)       
                elif args.instance_norm == 'without':
                    pass
                else:
                    ValueError(f"Unknown instance norm: {args.instance_norm}")    

                if args.input_mechanism == 'Channel_independent':
                    his_channel = his_channel.permute(0, 2, 1, 3) # (b, M, 2, his_len)
                    his_channel = his_channel.reshape(b * M, 2, his_len) # (b * M, 2, his_len)
                elif args.input_mechanism == 'Channel_mixing':
                    his_channel = his_channel.reshape(b, -1, his_len) # (b, 2 * M, his_len)
                else:
                    raise NotImplementedError
                
                # train the network
                optimizer.zero_grad()
                
                if args.model_name == 'Vanilla_RNN':
                    # out_tensor: predicted channel sequence
                    out_tensor = model.test_data(x = his_channel.permute(0, 2, 1), pred_len = pre_channel.shape[3], device = device) 
                    
                    if args.input_mechanism == 'Channel_independent':
                        # (b * M, pre_len, 2)
                        out_tensor = out_tensor.reshape(b, M, pre_len, 2)
                        out_tensor = out_tensor.permute(0, 3, 1, 2) # (b, 2, M, pre_len) 
                    else:
                        # (b, pre_len, 2 * M)
                        out_tensor = out_tensor.reshape(b, pre_len, 2, M)
                        out_tensor = out_tensor.permute(0, 2, 3, 1) # (b, 2, M, pre_len)
                        
                elif args.model_name == 'Vanilla_LSTM':
                    # out_tensor: predicted channel sequence
                    out_tensor = model.test_data(x = his_channel.permute(0, 2, 1), pred_len = pre_channel.shape[3], device = device)
                    
                    if args.input_mechanism == 'Channel_independent':
                        # (b * M, pre_len, 2)
                        out_tensor = out_tensor.reshape(b, M, pre_len, 2)
                        out_tensor = out_tensor.permute(0, 3, 1, 2) # (b, 2, M, pre_len) 
                    else:
                        # (b, pre_len, 2 * M)
                        out_tensor = out_tensor.reshape(b, pre_len, 2, M)
                        out_tensor = out_tensor.permute(0, 2, 3, 1) # (b, 2, M, pre_len)
                        
                elif args.model_name == 'Vanilla_GRU':
                    # out_tensor: predicted channel sequence
                    out_tensor = model.test_data(x = his_channel.permute(0, 2, 1), pred_len = pre_channel.shape[3], device = device)
                    
                    if args.input_mechanism == 'Channel_independent':
                        # (b * M, pre_len, 2)
                        out_tensor = out_tensor.reshape(b, M, pre_len, 2)
                        out_tensor = out_tensor.permute(0, 3, 1, 2) # (b, 2, M, pre_len) 
                    else:
                        # (b, pre_len, 2 * M)
                        out_tensor = out_tensor.reshape(b, pre_len, 2, M)
                        out_tensor = out_tensor.permute(0, 2, 3, 1) # (b, 2, M, pre_len)
                        
                elif args.model_name == 'Vanilla_Transformer':
                    
                    if args.input_mechanism == 'Channel_independent':
                        enc_inp = his_channel.permute(0, 2, 1) # (b * M, his_len, 2)
                        dec_inp =  torch.zeros(b * M, pre_len, 2).to(device) # (b * M, pre_len, 2)
                        dec_inp =  torch.cat([enc_inp[:, his_len - args.label_len : his_len, :], dec_inp], dim = 1) # (b * M, label_len + pre_len, 2)
                    else:
                        enc_inp = his_channel.permute(0, 2, 1) # (b, his_len, 2 * M)
                        dec_inp =  torch.zeros(b, pre_len, 2 * M).to(device) # (b, pre_len, 2 * M)
                        dec_inp =  torch.cat([enc_inp[:, his_len - args.label_len : his_len, :], dec_inp], dim = 1) # (b, label_len + pre_len, 2 * M)
                    
                    if args.output_attention:
                        out_tensor = model(enc_inp, dec_inp)[0]
                    else:
                        out_tensor = model(enc_inp, dec_inp)
                        
                    if args.input_mechanism == 'Channel_independent':
                        # (b * M, pre_len, 2)
                        out_tensor = out_tensor.reshape(b, M, pre_len, 2)
                        out_tensor = out_tensor.permute(0, 3, 1, 2) # (b, 2, M, pre_len) 
                    else:
                        # (b, pre_len, 2 * M)
                        out_tensor = out_tensor.reshape(b, pre_len, 2, M)
                        out_tensor = out_tensor.permute(0, 2, 3, 1) # (b, 2, M, pre_len)
                    
                elif args.model_name == 'Neural_ODE':
                    # out_tensor: predicted channel sequence
                    out_tensor = model(x = his_channel, pre_len = pre_len, pre_timeslot = pre_timeslot[0, :], device = device) 
                    
                    if args.input_mechanism == 'Channel_independent':
                        # (b * M, 2, pre_len) 
                        out_tensor = out_tensor.reshape(b, M, 2, pre_len)
                        out_tensor = out_tensor.permute(0, 2, 1, 3) # (b, 2, M, pre_len)   
                    else:
                        # (b, 2 * M, pre_len) 
                        out_tensor = out_tensor.reshape(b, 2, M, pre_len) 
                        
                elif args.model_name == 'Latent_ODE':
                    # out_tensor: predicted channel sequence
                    out_tensor = model(x = his_channel, pre_len = pre_len, his_timeslot = his_timeslot, pre_timeslot = pre_timeslot[0, :], device = device) 
                    
                    if args.input_mechanism == 'Channel_independent':
                        # (b * M, 2, pre_len)
                        out_tensor = out_tensor.reshape(b, M, 2, pre_len)
                        out_tensor = out_tensor.permute(0, 2, 1, 3) # (b, 2, M, pre_len)
                    else:
                        # (b, 2 * M, pre_len) 
                        out_tensor = out_tensor.reshape(b, 2, M, pre_len) 
                        
                elif args.model_name == 'CT_Transformer':
                                        
                    if args.input_mechanism == 'Channel_independent':
                        enc_inp = his_channel.permute(0, 2, 1) # (b * M, his_len, 2)
                        dec_inp =  torch.zeros(b * M, pre_len, 2).to(device) # (b * M, pre_len, 2)
                        dec_inp =  torch.cat([enc_inp[:, his_len - args.label_len : his_len, :], dec_inp], dim = 1) # (b * M, label_len + pre_len, 2)
                    else:
                        enc_inp = his_channel.permute(0, 2, 1) # (b, his_len, 2 * M)
                        dec_inp =  torch.zeros(b, pre_len, 2 * M).to(device) # (b, pre_len, 2 * M)
                        dec_inp =  torch.cat([enc_inp[:, his_len - args.label_len : his_len, :], dec_inp], dim = 1) # (b, label_len + pre_len, 2 * M)
                                        
                    pre_timeslot = pre_timeslot[0] # (pre_len + 1,)
                    pre_timeslot = pre_timeslot[1 :] # (pre_len,)
                    pre_timeslot = torch.cat([his_timeslot[his_len - args.label_len : his_len], pre_timeslot], dim = 0) # (label_len + pre_len)
                    
                    if args.output_attention:
                        out_tensor = model(enc_inp, dec_inp, his_timeslot = his_timeslot, pre_timeslot = pre_timeslot)[0]
                    else:
                        out_tensor = model(enc_inp, dec_inp, his_timeslot = his_timeslot, pre_timeslot = pre_timeslot)
                   
                    if args.input_mechanism == 'Channel_independent':
                        # (b * M, pre_len, 2)
                        out_tensor = out_tensor.reshape(b, M, pre_len, 2)
                        out_tensor = out_tensor.permute(0, 3, 1, 2) # (b, 2, M, pre_len) 
                    else:
                        # (b, pre_len, 2 * M)
                        out_tensor = out_tensor.reshape(b, pre_len, 2, M)
                        out_tensor = out_tensor.permute(0, 2, 3, 1) # (b, 2, M, pre_len)
                
                elif args.model_name == 'CT_Transformer_encoderonly_v0':
                                        
                    if args.input_mechanism == 'Channel_independent':
                        enc_inp = his_channel.permute(0, 2, 1) # (b * M, his_len, 2)
                        dec_inp =  torch.zeros(b * M, pre_len, 2).to(device) # (b * M, pre_len, 2)
                        enc_inp =  torch.cat([enc_inp[:, :, :], dec_inp], dim = 1) # (b * M, his + pre_len, 2)
                    else:
                        enc_inp = his_channel.permute(0, 2, 1) # (b, his_len, 2 * M)
                        dec_inp =  torch.zeros(b, pre_len, 2 * M).to(device) # (b, pre_len, 2 * M)
                        enc_inp =  torch.cat([enc_inp[:, :, :], dec_inp], dim = 1) # (b, his + pre_len, 2 * M)
                                        
                    pre_timeslot = pre_timeslot[0] # (pre_len + 1,)
                    pre_timeslot = pre_timeslot[1 :] # (pre_len,)
                    pre_timeslot = torch.cat([his_timeslot[:], pre_timeslot], dim = 0) # (his_len + pre_len)
                    
                    if args.output_attention:
                        out_tensor = model(x_enc = enc_inp, timeslot = pre_timeslot)[0]
                    else:
                        out_tensor = model(x_enc = enc_inp, timeslot = pre_timeslot)
                   
                    if args.input_mechanism == 'Channel_independent':
                        # (b * M, pre_len, 2)
                        out_tensor = out_tensor.reshape(b, M, pre_len, 2)
                        out_tensor = out_tensor.permute(0, 3, 1, 2) # (b, 2, M, pre_len) 
                    else:
                        # (b, pre_len, 2 * M)
                        out_tensor = out_tensor.reshape(b, pre_len, 2, M)
                        out_tensor = out_tensor.permute(0, 2, 3, 1) # (b, 2, M, pre_len)
                
                elif args.model_name == 'CT_Transformer_encoderonly_v1':
                                        
                    if args.input_mechanism == 'Channel_independent':
                        enc_inp = his_channel.permute(0, 2, 1) # (b * M, his_len, 2)
                    else:
                        enc_inp = his_channel.permute(0, 2, 1) # (b, his_len, 2 * M)
                                        
                    pre_timeslot = pre_timeslot[0] # (pre_len + 1,)
                    pre_timeslot = pre_timeslot[1 :] # (pre_len,)
                    
                    if args.output_attention:
                        out_tensor = model(x_enc = enc_inp, his_timeslot = his_timeslot, pre_timeslot = pre_timeslot)[0]
                    else:
                        out_tensor = model(x_enc = enc_inp, his_timeslot = his_timeslot, pre_timeslot = pre_timeslot)
                   
                    if args.input_mechanism == 'Channel_independent':
                        # (b * M, pre_len, 2)
                        out_tensor = out_tensor.reshape(b, M, pre_len, 2)
                        out_tensor = out_tensor.permute(0, 3, 1, 2) # (b, 2, M, pre_len) 
                    else:
                        # (b, pre_len, 2 * M)
                        out_tensor = out_tensor.reshape(b, pre_len, 2, M)
                        out_tensor = out_tensor.permute(0, 2, 3, 1) # (b, 2, M, pre_len) 
                
                if args.instance_norm == 'norm_2':
                    out_tensor_complex = torch.complex(out_tensor[:, 0, :, :], out_tensor[:, 1, :, :]).squeeze() # (b, M, pre_len)
                    out_tensor_complex = out_tensor_complex * norm_his_channel # (b, M, pre_len)
                    out_tensor_real = out_tensor_complex.real
                    out_tensor_imag = out_tensor_complex.imag
                    out_tensor = torch.stack([out_tensor_real, out_tensor_imag], dim = 1) # (b, 2, M, his_len)
                elif args.instance_norm == 'norm_max':
                    out_tensor_complex = torch.complex(out_tensor[:, 0, :, :], out_tensor[:, 1, :, :]).squeeze() # (b, M, pre_len)
                    out_tensor_complex = out_tensor_complex * max_abs # (b, M, pre_len)
                    out_tensor_real = out_tensor_complex.real
                    out_tensor_imag = out_tensor_complex.imag
                    out_tensor = torch.stack([out_tensor_real, out_tensor_imag], dim = 1) # (b, 2, M, his_len) 
                elif args.instance_norm == 'without':
                    pass
                                        
                nmse_loss, pre_norm = channel_NMSE(h_gt = pre_channel, h_pre = out_tensor) # (pre_len,)
                
                if args.loss_function == 'MSE':
                    loss = criterion(pre_channel, out_tensor) # MSE Loss
                elif args.loss_function == 'NMSE':
                    loss = torch.mean(nmse_loss) # NMSE Loss
                
                running_NMSE[:, 0] += nmse_loss.data.cpu()
                running_pre_norm[:, 0] += pre_norm.data.cpu()
                   
                # gradient back propagation
                loss.backward()
                # parameter optimization
                optimizer.step()
                running_loss += loss.item()
                
                # print(batch_num)

            losses = running_loss / batch_num
            #print results
            loss_train[tt, e] = losses
            print('Training Loss: %.6f' % losses)
            # average loss
            NMSE_losses = running_NMSE / batch_num
            pre_norm = running_pre_norm / batch_num 
            # print results
            print('Training NMSE for each time point:')
            print(NMSE_losses.T)
            print('Training prediction norm:')
            print(pre_norm.T)
            # eval mode, where dropout is off
            model.eval()
            # eval_losses: average loss in evaluation set
            
            NMSE_losses = eval(model, eval_loader, M, his_len, pre_len, b, device, criterion, his_timeslot, T_his)
            loss_eval[:, tt, e] = NMSE_losses.squeeze()
                                      
            if torch.mean(NMSE_losses) < min_nmse:
                min_nmse = torch.mean(NMSE_losses)
                model_name = version_name + '_' + args.channel_type + '_' + args.model_name + '_' + args.loss_function + '_tt' + str(tt) + '_MODEL.pth'
                torch.save(model.state_dict(), model_name)
            
            # train mode
            model.train()

            # save results into mat file            
            mat_name = version_name + '_' + args.channel_type + '_' + args.model_name + '_' + args.loss_function + '_tt' + str(tt) + '_result.mat'
            sio.savemat(mat_name, {
                'loss_train' : loss_train,
                'loss_eval' : loss_eval,
                'now_lr' : now_lr
            })
            
            print('This epoch takes %.1f s' % (time.time() - start))

if __name__ == '__main__':
    main()
