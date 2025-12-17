from os import listdir
from os.path import isfile, join
import scipy.io as sio
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import sys


class Dataloader():
    # initialization
    def __init__(self, path = '', batch_size = 64, device = 'cpu', 
                 M = 32, T_sum = 800, T_his = 40, T_pre = 5, his_len = 16, pre_len = 39,
                 ifintepolate_pilot = 'No', intepolate_points = np.array([]), intepolate_pilot_method = 'chebyshev'):

        self.batch_size = batch_size
        self.device = device
        self.M = M
        self.T_sum = T_sum # sum time sampling point
        self.T_his = T_his # historical time interval 
        self.T_pre = T_pre # super-resolution time interval
        self.his_len = his_len # length of historical channel sequence
        self.pre_len = pre_len # length of predicted channel sequence
        self.ifintepolate_pilot = ifintepolate_pilot
        self.intepolate_points = intepolate_points
        self.intepolate_pilot_method = intepolate_pilot_method

        # count file names
        self.files = [
            join(path, f) for f in listdir(path) if isfile(join(path, f))
        ]
        for i, f in enumerate(self.files):
            if not f.split('.')[-1] == 'mat':
                del (self.files[i])
        self.reset()

    # reset buffers
    def reset(self):
        self.unvisited_files = [f for f in self.files]
        # real and imag
        self.buffer_train = np.zeros((0, 2, self.M, self.his_len))  
        self.buffer_test = np.zeros((0, 2, self.M, self.pre_len))  
        self.buffer_pre_timeslot = np.zeros((0, self.pre_len + 1))   
        self.buffer_random_timeslot = np.zeros((0, 29))
    
    def channelnorm(H):
        H = H / np.sqrt(np.mean(np.abs(H)**2))
        return H
    

    def check_rows_identical(self, arr):
        if not np.all(arr == arr[0, :]):
            raise ValueError("There exist different elements among different rows!")
        
    # load data from .mat
    def load(self, file):
        data = sio.loadmat(file)
        
        train_r = data['train_r'] # (b, M, T_sum)
        train_i = data['train_i']
        
        train_r_gt = data['train_r_gt']
        train_i_gt = data['train_i_gt']
        
        # random_timeslot is for random pilot
        random_timeslot = data['his_timeslot'] - 1 # (b, his_len)
        random_timeslot = random_timeslot.astype(int)
        random_timeslot = random_timeslot[:, : 29]
        random_timeslot_tmp = np.expand_dims(random_timeslot, axis = 1) # (b, 1, his_len)
        random_timeslot_tmp = np.broadcast_to(random_timeslot_tmp, (256, self.M, 29)) # (b, M, his_len)
        
        pre_timeslot = data['pre_timeslot'] - 1 # (b, pre_len)
        pre_timeslot = pre_timeslot.astype(int)
        pre_timeslot = pre_timeslot[:, : self.pre_len]
        pre_timeslot_tmp = np.expand_dims(pre_timeslot, axis = 1) # (b, 1, pre_len)
        pre_timeslot_tmp = np.broadcast_to(pre_timeslot_tmp, (256, self.M, self.pre_len)) # (b, M, pre_len)
        
        # train_complex = train_r + 1j * train_i
        # norms = np.sqrt(np.mean(np.abs(train_complex)**2, axis = 2, keepdims = True))
        # train_normalized = train_complex / norms

        # train_r = train_normalized.real
        # train_i = train_normalized.imag
        
        if self.ifintepolate_pilot == False:
            his_channel = np.stack((train_r[:, :, 0 : self.his_len * self.T_his : self.T_his], train_i[:, :, 0 : self.his_len * self.T_his : self.T_his]), axis = 1) # (b, 2, M, his_len)
        elif self.ifintepolate_pilot == True:
            if self.intepolate_pilot_method == 'random':
                his_channel = np.stack((np.take_along_axis(train_r, random_timeslot_tmp, axis = 2), 
                                        np.take_along_axis(train_i, random_timeslot_tmp, axis = 2)), axis = 1) # (b, 2, M, his_len)       
            else: # chebyshev or uniform
                his_channel = np.stack((train_r[:, :, self.intepolate_points.astype(int)], train_i[:, :, self.intepolate_points.astype(int)]), axis = 1) # (b, 2, M, his_len)
        else:
            raise NotImplementedError
        
        pre_channel = np.stack((np.take_along_axis(train_r_gt, pre_timeslot_tmp, axis = 2), 
                                np.take_along_axis(train_i_gt, pre_timeslot_tmp, axis = 2)), axis = 1) # (b, 2, M, pre_len)        
                
        '''
        # channel visualization
        
        plt.figure(figsize = (10, 6))
        plt.plot(np.arange(0, 800), train_r_gt[0, 1, :], '-', label = 'Ground truth')
        plt.plot(self.intepolate_points.astype(int), his_channel[0, 0, 1, :], 'o', label = 'Estimated channel')
        plt.plot(pre_timeslot_tmp[0, 1, :], pre_channel[0, 0, 1, :], 's', label = 'Predicted target')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('Real part of channel')
        plt.legend()
        plt.grid(True)
        
        plt.savefig('Channel visualization.png')
        
        print("Have saved Channel visualization.png!")
        '''
                
        return his_channel, pre_channel, np.concatenate((self.intepolate_points[-1].astype(int) * np.ones((256, 1)), pre_timeslot), axis = 1), random_timeslot

    def next_batch(self):
        # serial load data
        done = False
        while self.buffer_train.shape[0] < self.batch_size:
            # if finishing load data
            if len(self.unvisited_files) == 0:
                done = True
                break
            his_channel, pre_channel, pre_timeslot, random_timeslot = self.load(self.unvisited_files.pop(0))

            # load data into buffers
            self.buffer_train = np.concatenate((self.buffer_train, his_channel), axis = 0)
            self.buffer_test = np.concatenate((self.buffer_test, pre_channel), axis = 0)
            self.buffer_pre_timeslot = np.concatenate((self.buffer_pre_timeslot, pre_timeslot), axis = 0)
            self.buffer_random_timeslot = np.concatenate((self.buffer_random_timeslot, random_timeslot), axis = 0)

        # get data from buffers
        out_size = min(self.batch_size, self.buffer_train.shape[0])

        batch_his_channel = self.buffer_train[0 : out_size, :, :, :]
        batch_pre_channel = self.buffer_test[0 : out_size, :, :, :]
        batch_pre_timeslot = self.buffer_pre_timeslot[0 : out_size, :]
        batch_random_timeslot = self.buffer_random_timeslot[0 : out_size, :]
        
        # delete readed data in buffer
        self.buffer_train = np.delete(self.buffer_train, np.s_[0 : out_size], 0)
        self.buffer_test = np.delete(self.buffer_test, np.s_[0 : out_size], 0)
        self.buffer_pre_timeslot = np.delete(self.buffer_pre_timeslot, np.s_[0 : out_size], 0)
        self.buffer_random_timeslot = np.delete(self.buffer_random_timeslot, np.s_[0 : out_size], 0)
        
        if done == False:
            try:
                self.check_rows_identical(arr = batch_pre_timeslot)
            except ValueError as e:
                print("There exist different elements among different rows for pre_timeslot:", e)
                sys.exit(1)
            try:
                self.check_rows_identical(arr = batch_random_timeslot)
            except ValueError as e:
                print("There exist different elements among different rows for random_timeslot:", e)
                sys.exit(1)

        # format transformation for reducing overhead
        batch_his_channel = np.float32(batch_his_channel)
        batch_pre_channel = np.float32(batch_pre_channel)
        batch_pre_timeslot = np.float32(batch_pre_timeslot)
        batch_random_timeslot = np.float32(batch_random_timeslot)

        # return data
        return torch.from_numpy(batch_his_channel).to(self.device), \
            torch.from_numpy(batch_pre_channel).to(self.device), \
            torch.from_numpy(batch_pre_timeslot).to(self.device), \
            torch.from_numpy(batch_random_timeslot).to(self.device), \
            done
