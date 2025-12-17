clear all;
close all;
clc;

load('dataset_1.mat')

batch_list = [11, 19, 27, 32];
train_r_gt = train_r_gt(batch_list, :, :);
train_i_gt = train_i_gt(batch_list, :, :);
train_r = train_r(batch_list, :, :);
train_i = train_i(batch_list, :, :);

save('dataset_2.mat')