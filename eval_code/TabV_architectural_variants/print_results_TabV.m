
clear all, close all, clc;

%
color_b = [0 0.4470 0.7410];
color_r = [1 0 0];
color_y = [0.9290 0.6940 0.1250];
color_p = [0.4940 0.1840 0.5560];
color_g = [0.4660 0.6740 0.1880];    
color_deepred = [0.6350 0.0780 0.1840];
color_orange = [0.8500 0.3250 0.0980];
color_lightblue = [0.3010 0.7450 0.9330];    % 浅蓝色
color_magenta = [0.6350 0.0780 0.9800];      % 品红色
color_lime = [0.8500 0.9000 0.0980];         % 淡绿色
color_teal = [0.2500 0.4500 0.4500];         % 蓝绿色
color_gold = [0.8540 0.6470 0.1250];         % 金黄色
color_gray = [0.5 0.5 0.5];                  % 中灰色
color_olive = [0.5960 0.7740 0.1840];        % 橄榄绿色
color_maroon = [0.5560 0.1840 0.2940];       % 栗色

my_colors = [
    1, 0, 0;     % 红色
    0, 1, 0;     % 绿色
    0, 0, 1;     % 蓝色 
    1, 1, 0;     % 黄色
    0, 1, 1;     % 青色
    1, 0, 1;     % 洋红色
    1, 0.5, 0;   % 橙色
    0.5, 0, 0.5; % 紫色
    0.6, 0.3, 0; % 棕色
    0.5, 0.5, 0.5 % 灰色
];

figure_num = 0;

%% Table V

load('Eval_qkv_interpolation.mat')
qkv_interpolation = 10 * log10(mean(loss_eval));
load('Eval_qk_interpolation_v_ODE.mat')
qk_interpolation_v_ODE = 10 * log10(mean(loss_eval));
load('Eval_qkv_ODE.mat')
qkv_ODE = 10 * log10(mean(loss_eval));
load('Eval_sparse_attention.mat')
sparse_attention = 10 * log10(mean(loss_eval));
load('Eval_enc_only.mat')
enc_only = 10 * log10(mean(loss_eval));
load('Eval_enc_ctsa.mat')
enc_ctsa = 10 * log10(mean(loss_eval));

fprintf(['NMSE for (q,k,v)-interpolation is ', num2str(qkv_interpolation), ' dB'])
fprintf(['\nNMSE for (q,k)-interpolation, v-ODE is ', num2str(qk_interpolation_v_ODE), ' dB'])
fprintf(['\nNMSE for (q,k,v)-ODE is ', num2str(qkv_ODE), ' dB'])
fprintf(['\nNMSE for sparse attention is ', num2str(sparse_attention), ' dB'])
fprintf(['\nNMSE for enc-only is ', num2str(enc_only), ' dB'])
fprintf(['\nNMSE for ct-enc is ', num2str(enc_ctsa), ' dB'])



