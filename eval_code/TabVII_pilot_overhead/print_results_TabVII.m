
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

%% Tab xx

load('Eval_uniform_pilot_overhead29.mat')
uniform_29 = 10 * log10(mean(loss_eval));
load('Eval_uniform_pilot_overhead36.mat')
uniform_36 = 10 * log10(mean(loss_eval));
load('Eval_uniform_pilot_overhead43.mat')
uniform_43 = 10 * log10(mean(loss_eval));
load('Eval_uniform_pilot_overhead50.mat')
uniform_50 = 10 * log10(mean(loss_eval));
load('Eval_random_pilot_overhead29.mat')
random_29 = 10 * log10(mean(loss_eval));
load('Eval_random_pilot_overhead36.mat')
random_36 = 10 * log10(mean(loss_eval));
load('Eval_random_pilot_overhead43.mat')
random_43 = 10 * log10(mean(loss_eval));
load('Eval_random_pilot_overhead50.mat')
random_50 = 10 * log10(mean(loss_eval));
load('Eval_chebyshev_pilot_overhead29.mat')
chebyshev_29 = 10 * log10(mean(loss_eval));


fprintf(['NMSE for 29 uniform pilots is ', num2str(uniform_29), ' dB'])
fprintf(['\nNMSE for 36 uniform pilots is ', num2str(uniform_36), ' dB'])
fprintf(['\nNMSE for 43 uniform pilots is ', num2str(uniform_43), ' dB'])
fprintf(['\nNMSE for 50 uniform pilots is ', num2str(uniform_50), ' dB'])
fprintf(['\nNMSE for 29 random pilots is ', num2str(random_29), ' dB'])
fprintf(['\nNMSE for 36 random pilots is ', num2str(random_36), ' dB'])
fprintf(['\nNMSE for 43 random pilots is ', num2str(random_43), ' dB'])
fprintf(['\nNMSE for 50 random pilots is ', num2str(random_50), ' dB'])
fprintf(['\nNMSE for 29 chebyshev pilots is ', num2str(chebyshev_29), ' dB'])



