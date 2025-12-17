
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

%% Fig. 9

GRU_snr = zeros(1, 4);
Vanilla_transformer_snr = zeros(1, 4);
Neural_ODE_snr = zeros(1, 4);
Latent_ODE_snr = zeros(1, 4);
CT_transformer_snr = zeros(1, 4);

snr_list = {'0dB', '5dB', '10dB', '15dB'};

for i = 1 : 4
    load(['Eval_GRU_', snr_list{i}, '.mat'])
    GRU_snr(i) = 10 * log10(mean(loss_eval));
    load(['Eval_Vanilla_transformer_', snr_list{i}, '.mat'])
    Vanilla_transformer_snr(i) = 10 * log10(mean(loss_eval));
    load(['Eval_Neural_ODE_', snr_list{i}, '.mat'])
    Neural_ODE_snr(i) = 10 * log10(mean(loss_eval));
    load(['Eval_Latent_ODE_', snr_list{i}, '.mat'])
    Latent_ODE_snr(i) = 10 * log10(mean(loss_eval));
    load(['Eval_CT_transformer_hybrid_', snr_list{i}, '.mat'])
    CT_transformer_snr(i) = 10 * log10(mean(loss_eval));
end

figure_num = figure_num + 1;
figure(figure_num)
hold on;
plot(1 : 1 : 4, GRU_snr, 's-', 'linewidth', 1.5, 'Color', color_g);
plot(1 : 1 : 4, Vanilla_transformer_snr, 'd-', 'linewidth', 1.5, 'Color', color_orange);
plot(1 : 1 : 4, Neural_ODE_snr, '+-', 'linewidth', 1.5, 'Color', color_p);
plot(1 : 1 : 4, Latent_ODE_snr, '*-', 'linewidth', 1.5, 'Color', color_b);
plot(1 : 1 : 4, CT_transformer_snr, '^-', 'linewidth', 1.5, 'Color', color_r);
grid on;
xlabel('SNR (dB)', 'FontSize', 14, 'interpreter',' latex');
ylabel('NMSE (dB)', 'FontSize', 14, 'interpreter',' latex');
xlim([1, 4])
xticks(1 : 1 : 4)
xticklabels({'0', '5', '10', '15'})
%ylim([-9, 0])
legend('GRU [11]', 'Vanilla transformer [12]', 'Neural ODE [30]', 'Latent ODE [45]', 'Continuous-time transformer', 'FontSize', 12, 'interpreter', 'latex')
legend('Location', 'northeast');



