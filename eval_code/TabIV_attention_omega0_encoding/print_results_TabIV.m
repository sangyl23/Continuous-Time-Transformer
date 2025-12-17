
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
pre_len = 8;

%% Table IV

load('Eval_standard_attention.mat')
standard_attention = 10 * log10(mean(loss_eval));
load('Eval_without_te.mat')
without_te = 10 * log10(mean(loss_eval));
wo_enc = loss_eval;
load('Eval_with_pe.mat')
with_pe = 10 * log10(mean(loss_eval));
pe = loss_eval;
load('Eval_omega0_1.mat')
omega0_1 = 10 * log10(mean(loss_eval));
load('Eval_omega0_10.mat')
omega0_10 = 10 * log10(mean(loss_eval));
load('Eval_omega0_30.mat')
omega0_30 = 10 * log10(mean(loss_eval));
hfpe = loss_eval;
load('Eval_omega0_50.mat')
omega0_50 = 10 * log10(mean(loss_eval));

fprintf(['NMSE for standard attention is ', num2str(standard_attention), ' dB'])
fprintf(['\nNMSE for omega0=1 is ', num2str(omega0_1), ' dB'])
fprintf(['\nNMSE for omega0=10 is ', num2str(omega0_10), ' dB'])
fprintf(['\nNMSE for omega0=30 is ', num2str(omega0_30), ' dB'])
fprintf(['\nNMSE for omega0=50 is ', num2str(omega0_50), ' dB'])
fprintf(['\nNMSE for without temporal encoding is ', num2str(without_te), ' dB'])
fprintf(['\nNMSE for with positional encoding is ', num2str(with_pe), ' dB'])

figure_num = figure_num + 1;
figure(figure_num)
hold on;
plot(1 : 1 : pre_len, 10 * log10(wo_enc), '+-', 'linewidth', 1.5, 'Color', color_p);
plot(1 : 1 : pre_len, 10 * log10(pe), '*-', 'linewidth', 1.5, 'Color', color_b);
plot(1 : 1 : pre_len, 10 * log10(hfpe), '^-', 'linewidth', 1.5, 'Color', color_r);
xl2 = xline([4, 8, 38 / 5], '--', 'linewidth', 1., 'Color', [0, 0, 0]);
legend('show');
grid on;
xlabel('Prediction time slot $t_{p}$ (ms)', 'FontSize', 14, 'interpreter',' latex');
ylabel('NMSE (dB)', 'FontSize', 14, 'interpreter',' latex');
xlim([1, pre_len])
xticks(1 : 1 : pre_len)
xticklabels({'5', '10','15', '20', '25', '30', '35', '40'})
%ylim([-12, -2])
ylim([-8, 1])
legend('Without any encoding', 'With positional encoding', 'With high-frequency temporal encoding', 'Aligned time slot', 'FontSize', 12, 'interpreter', 'latex')
legend('Location', 'northwest');



