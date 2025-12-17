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

data_idx = 1;
batch_idx = 11; % (11, 19, 22, 27, 32)
Space_vec = Space_basis(4, 4); % M x M 维度的空域双对角DFT矩阵
M = 32;
b = 256;
T_sum = 800;
pre_len = 200;
his_len = 8;
T_o = 40;
T_p = 5;

load('Eval_Neural_ODE.mat')
neural_ode_real = squeeze(pre_nodes(:, 1, :, :)); % (B, M, pre_len)
neural_ode_imag = squeeze(pre_nodes(:, 2, :, :));

load('Eval_Latent_ODE.mat')
latent_ode_real = squeeze(pre_nodes(:, 1, :, :)); % (B, M, pre_len)
latent_ode_imag = squeeze(pre_nodes(:, 2, :, :));

load('Eval_CT_transformer_hybrid.mat')
% calculate NMSE
gt_nodes_complex = squeeze(gt_nodes(:, 1, :, :)) + 1j * squeeze(gt_nodes(:, 2, :, :)); % (B, M, pre_len)
pre_nodes_complex = squeeze(pre_nodes(:, 1, :, :)) + 1j * squeeze(pre_nodes(:, 2, :, :)); % (B, M, pre_len)
diff_norm = squeeze(vecnorm(gt_nodes_complex - pre_nodes_complex, 2, 2) .^ 2); % 向量差的二范数平方
gt_norm = squeeze(vecnorm(gt_nodes_complex, 2, 2) .^ 2); % h_gt 的二范数平方
NMSE = mean(diff_norm ./ gt_norm, 1) % 沿 dim = 1 取均值 (pre_len)

CT_transformer_real = squeeze(pre_nodes(:, 1, :, :)); % (B, M, pre_len)
CT_transformer_imag = squeeze(pre_nodes(:, 2, :, :)); % (B, M, pre_len)
gt_real = squeeze(gt_nodes(:, 1, :, :)); % (B, M, pre_len)
gt_imag = squeeze(gt_nodes(:, 2, :, :)); % (B, M, pre_len)
gt_complex = gt_real + 1j * gt_imag;

figure_num = 0;

for b_idx = 1 : size(CT_transformer_real, 1)

    train_timesum = sum(squeeze(abs(gt_complex(b_idx, :, :))), 2);    
    [~, angle_idx] = max(train_timesum);

    figure_num = figure_num + 1;
    figure(figure_num)
    plot(1 : 1 : pre_len, squeeze(neural_ode_real(b_idx, angle_idx, :)), '-.', 'linewidth', 1.5, 'Color', color_p), hold on;
    plot(1 : 1 : pre_len, squeeze(latent_ode_real(b_idx, angle_idx, :)), ':', 'linewidth', 1.5, 'Color', color_b), hold on;
    plot(1 : 1 : pre_len, squeeze(CT_transformer_real(b_idx, angle_idx, :)), '--', 'linewidth', 1.5, 'Color', color_r), hold on;
    plot(1 : 1 : pre_len, squeeze(gt_real(b_idx, angle_idx, :)), '-', 'linewidth', 1.5, 'Color', color_gray), hold on;
    xlabel('Prediction time slot $t_{p}$ (ms)', 'FontSize', 14, 'interpreter',' latex')
    ylabel('Real part of channel element', 'FontSize', 14, 'interpreter',' latex')
    grid on;
    xlim([1, pre_len])
    ylim([-6, 6])
    xticks([25 : 25 : pre_len])
    xticklabels({'5', '10', '15', '20', '25', '30', '35', '40'})
    legend('Neural ODE [30]', 'Latent ODE [45]', 'Continuous-time transformer', 'Ground truth', 'FontSize', 12, 'interpreter', 'latex')
    legend('Location', 'southwest');
    
    figure_num = figure_num + 1;
    figure(figure_num)
    plot(1 : 1 : pre_len, squeeze(neural_ode_imag(b_idx, angle_idx, :)), '-.', 'linewidth', 1.5, 'Color', color_p), hold on;
    plot(1 : 1 : pre_len, squeeze(latent_ode_imag(b_idx, angle_idx, :)), ':', 'linewidth', 1.5, 'Color', color_b), hold on;
    plot(1 : 1 : pre_len, squeeze(CT_transformer_imag(b_idx, angle_idx, :)), '--', 'linewidth', 1.5, 'Color', color_r), hold on;
    plot(1 : 1 : pre_len, squeeze(gt_imag(b_idx, angle_idx, :)), '-', 'linewidth', 1.5, 'Color', color_gray), hold on;
    xlabel('Prediction time slot $t_{p}$ (ms)', 'FontSize', 14, 'interpreter',' latex')
    ylabel('Imaginary part of channel element', 'FontSize', 14, 'interpreter',' latex')
    grid on;
    xlim([1, pre_len])   
    ylim([-6, 6])
    xticks([25 : 25 : pre_len])
    xticklabels({'5', '10', '15', '20', '25', '30', '35', '40'})
    legend('Neural ODE [30]', 'Latent ODE [45]', 'Continuous-time transformer', 'Ground truth', 'FontSize', 12, 'interpreter', 'latex')
    legend('Location', 'southwest');

end