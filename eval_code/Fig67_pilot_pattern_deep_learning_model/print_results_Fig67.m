
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

%% Fig. 6 and 7

velocity_list = [20, 40, 60, 80];
GRU_uniform = zeros(length(velocity_list), pre_len);
GRU_chebyshev = zeros(length(velocity_list), pre_len);
Vanilla_transformer_uniform = zeros(length(velocity_list), pre_len);
Vanilla_transformer_chebyshev = zeros(length(velocity_list), pre_len);
Neural_ODE_uniform = zeros(length(velocity_list), pre_len);
Neural_ODE_chebyshev = zeros(length(velocity_list), pre_len);
Latent_ODE_uniform = zeros(length(velocity_list), pre_len);
Latent_ODE_chebyshev = zeros(length(velocity_list), pre_len);
CT_transformer_uniform = zeros(length(velocity_list), pre_len);
CT_transformer_chebyshev = zeros(length(velocity_list), pre_len);

for i = 1 : length(velocity_list)

    load(['Eval_GRU_uniform_', num2str(velocity_list(i)), 'kmh.mat'])
    GRU_uniform(i, :) = squeeze(loss_eval);

    load(['Eval_GRU_chebyshev_', num2str(velocity_list(i)), 'kmh.mat'])
    GRU_chebyshev(i, :) = squeeze(loss_eval);

    load(['Eval_Vanilla_transformer_uniform_', num2str(velocity_list(i)), 'kmh.mat'])
    Vanilla_transformer_uniform(i, :) = squeeze(loss_eval);

    load(['Eval_Vanilla_transformer_chebyshev_', num2str(velocity_list(i)), 'kmh.mat'])
    Vanilla_transformer_chebyshev(i, :) = squeeze(loss_eval);

    load(['Eval_Neural_ODE_uniform_', num2str(velocity_list(i)), 'kmh.mat'])
    Neural_ODE_uniform(i, :) = squeeze(loss_eval);

    load(['Eval_Neural_ODE_chebyshev_', num2str(velocity_list(i)), 'kmh.mat'])
    Neural_ODE_chebyshev(i, :) = squeeze(loss_eval);

    load(['Eval_Latent_ODE_uniform_', num2str(velocity_list(i)), 'kmh.mat'])
    Latent_ODE_uniform(i, :) = squeeze(loss_eval);

    load(['Eval_Latent_ODE_chebyshev_', num2str(velocity_list(i)), 'kmh.mat'])
    Latent_ODE_chebyshev(i, :) = squeeze(loss_eval);

    load(['Eval_CT_transformer_uniform_', num2str(velocity_list(i)), 'kmh.mat'])
    CT_transformer_uniform(i, :) = squeeze(loss_eval);

    load(['Eval_CT_transformer_chebyshev_', num2str(velocity_list(i)), 'kmh.mat'])
    CT_transformer_chebyshev(i, :) = squeeze(loss_eval);

end

figure_num = figure_num + 1;
figure(figure_num)
position = get(figure(figure_num), 'Position'); % 获取图窗口的位置和尺寸
set(gcf, 'Position', [position(1), position(2), 800, 400]); % [x, y, width, height]set(gcf, 'Position', [100, 100, 800, 600]); % [x, y, width, height]
hold on;
plot(1 : 1 : pre_len, 10 * log10(squeeze(GRU_uniform(1, :))), 's-', 'linewidth', 1.5, 'Color', color_g);
plot(1 : 1 : pre_len, 10 * log10(squeeze(Vanilla_transformer_uniform(1, :))), 'd-', 'linewidth', 1.5, 'Color', color_orange);
plot(1 : 1 : pre_len, 10 * log10(squeeze(Neural_ODE_uniform(1, :))), '+-', 'linewidth', 1.5, 'Color', color_p);
plot(1 : 1 : pre_len, 10 * log10(squeeze(Latent_ODE_uniform(1, :))), '*-', 'linewidth', 1.5, 'Color', color_b);
plot(1 : 1 : pre_len, 10 * log10(squeeze(CT_transformer_uniform(1, :))), '^-', 'linewidth', 1.5, 'Color', color_r);
xl1 = xline([2, 4, 6, 8], '--', 'linewidth', 1., 'Color', [0, 0, 0]);
legend('show');
grid on;
xlabel('Prediction time slot $t_{p}$ (ms)', 'FontSize', 15, 'interpreter',' latex');
ylabel('NMSE (dB)', 'FontSize', 15, 'interpreter',' latex');
xlim([1, pre_len])
xticks(1 : 1 : pre_len)
xticklabels({'5', '10','15', '20', '25', '30', '35', '40'})
ylim([-12, -2])
%ylim([-8, 1])
legend('GRU [11]', 'Vanilla transformer [12]', 'Neural ODE [30]', 'Latent ODE [45]', 'Continuous-time transformer', 'Aligned time slot', 'FontSize', 11, 'interpreter', 'latex')
legend('Location', 'northwest');

figure_num = figure_num + 1;
figure(figure_num)
position = get(figure(figure_num), 'Position'); % 获取图窗口的位置和尺寸
set(gcf, 'Position', [position(1), position(2), 800, 400]); % [x, y, width, height]set(gcf, 'Position', [100, 100, 800, 600]); % [x, y, width, height]
hold on;
plot(1 : 1 : pre_len, 10 * log10(squeeze(GRU_chebyshev(1, :))), 's-', 'linewidth', 1.5, 'Color', color_g);
plot(1 : 1 : pre_len, 10 * log10(squeeze(Vanilla_transformer_chebyshev(1, :))), 'd-', 'linewidth', 1.5, 'Color', color_orange);
plot(1 : 1 : pre_len, 10 * log10(squeeze(Neural_ODE_chebyshev(1, :))), '+-', 'linewidth', 1.5, 'Color', color_p);
plot(1 : 1 : pre_len, 10 * log10(squeeze(Latent_ODE_chebyshev(1, :))), '*-', 'linewidth', 1.5, 'Color', color_b);
plot(1 : 1 : pre_len, 10 * log10(squeeze(CT_transformer_chebyshev(1, :))), '^-', 'linewidth', 1.5, 'Color', color_r);
xl2 = xline([4, 8, 38 / 5], '--', 'linewidth', 1., 'Color', [0, 0, 0]);
legend('show');
grid on;
xlabel('Prediction time slot $t_{p}$ (ms)', 'FontSize', 15, 'interpreter',' latex');
ylabel('NMSE (dB)', 'FontSize', 15, 'interpreter',' latex');
xlim([1, pre_len])
xticks(1 : 1 : pre_len)
xticklabels({'5', '10','15', '20', '25', '30', '35', '40'})
ylim([-12, -2])
%ylim([-8, 1])
legend('GRU [11]', 'Vanilla transformer [12]', 'Neural ODE [30]', 'Latent ODE [45]', 'Continuous-time transformer', 'Aligned time slot', 'FontSize', 11, 'interpreter', 'latex')
legend('Location', 'northwest');

figure_num = figure_num + 1;
figure(figure_num)
position = get(figure(figure_num), 'Position'); % 获取图窗口的位置和尺寸
set(gcf, 'Position', [position(1), position(2), 800, 400]); % [x, y, width, height]set(gcf, 'Position', [100, 100, 800, 600]); % [x, y, width, height]
hold on;
plot(1 : 1 : pre_len, 10 * log10(squeeze(GRU_uniform(3, :))), 's-', 'linewidth', 1.5, 'Color', color_g);
plot(1 : 1 : pre_len, 10 * log10(squeeze(Vanilla_transformer_uniform(3, :))), 'd-', 'linewidth', 1.5, 'Color', color_orange);
plot(1 : 1 : pre_len, 10 * log10(squeeze(Neural_ODE_uniform(3, :))), '+-', 'linewidth', 1.5, 'Color', color_p);
plot(1 : 1 : pre_len, 10 * log10(squeeze(Latent_ODE_uniform(3, :))), '*-', 'linewidth', 1.5, 'Color', color_b);
plot(1 : 1 : pre_len, 10 * log10(squeeze(CT_transformer_uniform(3, :))), '^-', 'linewidth', 1.5, 'Color', color_r);
xl1 = xline([2, 4, 6, 8], '--', 'linewidth', 1., 'Color', [0, 0, 0]);
legend('show');
grid on;
xlabel('Prediction time slot $t_{p}$ (ms)', 'FontSize', 15, 'interpreter',' latex');
ylabel('NMSE (dB)', 'FontSize', 15, 'interpreter',' latex');
xlim([1, pre_len])
xticks(1 : 1 : pre_len)
xticklabels({'5', '10','15', '20', '25', '30', '35', '40'})
%ylim([-12, -2])
ylim([-8, 1])
legend('GRU [11]', 'Vanilla transformer [12]', 'Neural ODE [30]', 'Latent ODE [45]', 'Continuous-time transformer', 'Aligned time slot', 'FontSize', 11, 'interpreter', 'latex')
legend('Location', 'northwest');

figure_num = figure_num + 1;
figure(figure_num)
position = get(figure(figure_num), 'Position'); % 获取图窗口的位置和尺寸
set(gcf, 'Position', [position(1), position(2), 800, 400]); % [x, y, width, height]set(gcf, 'Position', [100, 100, 800, 600]); % [x, y, width, height]
hold on;
plot(1 : 1 : pre_len, 10 * log10(squeeze(GRU_chebyshev(3, :))), 's-', 'linewidth', 1.5, 'Color', color_g);
plot(1 : 1 : pre_len, 10 * log10(squeeze(Vanilla_transformer_chebyshev(3, :))), 'd-', 'linewidth', 1.5, 'Color', color_orange);
plot(1 : 1 : pre_len, 10 * log10(squeeze(Neural_ODE_chebyshev(3, :))), '+-', 'linewidth', 1.5, 'Color', color_p);
plot(1 : 1 : pre_len, 10 * log10(squeeze(Latent_ODE_chebyshev(3, :))), '*-', 'linewidth', 1.5, 'Color', color_b);
plot(1 : 1 : pre_len, 10 * log10(squeeze(CT_transformer_chebyshev(3, :))), '^-', 'linewidth', 1.5, 'Color', color_r);
xl2 = xline([4, 8, 38 / 5], '--', 'linewidth', 1., 'Color', [0, 0, 0]);
legend('show');
grid on;
xlabel('Prediction time slot $t_{p}$ (ms)', 'FontSize', 15, 'interpreter',' latex');
ylabel('NMSE (dB)', 'FontSize', 15, 'interpreter',' latex');
xlim([1, pre_len])
xticks(1 : 1 : pre_len)
xticklabels({'5', '10','15', '20', '25', '30', '35', '40'})
%ylim([-12, -2])
ylim([-8, 1])
legend('GRU [11]', 'Vanilla transformer [12]', 'Neural ODE [30]', 'Latent ODE [45]', 'Continuous-time transformer', 'Aligned time slot', 'FontSize', 11, 'interpreter', 'latex')
legend('Location', 'northwest');

figure_num = figure_num + 1;
figure(figure_num)
hold on;
plot(1 : 1 : length(velocity_list), 10 * log10(squeeze(mean(GRU_uniform(:, :), 2))), 's-', 'linewidth', 1.5, 'Color', color_g);
plot(1 : 1 : length(velocity_list), 10 * log10(squeeze(mean(Vanilla_transformer_uniform(:, :), 2))), 'd-', 'linewidth', 1.5, 'Color', color_orange);
plot(1 : 1 : length(velocity_list), 10 * log10(squeeze(mean(Neural_ODE_uniform(:, :), 2))), '+-', 'linewidth', 1.5, 'Color', color_p);
plot(1 : 1 : length(velocity_list), 10 * log10(squeeze(mean(Latent_ODE_uniform(:, :), 2))), '*-', 'linewidth', 1.5, 'Color', color_b);
plot(1 : 1 : length(velocity_list), 10 * log10(squeeze(mean(CT_transformer_uniform(:, :), 2))), '^-', 'linewidth', 1.5, 'Color', color_r);
grid on;
xlabel('UE velocity $v$ (km/h)', 'FontSize', 14, 'interpreter',' latex');
ylabel('NMSE (dB)', 'FontSize', 14, 'interpreter',' latex');
xlim([1, 4])
xticks(1 : 1 : 4)
xticklabels({'20', '40', '60', '80'})
ylim([-9, 0])
legend('GRU [11]', 'Vanilla transformer [12]', 'Neural ODE [30]', 'Latent ODE [45]', 'Continuous-time transformer', 'FontSize', 12, 'interpreter', 'latex')
legend('Location', 'southeast');

figure_num = figure_num + 1;
figure(figure_num)
hold on;
plot(1 : 1 : length(velocity_list), 10 * log10(squeeze(mean(GRU_chebyshev(:, :), 2))), 's-', 'linewidth', 1.5, 'Color', color_g);
plot(1 : 1 : length(velocity_list), 10 * log10(squeeze(mean(Vanilla_transformer_chebyshev(:, :), 2))), 'd-', 'linewidth', 1.5, 'Color', color_orange);
plot(1 : 1 : length(velocity_list), 10 * log10(squeeze(mean(Neural_ODE_chebyshev(:, :), 2))), '+-', 'linewidth', 1.5, 'Color', color_p);
plot(1 : 1 : length(velocity_list), 10 * log10(squeeze(mean(Latent_ODE_chebyshev(:, :), 2))), '*-', 'linewidth', 1.5, 'Color', color_b);
plot(1 : 1 : length(velocity_list), 10 * log10(squeeze(mean(CT_transformer_chebyshev(:, :), 2))), '^-', 'linewidth', 1.5, 'Color', color_r);
grid on;
xlabel('UE velocity $v$ (km/h)', 'FontSize', 14, 'interpreter',' latex');
ylabel('NMSE (dB)', 'FontSize', 14, 'interpreter',' latex');
xlim([1, 4])
xticks(1 : 1 : 4)
xticklabels({'20', '40', '60', '80'})
ylim([-9, 0])
legend('GRU [11]', 'Vanilla transformer [12]', 'Neural ODE [30]', 'Latent ODE [45]', 'Continuous-time transformer', 'FontSize', 12, 'interpreter', 'latex')
legend('Location', 'northwest');








