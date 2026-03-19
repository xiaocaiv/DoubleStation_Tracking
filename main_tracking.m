clear; close all; clc;

%% 随机种子设置
rng(1);

%% 参数设置
dt = 1;      % 采样周期，单位：秒
T = 100;     % 总仿真时间，单位：秒
t = 0:dt:T;  % 时间序列向量，从 0 到 100 秒

% 过程噪声协方差矩阵 Q 
q_var = 0.1; % 噪声方差
Q = q_var * [dt^3/3, 0,      dt^2/2, 0;
             0,      dt^3/3, 0,      dt^2/2;
             dt^2/2, 0,      dt,     0;
             0,      dt^2/2, 0,      dt];

% 测量噪声协方差矩阵 R，单位为 度^2
sigma_theta_deg = 2; % 设定测量标准差为 3 度
R = diag([sigma_theta_deg^2, sigma_theta_deg^2]); % 两个平台的测量噪声独立

% 目标真实初始状态 [x, y, vx, vy]
x0_true = [100; 100; -2; 5]; 

% 平台 1 的初始状态和运动参数
platform1.pos0 = [-400; 0];
platform1.vel = [0; 1];

% 平台 2 的初始状态和运动参数
platform2.pos0 = [400; 0];
platform2.vel = [0; 1];

%% 生成平台运动轨迹
platform1.traj = zeros(2, length(t));
platform2.traj = zeros(2, length(t));

for k = 1:length(t)
    platform1.traj(:, k) = platform1.pos0 + platform1.vel * t(k);
    platform2.traj(:, k) = platform2.pos0 + platform2.vel * t(k);
end

%% 生成目标真实轨迹 (CV 模型)
% omega = 0;
% [true_traj, F] = generate_cv_trajectory(x0_true, dt, t);
%% 目标拐弯运动 (CT 转弯模型)

omega = 0.01;                % 转弯角速度，单位 rad/s（约 5.73 deg/s）
[true_traj, ~] = generate_turn_trajectory(x0_true, omega, T, dt);
% 滤波器使用CV模型的F（与Q匹配）
F = [1 0 dt 0;
     0 1 0  dt;
     0 0 1  0;
     0 0 0  1];

%% 生成带噪声的方位测量值 (单位：度)
measurements = zeros(2, length(t)); 

for k = 1:length(t)
    dx1 = true_traj(1, k) - platform1.traj(1, k);
    dy1 = true_traj(2, k) - platform1.traj(2, k);
    theta1_true_deg = atan2(dy1, dx1) * 180 / pi; 
    
    dx2 = true_traj(1, k) - platform2.traj(1, k);
    dy2 = true_traj(2, k) - platform2.traj(2, k);
    theta2_true_deg = atan2(dy2, dx2) * 180 / pi; 
    
    measurements(1, k) = theta1_true_deg + sqrt(R(1,1)) * randn; 
    measurements(2, k) = theta2_true_deg + sqrt(R(2,2)) * randn; 
end

%% ========== 调用 EKF 滤波器 ==========
fprintf('\n===== 扩展卡尔曼滤波 (EKF) =====\n');

% 初始状态估计
x_est_init_ekf = x0_true + [10; 10; 0.5; -0.5];
% 初始协方差矩阵
P_est_init_ekf = diag([15^2, 15^2, 1^2, 1^2]); 

% 调用 EKF 函数
[x_est_ekf, P_est_ekf, tracked_traj_ekf] = EKF_filter(x_est_init_ekf, P_est_init_ekf, ...
                                                        measurements, platform1.traj, ...
                                                        platform2.traj, F, Q, R);

% 计算 EKF 跟踪误差
position_error_ekf = sqrt((tracked_traj_ekf(1, :) - true_traj(1, :)).^2 + ...
                          (tracked_traj_ekf(2, :) - true_traj(2, :)).^2);
mean_error_ekf = mean(position_error_ekf);
rmse_error_ekf = sqrt(mean(position_error_ekf.^2));
fprintf('EKF 平均跟踪误差: %.2f 米\n', mean_error_ekf);
fprintf('EKF 均方根误差 (RMSE): %.2f 米\n', rmse_error_ekf);

velocity_error_ekf = sqrt((x_est_ekf(3,:) - true_traj(3,:)).^2 + ...
                          (x_est_ekf(4,:) - true_traj(4,:)).^2);
mean_vel_error_ekf = mean(velocity_error_ekf);
rmse_vel_ekf = sqrt(mean(velocity_error_ekf.^2));
fprintf('EKF 平均速度误差: %.3f 米/秒\n', mean_vel_error_ekf);
fprintf('EKF 速度均方根误差 (RMSE): %.3f 米/秒\n', rmse_vel_ekf);

%% ========== 调用 UKF 滤波器 ==========
fprintf('\n===== 无迹卡尔曼滤波 (UKF) =====\n');

% 初始状态估计
x_est_init_ukf = x0_true + [10; 10; 0.5; -0.5];
% 初始协方差矩阵
P_est_init_ukf = diag([15^2, 15^2, 1^2, 1^2]); 

% 调用 UKF 函数
[x_est_ukf, P_est_ukf, tracked_traj_ukf] = UKF_filter(x_est_init_ukf, P_est_init_ukf, ...
                                                       measurements, platform1.traj, ...
                                                       platform2.traj, F, Q, R, dt,omega);

% 计算 UKF 跟踪误差
position_error_ukf = sqrt((tracked_traj_ukf(1, :) - true_traj(1, :)).^2 + ...
                          (tracked_traj_ukf(2, :) - true_traj(2, :)).^2);
mean_error_ukf = mean(position_error_ukf);
rmse_error_ukf = sqrt(mean(position_error_ukf.^2));
fprintf('UKF 平均跟踪误差: %.2f 米\n', mean_error_ukf);
fprintf('UKF 均方根误差 (RMSE): %.2f 米\n', rmse_error_ukf);

velocity_error_ukf = sqrt((x_est_ukf(3,:) - true_traj(3,:)).^2 + ...
                          (x_est_ukf(4,:) - true_traj(4,:)).^2);
mean_vel_error_ukf = mean(velocity_error_ukf);
rmse_vel_ukf = sqrt(mean(velocity_error_ukf.^2));
fprintf('UKF 平均速度误差: %.3f 米/秒\n', mean_vel_error_ukf);
fprintf('UKF 速度均方根误差 (RMSE): %.3f 米/秒\n', rmse_vel_ukf);

%% 性能对比
fprintf('\n===== EKF 与 UKF 性能对比 =====\n');
fprintf('位置 RMSE 改进: %.2f%% (EKF: %.2f m, UKF: %.2f m)\n', ...
    (rmse_error_ekf - rmse_error_ukf)/rmse_error_ekf*100, rmse_error_ekf, rmse_error_ukf);
fprintf('速度 RMSE 改进: %.2f%% (EKF: %.3f m/s, UKF: %.3f m/s)\n', ...
    (rmse_vel_ekf - rmse_vel_ukf)/rmse_vel_ekf*100, rmse_vel_ekf, rmse_vel_ukf);

%% ========== 绘制对比结果 ==========

save_path = 'D:\毕业论文\仿真结果\双基地单目标匀速'; 
if ~exist(save_path, 'dir'), mkdir(save_path); end

%% 绘制跟踪轨迹对比图（EKF vs UKF）
figure('Position', [100, 100, 1200, 900]);
plot(platform1.traj(1, :), platform1.traj(2, :), 'b-', 'LineWidth', 1.5); hold on;
plot(platform2.traj(1, :), platform2.traj(2, :), 'r-', 'LineWidth', 1.5);
plot(true_traj(1, :), true_traj(2, :), 'k-', 'LineWidth', 2.5, 'DisplayName', '真实轨迹');
plot(tracked_traj_ekf(1, :), tracked_traj_ekf(2, :), 'g--', 'LineWidth', 2, 'DisplayName', 'EKF 跟踪轨迹');
plot(tracked_traj_ukf(1, :), tracked_traj_ukf(2, :), 'm:', 'LineWidth', 2.5, 'DisplayName', 'UKF 跟踪轨迹');

grid on; axis equal;
xlabel('X 位置 (米)'); ylabel('Y 位置 (米)');
title('EKF vs UKF 双平台单目标跟踪轨迹对比');
legend('平台1轨迹', '平台2轨迹', '真实轨迹', 'EKF 跟踪轨迹', 'UKF 跟踪轨迹', 'Location', 'best');
saveas(gcf, fullfile(save_path, 'EKF_vs_UKF_跟踪轨迹.png'));

%% 绘制位置误差对比
figure;
plot(t, position_error_ekf, 'g-', 'LineWidth', 1.5, 'DisplayName', 'EKF 误差'); hold on;
plot(t, position_error_ukf, 'm-', 'LineWidth', 1.5, 'DisplayName', 'UKF 误差');
plot(t, rmse_error_ekf * ones(size(t)), 'g--', 'LineWidth', 1.5, 'DisplayName', sprintf('EKF RMSE: %.2f m', rmse_error_ekf));
plot(t, rmse_error_ukf * ones(size(t)), 'm--', 'LineWidth', 1.5, 'DisplayName', sprintf('UKF RMSE: %.2f m', rmse_error_ukf));
grid on;
xlabel('时间 (秒)'); ylabel('跟踪误差 (米)');
title('EKF vs UKF 位置跟踪误差对比');
legend('Location', 'best');
saveas(gcf, fullfile(save_path, 'EKF_vs_UKF_位置误差.png'));

%% 绘制累积位置 RMSE 对比
cumulative_rmse_ekf = zeros(size(t));
cumulative_rmse_ukf = zeros(size(t));
for k = 1:length(t)
    cumulative_rmse_ekf(k) = sqrt(mean(position_error_ekf(1:k).^2));
    cumulative_rmse_ukf(k) = sqrt(mean(position_error_ukf(1:k).^2));
end

figure;
plot(t, cumulative_rmse_ekf, 'g-', 'LineWidth', 1.5, 'DisplayName', 'EKF 累积 RMSE'); hold on;
plot(t, cumulative_rmse_ukf, 'm-', 'LineWidth', 1.5, 'DisplayName', 'UKF 累积 RMSE');
grid on;
xlabel('时间 (秒)');
ylabel('累积均方根误差 (米)');
title('EKF vs UKF 累积位置 RMSE 对比');
legend('Location', 'best');
saveas(gcf, fullfile(save_path, 'EKF_vs_UKF_累积位置RMSE.png'));

%% 绘制速度误差对比
figure;
plot(t, velocity_error_ekf, 'g-', 'LineWidth', 1.5, 'DisplayName', 'EKF 速度误差'); hold on;
plot(t, velocity_error_ukf, 'm-', 'LineWidth', 1.5, 'DisplayName', 'UKF 速度误差');
plot(t, rmse_vel_ekf * ones(size(t)), 'g--', 'LineWidth', 1.5, 'DisplayName', sprintf('EKF 速度 RMSE: %.3f m/s', rmse_vel_ekf));
plot(t, rmse_vel_ukf * ones(size(t)), 'm--', 'LineWidth', 1.5, 'DisplayName', sprintf('UKF 速度 RMSE: %.3f m/s', rmse_vel_ukf));
grid on;
xlabel('时间 (秒)');
ylabel('速度误差 (米/秒)');
title('EKF vs UKF 速度估计误差对比');
legend('Location', 'best');
saveas(gcf, fullfile(save_path, 'EKF_vs_UKF_速度误差.png'));

%% 绘制累积速度 RMSE 对比
cumulative_vel_rmse_ekf = zeros(size(t));
cumulative_vel_rmse_ukf = zeros(size(t));
for k = 1:length(t)
    cumulative_vel_rmse_ekf(k) = sqrt(mean(velocity_error_ekf(1:k).^2));
    cumulative_vel_rmse_ukf(k) = sqrt(mean(velocity_error_ukf(1:k).^2));
end

figure;
plot(t, cumulative_vel_rmse_ekf, 'g-', 'LineWidth', 1.5, 'DisplayName', 'EKF 累积速度 RMSE'); hold on;
plot(t, cumulative_vel_rmse_ukf, 'm-', 'LineWidth', 1.5, 'DisplayName', 'UKF 累积速度 RMSE');
grid on;
xlabel('时间 (秒)');
ylabel('累积速度 RMSE (米/秒)');
title('EKF vs UKF 累积速度 RMSE 对比');
legend('Location', 'best');
saveas(gcf, fullfile(save_path, 'EKF_vs_UKF_累积速度RMSE.png'));

%% 绘制位置估计对比
figure('Position', [100, 100, 1400, 600]);
subplot(2, 2, 1);
plot(t, true_traj(1, :), 'k-', 'LineWidth', 2, 'DisplayName', '真实值'); hold on;
plot(t, tracked_traj_ekf(1, :), 'g--', 'LineWidth', 1.5, 'DisplayName', 'EKF 估计');
plot(t, tracked_traj_ukf(1, :), 'm:', 'LineWidth', 1.5, 'DisplayName', 'UKF 估计');
grid on; title('X 方向位置估计对比'); ylabel('位置 (米)');
legend('Location', 'best');

subplot(2, 2, 2);
plot(t, true_traj(2, :), 'k-', 'LineWidth', 2, 'DisplayName', '真实值'); hold on;
plot(t, tracked_traj_ekf(2, :), 'g--', 'LineWidth', 1.5, 'DisplayName', 'EKF 估计');
plot(t, tracked_traj_ukf(2, :), 'm:', 'LineWidth', 1.5, 'DisplayName', 'UKF 估计');
grid on; title('Y 方向位置估计对比'); ylabel('位置 (米)');
legend('Location', 'best');

subplot(2, 2, 3);
plot(t, true_traj(3, :), 'k-', 'LineWidth', 2, 'DisplayName', '真实值'); hold on;
plot(t, x_est_ekf(3, :), 'g--', 'LineWidth', 1.5, 'DisplayName', 'EKF 估计');
plot(t, x_est_ukf(3, :), 'm:', 'LineWidth', 1.5, 'DisplayName', 'UKF 估计');
grid on; title('X 方向速度估计对比'); ylabel('速度 (米/秒)');
legend('Location', 'best');

subplot(2, 2, 4);
plot(t, true_traj(4, :), 'k-', 'LineWidth', 2, 'DisplayName', '真实值'); hold on;
plot(t, x_est_ekf(4, :), 'g--', 'LineWidth', 1.5, 'DisplayName', 'EKF 估计');
plot(t, x_est_ukf(4, :), 'm:', 'LineWidth', 1.5, 'DisplayName', 'UKF 估计');
grid on; title('Y 方向速度估计对比'); ylabel('速度 (米/秒)');
legend('Location', 'best');

sgtitle('EKF vs UKF 状态估计对比', 'FontSize', 14);
saveas(gcf, fullfile(save_path, 'EKF_vs_UKF_状态估计对比.png'));