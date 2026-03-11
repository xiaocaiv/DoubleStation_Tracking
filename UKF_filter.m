function [x_est, P_est, tracked_traj] = UKF_filter(x_est_init, P_est_init, measurements, ...
                                                     platform1_traj, platform2_traj, ...
                                                     F, Q, R, dt)
% 无迹卡尔曼滤波 (Unscented Kalman Filter, UKF) 函数
%
% 功能：基于双平台的方位角测量，使用 UKF 对目标位置进行递推估计
%       UKF 通过 sigma 点采样避免显式计算雅可比矩阵，适用于强非线性系统
%
% 输入参数：
%   x_est_init        - 初始状态估计值，大小为 [4, 1]，格式为 [x; y; vx; vy]
%   P_est_init        - 初始估计协方差矩阵，大小为 [4, 4]
%   measurements      - 两个平台的方位角测量值（度），大小为 [2, N]
%   platform1_traj    - 平台 1 的运动轨迹，大小为 [2, N]
%   platform2_traj    - 平台 2 的运动轨迹，大小为 [2, N]
%   F                 - 状态转移矩阵（线性部分），大小为 [4, 4]
%   Q                 - 过程噪声协方差矩阵，大小为 [4, 4]
%   R                 - 测量噪声协方差矩阵，大小为 [2, 2]
%   dt                - 采样周期（秒），用于数值计算
%
% 输出参数：
%   x_est             - 所有时刻的状态估计值，大小为 [4, N]
%                       其中 x_est(1,:) 为 X 位置，x_est(2,:) 为 Y 位置
%                           x_est(3,:) 为 X 方向速度，x_est(4,:) 为 Y 方向速度
%   P_est             - 所有时刻的估计协方差矩阵，大小为 [4, 4, N]
%   tracked_traj      - 跟踪得到的目标位置轨迹，大小为 [2, N]
%

% 获取测量的总时间步数
N = size(measurements, 2);

% 状态维度和参数设置
n_x = 4;            % 状态维度 [x, y, vx, vy]
n_z = 2;            % 测量维度 [theta1, theta2]

% ========== UKF 参数配置 ==========
% alpha: Sigma 点扩展参数（通常设为 1e-3 到 1），用于控制 Sigma 点相对于均值的距离
% beta: 用于融合高斯分布知识的参数（对于高斯分布，beta = 2 是最优的）
% kappa: 次级缩放参数，通常设为 0
alpha = 1e-3;
beta = 2;
kappa = 0;

% 计算缩放参数 lambda
lambda = alpha^2 * (n_x + kappa) - n_x;

% 计算 Sigma 点权重
W_m0 = lambda / (n_x + lambda);                          % 均值权重（m 表示 mean）
W_c0 = lambda / (n_x + lambda) + (1 - alpha^2 + beta);  % 协方差权重（c 表示 covariance）
W_m = ones(2*n_x, 1) / (2*(n_x + lambda));             % 其他 Sigma 点的均值权重
W_c = ones(2*n_x, 1) / (2*(n_x + lambda));             % 其他 Sigma 点的协方差权重

% 预分配存储空间
x_est = zeros(4, N);              % 状态估计值
P_est = zeros(4, 4, N);           % 估计协方差矩阵
tracked_traj = zeros(2, N);       % 跟踪位置轨迹

% Sigma 点存储空间
X_pred_sig = zeros(n_x, 2*n_x+1); % 预测阶段的 Sigma 点
Z_pred_sig = zeros(n_z, 2*n_x+1); % 测量空间的 Sigma 点

% 设置初值
x_est(:, 1) = x_est_init;
P_est(:, :, 1) = P_est_init;
tracked_traj(:, 1) = x_est_init(1:2, 1);

%% UKF 主循环
for k = 2:N
    % ========== 预测步骤 (Prediction Step) ==========
    
    % 生成 Sigma 点（基于上一时刻的状态估计和协方差）
    % 使用 Cholesky 分解来计算协方差矩阵的平方根
    sqrt_P = chol((n_x + lambda) * P_est(:, :, k-1), 'lower');
    
    % 中心点（第一个 Sigma 点）
    X_sig = zeros(n_x, 2*n_x+1);
    X_sig(:, 1) = x_est(:, k-1);
    
    % 其他 Sigma 点
    for i = 1:n_x
        X_sig(:, i+1) = x_est(:, k-1) + sqrt_P(:, i);
        X_sig(:, n_x+i+1) = x_est(:, k-1) - sqrt_P(:, i);
    end
    
    % 通过非线性状态转移函数传播 Sigma 点
    for i = 1:(2*n_x+1)
        X_pred_sig(:, i) = f_state(X_sig(:, i), F, dt);
    end
    
    % 重构预测状态（利用加权求和）
    Wm_all = [W_m0; W_m];    % [2n+1, 1]
    x_pred = X_pred_sig * Wm_all;  % [n_x, 1]
    
    % 重构预测协方差
    P_pred = zeros(n_x, n_x);
    for i = 1:(2*n_x+1)
        if i == 1
            P_pred = P_pred + W_c0 * (X_pred_sig(:, i) - x_pred) * (X_pred_sig(:, i) - x_pred)';
        else
            P_pred = P_pred + W_c(i-1) * (X_pred_sig(:, i) - x_pred) * (X_pred_sig(:, i) - x_pred)';
        end
    end
    % 加入过程噪声
    P_pred = P_pred + Q;
    
    % ========== 更新步骤 (Update Step) ==========
    
    % 基于预测状态生成新的 Sigma 点
    sqrt_P_pred = chol((n_x + lambda) * P_pred, 'lower');
    X_sig_pred = zeros(n_x, 2*n_x+1);
    X_sig_pred(:, 1) = x_pred;
    for i = 1:n_x
        X_sig_pred(:, i+1) = x_pred + sqrt_P_pred(:, i);
        X_sig_pred(:, n_x+i+1) = x_pred - sqrt_P_pred(:, i);
    end

    
    % 获取当前时刻的测量值和平台位置
    z = measurements(:, k);
    p1 = platform1_traj(:, k);
    p2 = platform2_traj(:, k);
    
    % 通过非线性测量函数传播预测 Sigma 点
    for i = 1:(2*n_x+1)
        Z_pred_sig(:, i) = h_measure(X_sig_pred(:, i), p1, p2);
    end
    
    % 重构预测测量值
    z_pred = Z_pred_sig * Wm_all;
    
    % 处理角度差的规整（避免角度跳变）
    z_pred(1) = mod(z_pred(1) + 180, 360) - 180;
    z_pred(2) = mod(z_pred(2) + 180, 360) - 180;
    
    % 计算测量空间的协方差矩阵 P_zz
    P_zz = zeros(n_z, n_z);
    % 计算交叉协方差矩阵 P_xz
    P_xz = zeros(n_x, n_z);
    
    for i = 1:(2*n_x+1)
        % 计算测量残差（新息）
        z_diff = Z_pred_sig(:, i) - z_pred;
        % 处理角度差的规整
        z_diff(1) = mod(z_diff(1) + 180, 360) - 180;
        z_diff(2) = mod(z_diff(2) + 180, 360) - 180;
        
        % 计算状态残差
        x_diff = X_sig_pred(:, i) - x_pred;
        
        if i == 1
            P_zz = P_zz + W_c0 * (z_diff * z_diff');
            P_xz = P_xz + W_c0 * (x_diff * z_diff');
        else
            P_zz = P_zz + W_c(i-1) * (z_diff * z_diff');
            P_xz = P_xz + W_c(i-1) * (x_diff * z_diff');
        end
    end
    
    % 加入测量噪声
    P_zz = P_zz + R;
    
    % 计算卡尔曼增益
    K = P_xz / P_zz;
    
    % 计算测量残差（观测值与预测值的差）
    y = z - z_pred;
    % 处理角度差的规整
    y(1) = mod(y(1) + 180, 360) - 180;
    y(2) = mod(y(2) + 180, 360) - 180;
    
    % 状态更新
    x_est(:, k) = x_pred + K * y;
    
    % 协方差更新
    P_est(:, :, k) = P_pred - K * P_zz * K';
    
    % 记录跟踪得到的位置
    tracked_traj(:, k) = x_est(1:2, k);
end

end

%% ========== 辅助函数 ==========

% 非线性状态转移函数（CV 匀速运动模型）
function x_next = f_state(x, F, dt)
    % 使用线性 CV 模型进行状态转移
    % x = [x; y; vx; vy]
    x_next = F * x;
end

% 非线性测量函数（方位角测量）
function z = h_measure(x, p1, p2)
    % 计算目标相对于两个平台的方位角（度）
    % x = [x; y; vx; vy]
    % z = [theta1; theta2]（单位：度）
    
    % 相对于平台 1 的方位角
    dx1 = x(1) - p1(1);
    dy1 = x(2) - p1(2);
    theta1_rad = atan2(dy1, dx1);
    theta1_deg = theta1_rad * 180 / pi;
    
    % 相对于平台 2 的方位角
    dx2 = x(1) - p2(1);
    dy2 = x(2) - p2(2);
    theta2_rad = atan2(dy2, dx2);
    theta2_deg = theta2_rad * 180 / pi;
    
    z = [theta1_deg; theta2_deg];
end