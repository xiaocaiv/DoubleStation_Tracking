function [x_est, P_est, tracked_traj] = UKF_filter(x_est_init, P_est_init, measurements, ...
                                                     platform1_traj, platform2_traj, ...
                                                     F_cv, Q, R, dt, omega)  % 建议传入 omega
% ... (前面的注释保持不变) ...
% 输入参数新增：
%   omega - 目标转弯角速度 (rad/s)
%   F_cv  - 原有的线性状态转移矩阵（如果 omega 为 0 时使用）

% 获取测量的总时间步数
N = size(measurements, 2);
n_x = 4;            % 状态维度 [x, y, vx, vy]
n_z = 2;            % 测量维度 [theta1, theta2]

% ========== UKF 参数配置 (建议优化参数以提高稳定性) ==========
alpha = 1e-3;       % 减小 alpha 使 Sigma 点更靠近均值，减少在强非线性区的发散
beta = 2;
kappa = 0;
lambda = alpha^2 * (n_x + kappa) - n_x;

% 计算 Sigma 点权重
W_m0 = lambda / (n_x + lambda);
W_c0 = lambda / (n_x + lambda) + (1 - alpha^2 + beta);
W_m = ones(2*n_x, 1) / (2*(n_x + lambda));
W_c = ones(2*n_x, 1) / (2*(n_x + lambda));

% --- 修改这里：构造完整的权重向量 ---
Wm_all = [W_m0; W_m];    % 用于均值计算 [2n+1, 1]
Wc_all = [W_c0; W_c];    % 用于协方差计算 [2n+1, 1]

% 预分配
x_est = zeros(4, N);
P_est = zeros(4, 4, N);
tracked_traj = zeros(2, N);
X_pred_sig = zeros(n_x, 2*n_x+1);

% 初值设置
x_est(:, 1) = x_est_init;
P_est(:, :, 1) = P_est_init;
tracked_traj(:, 1) = x_est_init(1:2, 1);

%% UKF 主循环
for k = 2:N
    % ========== 预测步骤 (Prediction Step) ==========
    % 1. 生成上一时刻的 Sigma 点
    sqrt_P = chol((n_x + lambda) * P_est(:, :, k-1), 'lower');
    X_sig = [x_est(:, k-1), x_est(:, k-1) + sqrt_P, x_est(:, k-1) - sqrt_P];
    
    % 2. 通过【CT 模型】传播 Sigma 点
    for i = 1:(2*n_x+1)
        X_pred_sig(:, i) = f_state_ct(X_sig(:, i), omega, dt); % 修改此处
    end
    
    % 3. 重构预测均值
    x_pred = X_pred_sig * Wm_all;
    
    % 4. 重构预测协方差

    P_pred = Q; 
    for i = 1:(2*n_x + 1)
        x_diff = X_pred_sig(:, i) - x_pred;
        % --- 修改这里：直接使用 i 索引 ---
        weight = Wc_all(i); 
        P_pred = P_pred + weight * (x_diff * x_diff');
    end
    
    % ========== 更新步骤 (Update Step) ==========
    % 使用最新的预测协方差重新采样（重采样能增加对非线性的捕捉）
    sqrt_P_pred = chol((n_x + lambda) * P_pred, 'lower');
    X_sig_pred = [x_pred, x_pred + sqrt_P_pred, x_pred - sqrt_P_pred];
    
    p1 = platform1_traj(:, k);
    p2 = platform2_traj(:, k);
    Z_pred_sig = zeros(n_z, 2*n_x+1);
    for i = 1:(2*n_x+1)
        Z_pred_sig(:, i) = h_measure(X_sig_pred(:, i), p1, p2);
    end
    
    % 5. 预测测量值均值
    z_pred = Z_pred_sig * Wm_all;
    % 角度规整（核心：防止平均值跳变）
    z_pred(1) = mod(z_pred(1) + 180, 360) - 180;
    z_pred(2) = mod(z_pred(2) + 180, 360) - 180;
    
    % 6. 计算协方差矩阵 P_zz, P_xz
    P_zz = R;
    P_xz = zeros(n_x, n_z);
    for i = 1:(2*n_x+1)
        % 测量残差规整
        z_diff = Z_pred_sig(:, i) - z_pred;
        z_diff(1) = mod(z_diff(1) + 180, 360) - 180;
        z_diff(2) = mod(z_diff(2) + 180, 360) - 180;
        
        x_diff = X_sig_pred(:, i) - x_pred;
        % --- 修改这里：直接使用 i 索引 ---
        weight = Wc_all(i); 
        
        P_zz = P_zz + weight * (z_diff * z_diff');
        P_xz = P_xz + weight * (x_diff * z_diff');
    end
    
    % 7. 正则化协方差（处理数值问题）
    P_zz = (P_zz + P_zz') / 2;
    
    % 8. 状态更新
    K = P_xz / P_zz;
    z_obs = measurements(:, k);
    y = z_obs - z_pred;
    y(1) = mod(y(1) + 180, 360) - 180;
    y(2) = mod(y(2) + 180, 360) - 180;
    
    x_est(:, k) = x_pred + K * y;
    P_est(:, :, k) = P_pred - K * P_zz * K';
    tracked_traj(:, k) = x_est(1:2, k);
end
end

%% ========== 修改后的辅助函数 ==========

% 非线性状态转移函数：匀速转弯 (CT) 模型
function x_next = f_state_ct(x, omega, dt)
    % x = [x; y; vx; vy]
    if abs(omega) < 1e-6
        % 如果角速度极小，退化为 CV 模型，防止除以零
        F_ct = [1, 0, dt, 0;
                0, 1, 0,  dt;
                0, 0, 1,  0;
                0, 0, 0,  1];
    else
        % CT 模型转移矩阵
        s = sin(omega * dt);
        c = cos(omega * dt);
        F_ct = [1, 0,  s/omega,       -(1-c)/omega;
                0, 1, (1-c)/omega,    s/omega;
                0, 0,  c,             -s;
                0, 0,  s,              c];
    end
    x_next = F_ct * x;
end

% 测量函数保持不变（方位角测量）
function z = h_measure(x, p1, p2)
    dx1 = x(1) - p1(1); dy1 = x(2) - p1(2);
    dx2 = x(1) - p2(1); dy2 = x(2) - p2(2);
    z = [atan2(dy1, dx1); atan2(dy2, dx2)] * (180/pi);
end