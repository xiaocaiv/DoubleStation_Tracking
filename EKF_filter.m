function [x_est, P_est, tracked_traj] = EKF_filter(x_est_init, P_est_init, measurements, ...
                                                     platform1_traj, platform2_traj, ...
                                                     F, Q, R)
% 扩展卡尔曼滤波 (Extended Kalman Filter, EKF) 函数
%
% 功能：基于双平台的方位角测量，对目标位置进行递推估计
%
% 输入参数：
%   x_est_init        - 初始状态估计值，大小为 [4, 1]，格式为 [x; y; vx; vy]
%   P_est_init        - 初始估计协方差矩阵，大小为 [4, 4]
%   measurements      - 两个平台的方位角测量值（度），大小为 [2, N]
%   platform1_traj    - 平台 1 的运动轨迹，大小为 [2, N]
%   platform2_traj    - 平台 2 的运动轨迹，大小为 [2, N]
%   F                 - 状态转移矩阵，大小为 [4, 4]
%   Q                 - 过程噪声协方差矩阵，大小为 [4, 4]
%   R                 - 测量噪声协方差矩阵，大小为 [2, 2]
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

% 预分配存储空间
x_est = zeros(4, N);              % 状态估计值
P_est = zeros(4, 4, N);           % 估计协方差矩阵
tracked_traj = zeros(2, N);       % 跟踪位置轨迹

% 设置初值
x_est(:, 1) = x_est_init;         % 初始状态估计
P_est(:, :, 1) = P_est_init;      % 初始协方差矩阵
tracked_traj(:, 1) = x_est_init(1:2, 1); % 初始位置

%% EKF 主循环
for k = 2:N
    % ========== 预测步骤 (Prediction Step) ==========
    % 根据上一时刻的估计值和状态转移矩阵预测当前时刻的状态
    x_pred = F * x_est(:, k-1);
    
    % 预测协方差矩阵（考虑过程噪声）
    P_pred = F * P_est(:, :, k-1) * F' + Q;
    
    % ========== 更新步骤 (Update Step) ==========
    % 获取当前时刻的测量值（带噪声的方位角，单位：度）
    z = measurements(:, k);
    
    % 获取当前时刻两个平台的位置
    p1 = platform1_traj(:, k);
    p2 = platform2_traj(:, k);
    
    % 计算预测状态对应的测量值（预测方位角）
    % 平台 1
    dx1_pred = x_pred(1) - p1(1);
    dy1_pred = x_pred(2) - p1(2);
    theta1_pred_deg = atan2(dy1_pred, dx1_pred) * 180 / pi;
    
    % 平台 2
    dx2_pred = x_pred(1) - p2(1);
    dy2_pred = x_pred(2) - p2(2);
    theta2_pred_deg = atan2(dy2_pred, dx2_pred) * 180 / pi;
    
    z_pred = [theta1_pred_deg; theta2_pred_deg];
    
    % 计算雅可比矩阵 H（测量函数对状态的偏导数）
    % H 矩阵用于将状态空间映射到测量空间
    r1_sq = dx1_pred^2 + dy1_pred^2;  % 平台 1 到目标的距离平方
    r2_sq = dx2_pred^2 + dy2_pred^2;  % 平台 2 到目标的距离平方
    
    rad2deg = 180 / pi;  % 弧度转度的系数
    H = zeros(2, 4);
    
    % 平台 1 的雅可比矩阵元素
    H(1, 1) = -dy1_pred / r1_sq * rad2deg;
    H(1, 2) = dx1_pred / r1_sq * rad2deg;
    
    % 平台 2 的雅可比矩阵元素
    H(2, 1) = -dy2_pred / r2_sq * rad2deg;
    H(2, 2) = dx2_pred / r2_sq * rad2deg;
    
    % 计算新息协方差矩阵 S（测量残差的协方差）
    S = H * P_pred * H' + R;
    
    % 计算卡尔曼增益 K（用于平衡预测值和测量值的权重）
    K = P_pred * H' / S;
    
    % 计算测量残差（新息）：实际测量值与预测测量值的差异
    y = z - z_pred;
    
    % 将角度差规整到 [-180, 180] 度范围内，避免角度跳变问题
    % 例如：359° 和 -1° 的差应该是 2°，而不是 360°
    y(1) = mod(y(1) + 180, 360) - 180;
    y(2) = mod(y(2) + 180, 360) - 180;
    
    % 状态更新：结合预测值和测量信息得到最优估计
    x_est(:, k) = x_pred + K * y;
    
    % 协方差更新：更新对估计不确定度的描述
    P_est(:, :, k) = (eye(4) - K * H) * P_pred;
    
    % 记录跟踪得到的位置（仅保存 X 和 Y 坐标）
    tracked_traj(:, k) = x_est(1:2, k);
end

end