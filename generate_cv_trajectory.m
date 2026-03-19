function [true_traj, F] = generate_cv_trajectory(x0_true, dt, t)
% 生成目标真实轨迹 (匀速直线运动CV模型)
% 输入:
%   x0_true  : [4x1] 初始状态 [x; y; vx; vy]
%   dt       : 采样周期
%   t        : 时间向量 1xN
% 输出:
%   true_traj: [4xN] 每时刻 [x; y; vx; vy]
%   F        : [4x4] 状态转移矩阵

N = length(t);

F = [1 0 dt 0;
     0 1 0 dt;
     0 0 1 0;
     0 0 0 1];

true_traj = zeros(4, N);
true_traj(:, 1) = x0_true;

for k = 2:N
    true_traj(:, k) = F * true_traj(:, k-1);
end
end