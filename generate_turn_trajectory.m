function [true_traj,F] = generate_turn_trajectory(x0, omega, T, dt)
% 生成匀速转弯（CT）运动的目标真实轨迹
% 输入：
%   x0    - 初始状态 [x; y; vx; vy] (4x1)
%   omega - 转弯角速度 (rad/s)，正值表示逆时针转弯
%   T     - 总仿真时间 (秒)
%   dt    - 采样周期 (秒)
% 输出：
%   true_traj - 状态序列，大小为 [4, N]，N = floor(T/dt)+1
%               其中 true_traj(1,:) 为 x 位置，true_traj(2,:) 为 y 位置
%                    true_traj(3,:) 为 vx，true_traj(4,:) 为 vy

t = 0:dt:T;
N = length(t);
true_traj = zeros(4, N);
true_traj(:,1) = x0;

% 匀速转弯离散化状态转移矩阵（CT模型）
% 参考文献：Bar-Shalom, "Estimation with Applications to Tracking and Navigation"
for k = 2:N
    if abs(omega) < 1e-6   % 接近直线运动，用CV模型避免数值问题
        F =    [1, 0, dt, 0;
                0, 1, 0, dt;
                0, 0, 1,  0;
                0, 0, 0,  1];
    else
        w = omega;
        s = sin(w*dt);
        c = cos(w*dt);
        F =    [1, 0,  s/w, -(1-c)/w;
                0, 1, (1-c)/w,  s/w;
                0, 0,  c,      -s;
                0, 0,  s,       c];
    end
    true_traj(:,k) = F * true_traj(:,k-1);
end
end