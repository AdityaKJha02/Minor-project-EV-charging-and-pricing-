function [minV, sysLoss] = runLoadFlow24h(file, mpc)

define_constants;

% Read Excel
P = readmatrix(file, 'Sheet', 'P_Load');
Q = readmatrix(file, 'Sheet', 'Q_Load');

nb = size(P,1);
nh = size(P,2);

V_all = zeros(nb, nh);
sysLoss = zeros(1, nh);

for h = 1:nh

    mpc.bus(:, PD) = P(:,h);
    mpc.bus(:, QD) = Q(:,h);

    results = runpf(mpc, mpoption('verbose',0,'out.all',0));

    % Store voltages
    V_all(:,h) = results.bus(:, VM);

    % System loss
    loss = sum(results.branch(:, PF) + results.branch(:, PT));
    sysLoss(h) = loss;
end

% Minimum voltage per bus over 24h
minV = min(V_all, [], 2);

end