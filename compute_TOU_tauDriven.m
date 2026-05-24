function [N_EV_TOU, N_EV_int, E_EV_TOU, tau_h, kappa, bounds, price_h] = ...
compute_TOU_tauDriven( ...
RTP_file, ...
EV_file, sheetNames_EV, ...
fcsID, prices, base_price)

% prices = [p_off p_mid p_peak]

fprintf('\n========================================\n');
fprintf('FCS %d | TOU Pricing\n', fcsID);
fprintf('========================================\n');

%% ---------------------------
% STEP 1: RTP → Hourly tau
%% ---------------------------
M = readmatrix(RTP_file);

time_min = M(:,1);
rhoSELL  = M(:,3);

validIdx = ~isnan(time_min) & ~isnan(rhoSELL);
time_min = time_min(validIdx);
rhoSELL  = rhoSELL(validIdx);

idx24h = (time_min >= 0) & (time_min <= 1430);
time_min = time_min(idx24h);
rhoSELL  = rhoSELL(idx24h);

hour_index = floor(time_min / 60);

rho_hourly = zeros(24,1);

for h = 0:23
    idx = (hour_index == h);
    rho_hourly(h+1) = mean(rhoSELL(idx));
end

inv_p = 1 ./ rho_hourly;
tau_h = inv_p ./ max(inv_p);

%% ---------------------------
% STEP 2: Quantile bounds
%% ---------------------------
q1 = quantile(tau_h,0.33);
q2 = quantile(tau_h,0.66);

bounds = [q1 q2];

price_h = zeros(24,1);

for h = 1:24

    if tau_h(h) >= q2
        price_h(h) = prices(1) * base_price;   % OFF PEAK

    elseif tau_h(h) >= q1
        price_h(h) = prices(2) * base_price;   % MID

    else
        price_h(h) = prices(3) * base_price;   % PEAK
    end

end

%% ---------------------------
% STEP 3: New tau from price
%% ---------------------------
inv_price = 1 ./ price_h;
tau_new = inv_price ./ max(inv_price);

%% ---------------------------
% STEP 4: Hourly EV arrivals
%% ---------------------------
N_EV_10min = zeros(144,1);

for k = 1:144
    N_EV_10min(k) = getEVcountAtTime( ...
        EV_file, sheetNames_EV, fcsID, (k-1)*10);
end

N_EV_hour = reshape(N_EV_10min,6,24);
N_FPM = sum(N_EV_hour,1)';

%% ---------------------------
% STEP 5: Normalization κ
%% ---------------------------
kappa = sum(N_FPM) / sum(tau_new .* N_FPM);

%% ---------------------------
% STEP 6: TOU arrivals
%% ---------------------------
N_EV_TOU = kappa * tau_new .* N_FPM;

%% ---------------------------
% STEP 7: Energy calculation
%% ---------------------------
EV_temp = readmatrix(EV_file,'Sheet',sheetNames_EV{fcsID});

P_EV_10min = EV_temp(2:145,7);

P_reshaped = reshape(P_EV_10min,6,24);

P_hourly = mean(P_reshaped,1)';

E_FPM_hour = P_hourly;

E_avg = sum(E_FPM_hour) / sum(N_FPM);

E_EV_TOU = N_EV_TOU * E_avg;

%% ---------------------------
% STEP 8: Integerization
%% ---------------------------
floorEV = floor(N_EV_TOU);

residual = N_EV_TOU - floorEV;

missing = round(sum(N_EV_TOU) - sum(floorEV));

[~,idx] = sort(residual,'descend');

N_EV_int = floorEV;
N_EV_int(idx(1:missing)) = N_EV_int(idx(1:missing)) + 1;

end
