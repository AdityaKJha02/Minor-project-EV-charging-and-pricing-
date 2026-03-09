function [N_EV_CPP, N_EV_int, E_EV_CPP, tau_new, kappa, cpp_idx, price_h] = ...
    compute_CPP_tauDriven( ...
        RTP_file, ...
        EV_file, sheetNames_EV, fcsID, ...
        multiplier, base_price)

% CPP applied on top 5 hours (lowest tau)

fprintf('\n========================================\n');
fprintf('FCS %d | CPP Pricing (Hourly)\n', fcsID);
fprintf('========================================\n');

%% =========================================================
% STEP 1: RTP → Hourly tau
%% =========================================================

M = readmatrix(RTP_file);

time_min = M(:,1);
rhoSELL  = M(:,3);

valid = ~isnan(time_min) & ~isnan(rhoSELL);

time_min = time_min(valid);
rhoSELL  = rhoSELL(valid);

hour_index = floor(time_min/60);

rho_hourly = zeros(24,1);

for h = 0:23
    idx = hour_index==h;
    rho_hourly(h+1) = mean(rhoSELL(idx));
end

inv_p = 1./rho_hourly;
tau_base = inv_p./max(inv_p);

%% =========================================================
% STEP 2: Select top 5 CPP hours
%% =========================================================

[~,idxSort] = sort(tau_base,'ascend');

cpp_idx = idxSort(1:5);     % top 5 lowest tau hours

price_h = base_price * ones(24,1);

price_h(cpp_idx) = multiplier * base_price;

%% =========================================================
% STEP 3: New τ
%% =========================================================

inv_price = 1./price_h;
tau_new = inv_price ./ max(inv_price);

%% =========================================================
% STEP 4: Hourly EV arrivals
%% =========================================================

N_EV_10min = zeros(144,1);

for k = 1:144

    N_EV_10min(k) = getEVcountAtTime( ...
        EV_file, sheetNames_EV, fcsID, (k-1)*10);

end

N_EV_hour = reshape(N_EV_10min,6,24);

N_FPM = sum(N_EV_hour,1)';

%% =========================================================
% STEP 5: κ normalization
%% =========================================================

kappa = sum(N_FPM) / sum(tau_new .* N_FPM);

%% =========================================================
% STEP 6: CPP arrivals
%% =========================================================

N_EV_CPP = kappa * tau_new .* N_FPM;

%% =========================================================
% STEP 7: Convert EV power → hourly
%% =========================================================

EV_temp = readmatrix(EV_file,'Sheet',sheetNames_EV{fcsID});

P_EV_10min = EV_temp(2:145,7);

P_reshaped = reshape(P_EV_10min,6,24);

P_hourly = mean(P_reshaped,1)';

E_FPM_hour = P_hourly;      % 1 hr slot

%% Average energy per EV

e_bar = sum(E_FPM_hour) / max(sum(N_FPM),1);

E_EV_CPP = N_EV_CPP * e_bar;

%% =========================================================
% STEP 8: Integerization
%% =========================================================

floorEV = floor(N_EV_CPP);

residual = N_EV_CPP - floorEV;

missing = round(sum(N_EV_CPP) - sum(floorEV));

[~,idx] = sort(residual,'descend');

N_EV_int = floorEV;

N_EV_int(idx(1:missing)) = N_EV_int(idx(1:missing)) + 1;

end
