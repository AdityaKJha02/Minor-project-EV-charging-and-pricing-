function [N_EV_DPM, N_EV_approx, E_EV_DPM, tau_h, price_h, kappa] = ...
        compute_PED_DPM( ...
        timeGrid_min, RTP_file, ...
        EV_file, sheetNames_EV, fcsID, ...
        base_price)



% =========================================================
% COMPUTE_PED_DPM
% Implements PED / DPM model (Eqs. 19–22)
%
% INPUTS:
%   timeGrid_min : vector of times (minutes, multiples of 10)
%   RTP_file     : 'RTP_grid_cost_10min.xlsx'
%   EV_file      : 'EVFCS_10min_Arrival_Power.xlsx'
%   sheetNames_EV: sheet names per FCS
%   fcsID        : station index
%
% OUTPUTS:
%   N_EV_DPM     : EV arrivals under DPM (vector)
%   tau_h        : price convenience vector
%   kappa        : normalization factor
% =========================================================

%% ---------------------------
% STEP 1: Read RTP (ρ_SELL)
% ---------------------------
M = readmatrix(RTP_file);
timeRTP = M(:,1);
rhoSELL = M(:,3);

inv_price = 1 ./ rhoSELL;
tau_all   = inv_price ./ max(inv_price);   % Eq. (22)

%% ---------------------------
% STEP 2: Get τʰ at requested times
% ---------------------------
nT = numel(timeGrid_min);
tau_h = zeros(nT,1);

for k = 1:nT
    idx = find(timeRTP == timeGrid_min(k), 1);
    if isempty(idx)
        error('Time %d not found in RTP file', timeGrid_min(k));
    end
    tau_h(k) = tau_all(idx);
end

%% ---------------------------
% STEP 3: Get N_EV^h,FPM from EV file
% ---------------------------
N_EV_FPM = zeros(nT,1);

for k = 1:nT
    N_EV_FPM(k) = getEVcountAtTime( ...
        EV_file, sheetNames_EV, fcsID, timeGrid_min(k));
end
%% ---------------------------
% STEP 3: SIGMOID PRICE FROM τ
% ---------------------------
%% ---------------------------
% STEP 3: DPM PRICE FROM τ (FIXED)
% ---------------------------

% Desired absolute price band
p_min = 7;      % Rs/kWh
p_max = 13;     % Rs/kWh
beta  = 8;      % sensitivity
tau0  = 0.5;    % midpoint

price_norm = p_min + ...
    (p_max - p_min) ./ ...
    (1 + exp(beta * (tau_h - tau0)));

price_h = base_price .* price_norm;



%% ---------------------------
% STEP 4: Compute κ (Eq. 21)
% ---------------------------
kappa = sum(N_EV_FPM) / sum(tau_h .* N_EV_FPM);

%% ---------------------------
% STEP 5: Compute N_EV^h,DPM (Eq. 19)
% ---------------------------
N_EV_DPM = kappa * tau_h .* N_EV_FPM;


%% ---------------------------
% STEP 6: Compute E_D,EV^{h,DPM}
% ---------------------------
dt = 1/6;   % 10 min in hours
nT = numel(timeGrid_min);

% Read EV power for this FCS
EV_temp = readmatrix(EV_file,'Sheet',sheetNames_EV{fcsID});
P_EV = EV_temp(:,7);      % kW
timeCol = EV_temp(:,3);   % minutes

% --- FPM hourly energy ---
E_EV_FPM_h = zeros(nT,1);

for k = 1:nT
    idx = find(timeCol == timeGrid_min(k),1);
    if ~isempty(idx)
        E_EV_FPM_h(k) = P_EV(idx) * dt;
    end
end

% --- Average energy per EV ---
EVtimeline_file = 'EVFCS_EV_Timeline.xlsx';
sheet_tl = sheetNames_EV{fcsID};
e_bar = getAverageSessionEnergy(EVtimeline_file, sheet_tl);

% fallback if no timeline data:
if e_bar == 0
    N_EV_FPM_total = sum(N_EV_FPM);
    if N_EV_FPM_total > 0
        e_bar = sum(E_EV_FPM_h) / N_EV_FPM_total;
    else
        e_bar = 0;
    end
end


% --- DPM energy ---
E_EV_DPM = N_EV_DPM * e_bar;

% ---------------------------------
% INTEGER EV ARRIVALS (NO RNG, MASS CONSERVING)
% ---------------------------------

N_EV_floor = floor(N_EV_DPM);
residual   = N_EV_DPM - N_EV_floor;

% Number of EVs to reassign
missingEVs = round(sum(N_EV_DPM) - sum(N_EV_floor));

% Assign extra EVs to highest residual slots
[~, idx] = sort(residual, 'descend');

N_EV_approx = N_EV_floor;
N_EV_approx(idx(1:missingEVs)) = N_EV_approx(idx(1:missingEVs)) + 1;


end

