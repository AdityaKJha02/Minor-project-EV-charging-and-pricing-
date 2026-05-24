 function [N_EV_DPM, N_EV_int, E_EV_DPM, E_EV_FPM_kWh, tau_h, price_h, kappa] = ...
    compute_PED_DPM_sigmoid( ...
        timeGrid_min, RTP_file, ...
        EV_file, sheetNames_EV, ...
        summary_file, ...
        fcsID, base_price)

fprintf('\n========================================\n');
fprintf('FCS %d | Base Price = %.2f Rs/kWh\n', fcsID, base_price);
fprintf('========================================\n');
% =========================================================
% PED / DPM with sigmoid price scaling around base_price
% Price range ≈ [0.7, 1.3] * base_price
% =========================================================

%% ---------------------------
% STEP 1: RTP → τ
% ---------------------------
%% ---------------------------
% STEP 1: RTP → Hourly τ (0–1430 only)
%% ---------------------------

M = readmatrix(RTP_file);

time_min = M(:,1);      % Column A
rhoSELL  = M(:,3);      % Column C

% Remove NaNs
validIdx = ~isnan(time_min) & ~isnan(rhoSELL);
time_min = time_min(validIdx);
rhoSELL  = rhoSELL(validIdx);

% Keep only full 24h data (0 to 1430)
idx24h = (time_min >= 0) & (time_min <= 1430);
time_min = time_min(idx24h);
rhoSELL  = rhoSELL(idx24h);

% Now convert to hourly
hour_index = floor(time_min / 60);

rho_hourly = zeros(24,1);

for h = 0:23
    idx = (hour_index == h);
    rho_hourly(h+1) = mean(rhoSELL(idx));
end

% Compute tau
inv_price = 1 ./ rho_hourly;
tau_h = inv_price ./ max(inv_price);

nT = 24;

fprintf('\nSTEP 1: Hourly RTP → tau (0–1430 min used)\n');
disp(tau_h');


%% ---------------------------
% STEP 2: Baseline EV arrivals (Hourly FPM)
% ---------------------------

% Read full 10-min EV arrival data
EV_temp = readmatrix(EV_file,'Sheet',sheetNames_EV{fcsID});
timeCol = EV_temp(:,3);

N_EV_10min = zeros(144,1);

for k = 1:144
    N_EV_10min(k) = getEVcountAtTime( ...
        EV_file, sheetNames_EV, fcsID, (k-1)*10);
end

% Convert 10-min → hourly (sum 6 slots)
N_EV_reshaped = reshape(N_EV_10min, 6, 24);
N_EV_FPM = sum(N_EV_reshaped,1)';   % 24x1

fprintf('\nSTEP 2: Hourly EV Arrivals (FPM)\n');
fprintf('Total FPM EVs = %.2f\n', sum(N_EV_FPM));
disp(N_EV_FPM');

%% ---------------------------
% STEP 2B: Hourly FPM Energy from Arrival Power File
%% ---------------------------
%% STEP 2B: Hourly FPM Power (Correct Logic)

% Determine available rows
nRows = size(EV_temp,1);

% Always extract first 144 slots safely
if nRows >= 144
    P_EV_10min = EV_temp(2:145,7);   % First 144 time slots
else
    error('EV data must contain at least 144 rows');
end

% Reshape into 6 slots × 24 hours
P_reshaped = reshape(P_EV_10min, 6, 24);

% Hourly average power (kW)
P_EV_FPM_hourly = mean(P_reshaped, 1)';   % 24×1

% Since duration = 1 hour → Energy = Power
E_EV_FPM_kWh = P_EV_FPM_hourly;          % 24×1

fprintf('\nSTEP 2B: Hourly FPM Power\n');
fprintf('Total FPM Energy = %.2f kWh\n', sum(E_EV_FPM_kWh));

%% ---------------------------
% STEP 3: Sigmoid price multiplier
% ---------------------------
m_min = 0.7;      % → ~7 Rs if base=10
m_max = 1.3;      % → ~13 Rs if base=10
beta  = 10;        % steepness
tau0  = 0.6;      % midpoint

multiplier = m_min + ...
    (m_max - m_min) ./ ...
    (1 + exp(beta * (tau_h - tau0)));

price_h = base_price .* multiplier;
fprintf('\nSTEP 3: Price Calculation\n');
fprintf('Price range = %.2f to %.2f Rs/kWh\n', ...
        min(price_h), max(price_h));
fprintf('First 5 prices:\n');
disp(price_h(1:5));

%% ---------------------------
% STEP 4: Hourly EV arrivals (from 10-min data)
%% ---------------------------

N_EV_10min = zeros(144,1);

for k = 1:144

    N_EV_10min(k) = getEVcountAtTime( ...
        EV_file, sheetNames_EV, fcsID, (k-1)*10);

end

% Convert to hourly arrivals
N_EV_hour = reshape(N_EV_10min,6,24);

N_EV_FPM = sum(N_EV_hour,1)';     % 24×1

fprintf('\nSTEP 4: Hourly EV Arrivals (FPM)\n');
fprintf('Total FPM EVs = %.2f\n', sum(N_EV_FPM));


%% ---------------------------
% STEP 5: κ normalization
%% ---------------------------

kappa = sum(N_EV_FPM) / sum(tau_h .* N_EV_FPM);

fprintf('\nSTEP 5: Normalization\n');
fprintf('kappa = %.4f\n', kappa);


%% ---------------------------
% STEP 6: PED demand response
%% ---------------------------

gamma = 1;     % PED elasticity factor

N_EV_DPM = kappa * (tau_h.^gamma) .* N_EV_FPM;

fprintf('\nSTEP 6: PED Demand Response\n');
fprintf('Total DPM EVs (fractional) = %.2f\n', sum(N_EV_DPM));


%% ---------------------------
% STEP 7: Integerization
%% ---------------------------

floorEV = floor(N_EV_DPM);

residual = N_EV_DPM - floorEV;

missing = round(sum(N_EV_DPM) - sum(floorEV));

[~,idx] = sort(residual,'descend');

N_EV_int = floorEV;
N_EV_int(idx(1:missing)) = N_EV_int(idx(1:missing)) + 1;

fprintf('\nSTEP 7: Integerization\n');
fprintf('Total EVs (integer) = %d\n', sum(N_EV_int));


%% ---------------------------
% STEP 8: Energy calculation
%% ---------------------------

EV_temp = readmatrix(EV_file,'Sheet',sheetNames_EV{fcsID});

% Extract 10-min charging power
P_EV_10min = EV_temp(2:145,7);

% Convert to hourly power
P_reshaped = reshape(P_EV_10min,6,24);

P_hourly = mean(P_reshaped,1)';   % 24×1

E_EV_FPM_kWh = P_hourly;

fprintf('\nSTEP 8: Hourly FPM Energy\n');
fprintf('Total FPM Energy = %.2f kWh\n', sum(E_EV_FPM_kWh));


%% ---------------------------
% STEP 9: Average energy per EV
%% ---------------------------

E_avg = sum(E_EV_FPM_kWh) / max(sum(N_EV_FPM),1);

fprintf('Average Energy per EV = %.4f kWh\n', E_avg);


%% ---------------------------
% STEP 10: DPM Energy
%% ---------------------------

E_EV_DPM = N_EV_int .* E_avg;

P_EV_DPM = E_EV_DPM;     % hourly slot

fprintf('\nSTEP 10: DPM Energy\n');
fprintf('Total DPM Energy = %.2f kWh\n', sum(E_EV_DPM));

%% ---------------------------
% STEP 11: Export All Results to Excel
%% ---------------------------

hours = (0:23)';

% Recompute FPM arrivals for table (already calculated earlier)
N_EV_10min = zeros(144,1);
for k = 1:144
    N_EV_10min(k) = getEVcountAtTime( ...
        EV_file, sheetNames_EV, fcsID, (k-1)*10);
end

N_EV_hour = reshape(N_EV_10min,6,24);
N_EV_FPM = sum(N_EV_hour,1)';

% Hourly FPM power already calculated
E_FPM = E_EV_FPM_kWh;

% Fractional DPM EVs already available
N_DPM_frac = N_EV_DPM;

% Integerized EVs
N_DPM_int = N_EV_int;

% Energy results
E_DPM = E_EV_DPM;

% Replicate scalar κ
kappa_vec = repmat(kappa,24,1);

% Create table
T_PED = table( ...
    hours, ...
    tau_h, ...
    price_h, ...
    N_EV_FPM, ...
    N_DPM_frac, ...
    N_DPM_int, ...
    E_FPM, ...
    E_DPM, ...
    kappa_vec, ...
    'VariableNames', ...
    {'Hour',...
     'Tau_h',...
     'Price_Rs_per_kWh',...
     'N_EV_FPM',...
     'N_EV_DPM_frac',...
     'N_EV_DPM_int',...
     'E_EV_FPM_kWh',...
     'E_EV_DPM_kWh',...
     'Kappa'});

fprintf('\nSTEP 11: Exporting results to Excel\n');

outputFile = 'PED_24Hour_Detailed_Results.xlsx';
sheetName = sprintf('FCS_%d',fcsID);

writetable(T_PED,outputFile,'Sheet',sheetName);

fprintf('Results written to %s | Sheet: %s\n',outputFile,sheetName);
