clc; clear; close all;

%% ================================
% FILE INPUTS
%% ================================

EV_file  = 'EVFCS_10min_Arrival_Power_fp (3).xlsx';
RTP_file = 'RTP_grid_cost_10min.xlsx';

sheetNames_EV = {
    'EVFCS_1_6_2'
    'EVFCS_2_24_21'
    'EVFCS_3_15_22'
    'EVFCS_4_13_12'
    'EVFCS_5_7_18'
    'EVFCS_6_10_15'
};

%% ================================
% SETTINGS
%% ================================

base_price = 9.58;
tou_prices = [0.6 1.0 1.5];   % OFF MID PEAK multipliers

N_co = [8 5 2 5 4 5];

nFCS = numel(sheetNames_EV);
hours = (0:23)';

fprintf('\n========================================\n');
fprintf('TOU HOURLY WEIGHT GENERATION\n');
fprintf('========================================\n');

%% ================================
% STORAGE
%% ================================

Nev_matrix = zeros(24,nFCS);

%% ================================
% STEP 1: RUN TOU MODEL FOR EACH FCS
%% ================================

for fcsID = 1:nFCS
    
    fprintf('\n----------------------------------------\n');
    fprintf('Running TOU Model for FCS %d\n',fcsID);
    fprintf('Chargers at this station = %d\n',N_co(fcsID));
    fprintf('----------------------------------------\n');
    
    [N_EV_TOU, ~, ~, tau_h, kappa, bounds, price_h] = ...
        compute_TOU_tauDriven( ...
        RTP_file, ...
        EV_file, sheetNames_EV, ...
        fcsID, tou_prices, base_price);
    
    Nev_matrix(:,fcsID) = N_EV_TOU;
    
    fprintf('Hourly EV arrivals after TOU response (FCS %d):\n',fcsID);
    disp(N_EV_TOU');
    
end

%% ================================
% STEP 2: TOTAL EVs PER HOUR
%% ================================

fprintf('\n========================================\n');
fprintf('STEP 2: Hourly EV Totals (All Stations)\n');
fprintf('========================================\n');

N_hour = sum(Nev_matrix,2);

for h = 1:24
    fprintf('Hour %02d : N_h = %.4f EVs\n',h-1,N_hour(h));
end

%% ================================
% STEP 3: TOTAL EVs IN DAY
%% ================================

totalEV = sum(N_hour);

fprintf('\n========================================\n');
fprintf('STEP 3: Total EVs Across Day\n');
fprintf('========================================\n');

fprintf('Total EVs across all hours = %.4f\n',totalEV);

%% ================================
% STEP 4: NORMALIZED WEIGHTS
%% ================================

fprintf('\n========================================\n');
fprintf('STEP 4: Compute Normalized Hourly Weights\n');
fprintf('========================================\n');

W_hour = (N_hour * 9.88) / totalEV;

for h = 1:24
    fprintf('Hour %02d : Weight = %.6f\n',h-1,W_hour(h));
end

fprintf('\nCheck: Sum of weights = %.6f (should be 9.88)\n',sum(W_hour));

%% ================================
% STEP 5: WRITE EXCEL FILE
%% ================================

T = table(hours, N_hour, W_hour,...
    'VariableNames',{'Hour','Total_EV','Normalized_Weight'});

outputFile = 'TOU_Hourly_EV_Weights_AllFCS.xlsx';

writetable(T,outputFile);

fprintf('\n========================================\n');
fprintf('Excel file generated successfully\n');
fprintf('File: %s\n',outputFile);
fprintf('========================================\n');


%% =========================================================
% STEP 6: EXPORT COMPLETE TOU 24-HOUR RESULTS
%% =========================================================

fprintf('\n========================================\n');
fprintf('STEP 6: Export Complete TOU Results\n');
fprintf('========================================\n');

outputFile2 = 'TOU_24Hour_Detailed_Results.xlsx';

for fcsID = 1:nFCS
    
    fprintf('\nProcessing detailed table for FCS %d\n',fcsID);
    
    %% --- Run function again to capture outputs
    
    [N_EV_TOU, N_EV_int, E_EV_TOU, tau_h, kappa, bounds, price_h] = ...
        compute_TOU_tauDriven( ...
        RTP_file, ...
        EV_file, sheetNames_EV, ...
        fcsID, tou_prices, base_price);
    
    %% --- Recompute baseline arrivals
    
    N_EV_10min = zeros(144,1);
    
    for k = 1:144
        N_EV_10min(k) = getEVcountAtTime( ...
            EV_file, sheetNames_EV, fcsID, (k-1)*10);
    end
    
    N_EV_hour = reshape(N_EV_10min,6,24);
    N_FPM = sum(N_EV_hour,1)';
    
    %% --- Compute tau_new again
    
    inv_price = 1 ./ price_h;
    tau_new = inv_price ./ max(inv_price);
    
    %% --- Energy baseline
    
    EV_temp = readmatrix(EV_file,'Sheet',sheetNames_EV{fcsID});
    
    P_EV_10min = EV_temp(2:145,7);
    
    P_reshaped = reshape(P_EV_10min,6,24);
    
    P_hourly = mean(P_reshaped,1)';
    
    E_FPM_hour = P_hourly;
    
    %% --- Create table
    
    T_detail = table( ...
        hours, ...
        tau_h, ...
        tau_new, ...
        price_h, ...
        N_FPM, ...
        N_EV_TOU, ...
        N_EV_int, ...
        E_FPM_hour, ...
        E_EV_TOU, ...
        'VariableNames', ...
        {'Hour',...
         'Tau_Base',...
         'Tau_TOU',...
         'Price_Rs_per_kWh',...
         'N_EV_FPM',...
         'N_EV_TOU_frac',...
         'N_EV_TOU_int',...
         'E_EV_FPM_kWh',...
         'E_EV_TOU_kWh'});
    
    sheetName = sprintf('FCS_%d',fcsID);
    writetable(T_detail,outputFile2,'Sheet',sheetName);
    
end

fprintf('\n========================================\n');
fprintf('Detailed TOU Excel file generated\n');
fprintf('File: %s\n',outputFile2);
fprintf('========================================\n');
