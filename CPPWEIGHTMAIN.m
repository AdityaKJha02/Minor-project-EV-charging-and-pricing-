clc; clear; close all;

%% ================================
% SETTINGS
%% ================================

EV_file  = 'EVFCS_10min_Arrival_Power_fp (3).xlsx';

sheetNames_EV = {
    'EVFCS_1_6_2'
    'EVFCS_2_24_21'
    'EVFCS_3_15_22'
    'EVFCS_4_13_12'
    'EVFCS_5_7_18'
    'EVFCS_6_10_15'
};

RTP_file = 'RTP_grid_cost_10min.xlsx';

base_price = 8.68;
cpp_multiplier = 2;

N_co = [8 5 2 5 4 5];

nFCS = numel(sheetNames_EV);
hours = (0:23)';

fprintf('\n========================================\n');
fprintf('CPP HOURLY WEIGHT GENERATION\n');
fprintf('========================================\n');

%% ================================
% STORAGE
%% ================================

Nev_matrix = zeros(24,nFCS);

%% ================================
% STEP 1: RUN CPP MODEL FOR EACH FCS
%% ================================

for fcsID = 1:nFCS
    
    fprintf('\n----------------------------------------\n');
    fprintf('Running CPP Model for FCS %d\n',fcsID);
    fprintf('Number of chargers = %d\n',N_co(fcsID));
    fprintf('----------------------------------------\n');
    
    [N_EV_CPP, ~, ~, tau_new, kappa, cpp_idx, price_h] = ...
        compute_CPP_tauDriven( ...
        RTP_file, ...
        EV_file, sheetNames_EV, ...
        fcsID, cpp_multiplier, base_price);
    
    Nev_matrix(:,fcsID) = N_EV_CPP;
    
    fprintf('CPP hours for FCS %d: ',fcsID);
    fprintf('%d ', cpp_idx-1);
    fprintf('\n');
    
    fprintf('Hourly EV arrivals after CPP response:\n');
    disp(N_EV_CPP');
    
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
% STEP 5: EXPORT TO EXCEL
%% ================================

T = table(hours, N_hour, W_hour,...
    'VariableNames',{'Hour','Total_EV','Normalized_Weight'});

outputFile = 'CPP_Hourly_EV_Weights_AllFCS.xlsx';

writetable(T,outputFile);

fprintf('\n========================================\n');
fprintf('Excel file generated successfully\n');
fprintf('File: %s\n',outputFile);
fprintf('========================================\n');

%% =========================================================
% STEP 6: EXPORT 24-HOUR CPP PRICING MECHANISM
%% =========================================================

fprintf('\n========================================\n');
fprintf('STEP 6: Export 24-Hour CPP Pricing Mechanism\n');
fprintf('========================================\n');

%% Recompute base tau from RTP

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

%% CPP flag (1 = CPP hour)

cpp_flag = zeros(24,1);
cpp_flag(cpp_idx) = 1;

%% Display table in console

fprintf('\nHour  Tau_base  Tau_new  CPP_flag  Price\n');

for h = 1:24
    fprintf('%2d   %8.4f   %8.4f   %5d     %8.4f\n',...
        h-1,tau_base(h),tau_new(h),cpp_flag(h),price_h(h));
end

%% Create table for Excel

T_price = table( ...
    hours, ...
    tau_base, ...
    tau_new, ...
    cpp_flag, ...
    price_h, ...
    'VariableNames', ...
    {'Hour','Tau_Base','Tau_CPP','CPP_Flag','Price_Rs_per_kWh'});

priceFile = 'CPP_24Hour_Pricing_Mechanism.xlsx';

writetable(T_price,priceFile);

fprintf('\n========================================\n');
fprintf('CPP pricing mechanism exported\n');
fprintf('File: %s\n',priceFile);
fprintf('========================================\n');


%% =========================================================
% STEP 6: EXPORT COMPLETE CPP RESULTS (24 HOURS)
%% =========================================================

fprintf('\n========================================\n');
fprintf('STEP 6: Export Complete CPP Results\n');
fprintf('========================================\n');

outputFile2 = 'CPP_24Hour_All_Results.xlsx';

if exist(outputFile2,'file')
    delete(outputFile2);
end

for fcsID = 1:nFCS
    
    fprintf('\nProcessing detailed results for FCS %d\n',fcsID);
    
    [N_EV_CPP, N_EV_int, E_EV_CPP, tau_new, kappa, cpp_idx, price_h] = ...
        compute_CPP_tauDriven( ...
        RTP_file, ...
        EV_file, sheetNames_EV, ...
        fcsID, cpp_multiplier, base_price);
    
    %% CPP flag vector
    
    cpp_flag = zeros(24,1);
    cpp_flag(cpp_idx) = 1;
    
    %% replicate scalar values
    
    kappa_vec = repmat(kappa,24,1);
    
    %% create table
    
    T_CPP = table( ...
        hours, ...
        tau_new, ...
        price_h, ...
        cpp_flag, ...
        N_EV_CPP, ...
        N_EV_int, ...
        E_EV_CPP, ...
        kappa_vec, ...
        'VariableNames', ...
        {'Hour',...
         'Tau_CPP',...
         'Price_Rs_per_kWh',...
         'CPP_Flag',...
         'N_EV_CPP_frac',...
         'N_EV_CPP_int',...
         'E_EV_CPP_kWh',...
         'Kappa'});
    
    %% write sheet
    
    sheetName = sprintf('FCS_%d',fcsID);
    
    writetable(T_CPP,outputFile2,'Sheet',sheetName);
    
    fprintf('Sheet written: %s\n',sheetName);
    
end

fprintf('\n========================================\n');
fprintf('Detailed CPP Excel file generated\n');
fprintf('File: %s\n',outputFile2);
fprintf('========================================\n');
