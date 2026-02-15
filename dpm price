%% =========================================================
%% 20. DPM BASE PRICE
%% =========================================================


fprintf('\n========================================\n');
fprintf('   DPM BASE PRICE SEARCH STARTED\n');
fprintf('========================================\n');

%% ======================================================
% USER SETTINGS
%% ======================================================

base_start = 9.6;      % Rs/kWh
base_end   = 50;     % Rs/kWh
step       = .1;      % Rs/kWh

PM_DES = 0.25;       % desired profit margin

battery_kWh = 1600;

timeGrid_min = (0:10:1430)';

EV_file  = 'EVFCS_10min_Arrival_Power.xlsx';
RTP_file = 'RTP_grid_cost_10min.xlsx';

TEMP_BSA_file = 'TEMP_BSA_FPM.xlsx';
PED_excel     = 'PED_Detailed_Results_AllFCS.xlsx';

sheetNames_EV = {
 'EVFCS_1_6_2'
 'EVFCS_2_24_21'
 'EVFCS_3_15_22'
 'EVFCS_4_13_12'
 'EVFCS_5_7_18'
 'EVFCS_6_10_15'
};

nFCS = numel(sheetNames_EV);

%% ======================================================
% STATION PARAMETERS
%% ======================================================

N_co = [13 10 2 10 6 8];

PV_module_area = 2.6;

A_fc = zeros(1,nFCS);
N_pv = zeros(1,nFCS);

for i = 1:nFCS
    A_fc(i) = Area_calc(N_co(i));
    N_pv(i) = round(0.85*A_fc(i)/PV_module_area);
end

beta_SS = zeros(1,nFCS);

for i = 1:nFCS
    M = readmatrix(EV_file,'Sheet',sheetNames_EV{i});
    beta_SS(i) = max(M(:,7));
end

%% ======================================================
% BASE PRICE SEARCH LOOP
%% ======================================================

for base_price = base_start:step:base_end

    OUT = dpm_revenue_energy_block( ...
        base_price, ...
        timeGrid_min, ...
        EV_file, RTP_file, ...
        sheetNames_EV, ...
        TEMP_BSA_file, PED_excel, ...
        battery_kWh, N_co, beta_SS, N_pv);

    if OUT.PM >= PM_DES

        fprintf('\n========================================\n');
        fprintf(' BASE PRICE FOUND = %.2f Rs/kWh\n',base_price);
        fprintf(' PROFIT MARGIN   = %.4f\n',OUT.PM);
        fprintf(' TOTAL REVENUE   = %.2f Rs\n',OUT.Revenue);
        fprintf(' TOTAL COST      = %.2f Rs\n',OUT.Cd);
        fprintf('========================================\n');

        RESULT.base_price = base_price;
        RESULT.Revenue   = OUT.Revenue;
        RESULT.Cost      = OUT.Cd;
        RESULT.PM        = OUT.PM;

        save('DPM_BasePrice_Result.mat','RESULT');
        return;
    end

end

%% ======================================================
% IF NOT FOUND
%% ======================================================

fprintf('\n========================================\n');
fprintf(' No base price meets PM >= %.2f\n',PM_DES);
fprintf('========================================\n');

RESULT.base_price = NaN;
RESULT.Revenue   = NaN;
RESULT.Cost      = NaN;
RESULT.PM        = NaN;

save('DPM_BasePrice_Result.mat','RESULT');
