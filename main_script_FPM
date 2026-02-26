clc; clear; close all;

%% =========================================================
%% 15. FPM ALGORITHM – COMPLETE MAIN SCRIPT
%% =========================================================
EV_file  = 'EVFCS_10min_Arrival_Power.xlsx';
PV_file  = 'PV_Results_6_NPV_10min.xlsx';
RTP_file = 'RTP_grid_cost_10min.xlsx';
tempExcel = 'TEMP_BSA_FPM.xlsx';

sheetNames_EV = {
   'EVFCS_1_6_2'
   'EVFCS_2_24_21'
   'EVFCS_3_15_22'
   'EVFCS_4_13_12'
   'EVFCS_5_7_18'
   'EVFCS_6_10_15'
};

sheetNames_PV = {
   'NPV_400'
   'NPV_390'
   'NPV_420'
   'NPV_405'
   'NPV_397'
   'NPV_408'
};

%% =========================================================
% STATION DATA
%% =========================================================
N_co = [8 5 2 5 4 5];          % chargers per FCS
nFCS = numel(N_co);
SOC_init_vec = [50 55 70 55 60 60];   % one value per FCS
CD_factor_vec = [1.44 1.58 3.13 1.49 1.67 1.61];
PV_module_area = 2.584;          % m^2

A_fc = zeros(1,nFCS);
for i = 1:nFCS
    A_fc(i) = Area_calc(N_co(i));
end

N_pv = round(0.85 .* A_fc ./ PV_module_area);

%% =========================================================
% BATTERY SIZE SWEEP
%% =========================================================
batterySizes = 300:200:2500;   % kWh sweep
nB = numel(batterySizes);
FP_vec        = zeros(nB,1);
FixedCost_vec = zeros(nB,1);
VarCost_vec   = zeros(nB,1);
TotalCost_vec = zeros(nB,1);
Revenue_vec   = zeros(nB,1);
PM_vec        = zeros(nB,1);

%% =========================================================
% FILE INPUTS
%% =========================================================


%% =========================================================
% SUBSTATION CAPACITY
%% =========================================================
beta_SS = zeros(1,nFCS);

for i = 1:nFCS
    EV_temp = readmatrix(EV_file,'Sheet',sheetNames_EV{i});
    beta_SS(i) = max(EV_temp(:,7));   % EV power column
end

%% =========================================================
% FPM PARAMETERS
%% =========================================================
params.dt           = 1/6;        % 10 min
params.PM_DES       = 0.25;       % desired profit margin
params.rho_SELL_0   = 0.1;        % starting flat price
params.rho_SELL_max = 30;         % max search limit
params.delta_rho    = 0.01;       % search step

rtp = readmatrix(RTP_file);
params.rho_PUR = rtp(1:144,2);    % purchase price

%% =========================================================
% MAIN LOOP
%% =========================================================
for k = 1:nB

    batterySize = batterySizes(k);

    fprintf('\nRunning BESS size = %d kWh\n', batterySize);

    battery_vec = batterySize * ones(1,nFCS);

    % ------------------------------
    % Fixed Cost (Eq. 35)
    % ------------------------------
    C_FIX_vec   = Cd_fixcost_FPM(battery_vec, N_co, beta_SS, N_pv);
    C_OandM_vec = Cd_OandM_FPM(C_FIX_vec);

    params.C_FIX   = C_FIX_vec;
    params.C_OandM = C_OandM_vec;
    params.E_BESS  = batterySize;

    % ------------------------------
    % BSA (Battery Scheduling)
    % ------------------------------
    BSA = BSA_step1_step2( ...
        EV_file, PV_file, RTP_file, ...
        sheetNames_EV, sheetNames_PV, ...
        batterySize, SOC_init_vec, CD_factor_vec);

    % Generate temporary Excel
    BSA_step5_generateExcel(BSA, tempExcel);

    % ------------------------------
    % Run FPM pricing search
    % ------------------------------
    FPM = run_FPM_from_BSA_Excel(params, tempExcel);

    % ------------------------------
    % Store outputs
    % ------------------------------
    FP_vec(k)        = FPM.rho_SELL;
    Revenue_vec(k)   = FPM.Total_Revenue;
    TotalCost_vec(k) = FPM.Total_Cost;
    PM_vec(k)        = FPM.PM;

    FixedCost_vec(k) = sum(C_FIX_vec) + sum(C_OandM_vec);
    VarCost_vec(k)   = FPM.Total_Cost - FixedCost_vec(k);

end

%% =========================================================
% SAVE RESULTS
%% =========================================================
outputFile = 'FPM_BESS_Results.xlsx';

if exist(outputFile,'file')
    delete(outputFile);
end

T = table( ...
    batterySizes', ...
    FP_vec, ...
    Revenue_vec, ...
    FixedCost_vec, ...
    VarCost_vec, ...
    TotalCost_vec, ...
    PM_vec, ...
    'VariableNames', ...
    {'Battery_kWh', ...
     'FlatPrice_Rs_per_kWh', ...
     'Revenue_Rs', ...
     'FixedCost_Rs', ...
     'VariableCost_Rs', ...
     'TotalCost_Rs', ...
     'ProfitMargin'} );

writetable(T, outputFile);

disp('Excel file generated: FPM_BESS_Results.xlsx');

%% =========================================================
% FIGURE – FLAT PRICE VS BATTERY SIZE
%% =========================================================
figure;
plot(batterySizes, FP_vec, '-o','LineWidth',2);
grid on;
xlabel('Battery capacity (kWh)');
ylabel('Flat price (Rs/kWh)');
title('Optimized Flat Price vs Battery Size');

%% =========================================================
% EXPORT DETAILED ECONOMIC RESULTS
%% =========================================================
outputFile = 'Fig7_FPM.xlsx';
if exist(outputFile,'file')
   delete(outputFile);
end
Results_Table = table( ...
   batterySizes', ...
   FP_vec, ...
   FixedCost_vec, ...
   VarCost_vec, ...
   TotalCost_vec, ...
   Revenue_vec, ...
   PM_vec, ...
   'VariableNames', ...
   {'BESS_kWh', ...
    'EV_PRICE', ...
    'Fixed_Cost', ...
    'Variable_Cost', ...
    'Total_Cost', ...
    'Revenue', ...
    'PM'} );
writetable(Results_Table, outputFile);
%% =========================================================
% FIND OPTIMAL (MINIMUM FLAT PRICE)
%% =========================================================
[FP_min, idx_min] = min(FP_vec);
Fixed_opt  = FixedCost_vec(idx_min);
Var_opt    = VarCost_vec(idx_min);
Total_opt  = TotalCost_vec(idx_min);
Revenue_opt= Revenue_vec(idx_min);
PM_opt     = PM_vec(idx_min);

%% =========================================================
% LEAVE 2 ROWS GAP AND WRITE OPTIMAL VALUES
%% =========================================================
gapRow = height(Results_Table) + 4;   % 2 empty rows
OptimalSummary = {
   'BESS_kWh',        batterySizes(idx_min);
   'EV_PRICE',        FP_min;
   'Fixed_Cost',      Fixed_opt;
   'Variable_Cost',   Var_opt;
   'Total_Cost',      Total_opt;
   'Revenue',         Revenue_opt;
   'PM',              PM_opt
};
writecell(OptimalSummary, outputFile, ...
   'Range', sprintf('A%d', gapRow));
disp('Economic Excel file generated successfully.');


%% =========================================================
% REGENERATE BSA FILE FOR OPTIMAL BATTERY SIZE
%% =========================================================

optimalBattery = batterySizes(idx_min);

fprintf('\nRe-running BSA for optimal battery size = %d kWh\n', optimalBattery);

% Recompute battery vector
battery_vec_opt = optimalBattery * ones(1,nFCS);

% Recompute fixed cost (optional but clean)
C_FIX_vec   = Cd_fixcost_FPM(battery_vec_opt, N_co, beta_SS, N_pv);
C_OandM_vec = Cd_OandM_FPM(C_FIX_vec);

% Update params
params.C_FIX   = C_FIX_vec;
params.C_OandM = C_OandM_vec;
params.E_BESS  = optimalBattery;

% Run BSA again
BSA_opt = BSA_step1_step2( ...
    EV_file, PV_file, RTP_file, ...
    sheetNames_EV, sheetNames_PV, ...
    optimalBattery, SOC_init_vec, CD_factor_vec);

% Overwrite TEMP_BSA_FPM.xlsx with optimal result
BSA_step5_generateExcel(BSA_opt, tempExcel);

disp('✔ TEMP_BSA_FPM.xlsx overwritten with optimal battery results');
