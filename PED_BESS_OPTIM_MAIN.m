clc; clear; close all;

%% =========================================================
% FILES
%% =========================================================
EV_file     = 'EVFCS_10min_Arrival_Power_DPM_FINAL.xlsx';
PV_file     = 'PV_Results_6_NPV_10min.xlsx';
RTP_file    = 'RTP_grid_cost_10min.xlsx';

BSA_excel   = 'TEMP_BSA_PED.xlsx';
PED_excel   = 'PED_DPM_Hourly_AllFCS.xlsx';

%% =========================================================
% STATION CONFIG
%% =========================================================
sheetNames_EV = {
   'EVFCS_1_2_6'
   'EVFCS_2_24_21'
   'EVFCS_3_15_22'
   'EVFCS_4_12_13'
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

nFCS = numel(sheetNames_EV);
% Storage for ALL iterations (Excel table)
BESS_all   = [];
Price_all  = [];
Fix_all    = [];
Var_all    = [];
Tot_all    = [];
Rev_all    = [];
PM_all     = [];

N_co = [8 5 2 5 4 5];
SOC_init_vec = [50 55 50 55 50 60];
CD_factor_vec = [1.44 1.58 3.13 1.49 1.67 1.61];

PV_module_area = 2.584;
params.timeGrid_min = 0:10:1430;
params.RTP_file     = RTP_file;
params.EV_file      = EV_file;
params.sheetNames_EV = sheetNames_EV;
params.summary_file = 'EVFCS_FCS_Summary_fp (2).xlsx';

%% =========================================================
% AREA + PV
%% =========================================================
A_fc = zeros(1,nFCS);

for i = 1:nFCS
    A_fc(i) = Area_calc(N_co(i));
end

N_pv = round(0.85 .* A_fc ./ PV_module_area);

%% =========================================================
% SUBSTATION CAPACITY
%% =========================================================
beta_SS = zeros(1,nFCS);

for i = 1:nFCS
    EV_temp = readmatrix(EV_file,'Sheet',sheetNames_EV{i});
    beta_SS(i) = max(EV_temp(:,7));
end

%% =========================================================
% BATTERY SWEEP
%% =========================================================
batterySizes = 1100:100:1600;
nB = numel(batterySizes);

BasePrice_vec  = zeros(nB,1);
FixedCost_vec  = zeros(nB,1);
VarCost_vec    = zeros(nB,1);
TotalCost_vec  = zeros(nB,1);
Revenue_vec    = zeros(nB,1);
PM_vec         = zeros(nB,1);

%% =========================================================
% RTP
%% =========================================================
rtp = readmatrix(RTP_file);

params.rho_PUR = rtp(2:145,2);
params.dt      = 1/6;

%% =========================================================
% PED PARAMETERS (same logic as FPM)
%% =========================================================
params.PM_DES       = 0.25;
params.basePrice0   = 08.70;% 8.74 at 1300 as on 07-03
params.basePriceMax = 30;
params.deltaPrice   = 0.01;

%% =========================================================
% MAIN LOOP
%% =========================================================
%% =========================================================
% MAIN LOOP (Same structure as FPM)
%% =========================================================

for k = 1:nB

    batterySize = batterySizes(k);

    fprintf('\nRunning PED–DPM for BESS = %d kWh\n', batterySize);

    battery_vec = batterySize * ones(1,nFCS);

    % ------------------------------
    % Fixed Cost
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

    % Generate BSA Excel
    BSA_step5_generateExcel(BSA, BSA_excel);

    % ------------------------------
    % Run PED–DPM pricing optimisation
    % (PRICE SEARCH INSIDE FUNCTION)
    % ------------------------------
    PED = PED_DPM_from_BSA_Excel(params, BSA_excel, PED_excel);

    % ------------------------------
    % Store outputs
    % ------------------------------
    BasePrice_vec(k) = PED.basePrice;
    Revenue_vec(k)   = PED.Total_Revenue;
    TotalCost_vec(k) = PED.Total_Cost;
    PM_vec(k)        = PED.PM;

    FixedCost_vec(k) = sum(C_FIX_vec) + sum(C_OandM_vec);
    VarCost_vec(k)   = PED.Total_Cost - FixedCost_vec(k);

end
%% =========================================================
% SAVE RESULTS
%% =========================================================
outputFile = 'PED_DPM_BESS_Results.xlsx';

if exist(outputFile,'file')
    delete(outputFile);
end

T = table( ...
    batterySizes', ...
    BasePrice_vec, ...
    Revenue_vec, ...
    FixedCost_vec, ...
    VarCost_vec, ...
    TotalCost_vec, ...
    PM_vec, ...
    'VariableNames', ...
    {'Battery_kWh', ...
     'BasePrice_Rs_per_kWh', ...
     'Revenue_Rs', ...
     'FixedCost_Rs', ...
     'VariableCost_Rs', ...
     'TotalCost_Rs', ...
     'ProfitMargin'} );

writetable(T, outputFile);

disp('Excel file generated: PED_DPM_BESS_Results.xlsx');


%% =========================================================
% FIGURE – BASE PRICE VS BATTERY SIZE
%% =========================================================
figure(1); clf;

plot(batterySizes, BasePrice_vec, '-o','LineWidth',2);

grid on;
xlabel('Battery capacity (kWh)');
ylabel('Base price (Rs/kWh)');
title('Optimized Base Price vs Battery Size');


figure(2); clf;

plot(batterySizes, PM_vec, '-s','LineWidth',2);

grid on;
xlabel('Battery capacity (kWh)');
ylabel('Profit Margin');
title('PED–DPM Profit Margin vs Battery Size');

%% =========================================================
% EXPORT DETAILED ECONOMIC RESULTS
%% =========================================================
outputFile = 'Fig7_PED_DPM.xlsx';

if exist(outputFile,'file')
   delete(outputFile);
end

Results_Table = table( ...
   batterySizes', ...
   BasePrice_vec, ...
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
% FIND OPTIMAL (MINIMUM BASE PRICE)
%% =========================================================
[Price_min, idx_min] = min(BasePrice_vec);

Fixed_opt   = FixedCost_vec(idx_min);
Var_opt     = VarCost_vec(idx_min);
Total_opt   = TotalCost_vec(idx_min);
Revenue_opt = Revenue_vec(idx_min);
PM_opt      = PM_vec(idx_min);


%% =========================================================
% LEAVE 2 ROWS GAP AND WRITE OPTIMAL VALUES
%% =========================================================
gapRow = height(Results_Table) + 4;

OptimalSummary = {
   'BESS_kWh',        batterySizes(idx_min);
   'EV_PRICE',        Price_min;
   'Fixed_Cost',      Fixed_opt;
   'Variable_Cost',   Var_opt;
   'Total_Cost',      Total_opt;
   'Revenue',         Revenue_opt;
   'PM',              PM_opt
};

writecell(OptimalSummary, outputFile, ...
   'Range', sprintf('A%d', gapRow));

disp('PED–DPM economic Excel file generated successfully.');

%% =========================================================
% REGENERATE BSA FILE FOR OPTIMAL BATTERY SIZE (PED–DPM)
%% =========================================================

optimalBattery = batterySizes(idx_min);

fprintf('\nRe-running BSA for optimal battery size = %d kWh (PED–DPM)\n', optimalBattery);

% Recompute battery vector
battery_vec_opt = optimalBattery * ones(1,nFCS);

% Recompute fixed costs
C_FIX_vec   = Cd_fixcost_FPM(battery_vec_opt, N_co, beta_SS, N_pv);
C_OandM_vec = Cd_OandM_FPM(C_FIX_vec);

% Update parameters
params.C_FIX   = C_FIX_vec;
params.C_OandM = C_OandM_vec;
params.E_BESS  = optimalBattery;

%% ------------------------------
% Run BSA again
%% ------------------------------
BSA_opt = BSA_step1_step2( ...
    EV_file, PV_file, RTP_file, ...
    sheetNames_EV, sheetNames_PV, ...
    optimalBattery, SOC_init_vec, CD_factor_vec);

%% ------------------------------
% Overwrite BSA Excel with optimal result
%% ------------------------------
BSA_step5_generateExcel(BSA_opt, BSA_excel);

disp('✔ TEMP_BSA_PED.xlsx overwritten with optimal battery results');
