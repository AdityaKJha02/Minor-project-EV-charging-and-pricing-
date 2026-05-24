function export_FPM_results_to_excel(FPM, BSA_excel, outputExcel)
% =========================================================
% EXPORT FINAL FPM RESULTS (Paper Fig. 2 - FINAL BLOCK)
% Uses ONLY existing FPM + BSA Excel outputs
% =========================================================

if exist(outputExcel,'file')
    delete(outputExcel);
end

dt = 1/6;

%% -------------------------------------------------------
% 1. WRITE SUMMARY (GLOBAL RESULTS)
% -------------------------------------------------------
summary = {
    'Optimal_EV_Selling_Price',  FPM.rho_SELL;
    'Profit_Margin',             FPM.PM;
    'Total_Revenue',             FPM.Total_Revenue;
    'Total_Cost',                FPM.Total_Cost
};

writecell(summary, outputExcel, 'Sheet', 'FPM_Summary', 'Range', 'A1');

%% -------------------------------------------------------
% 2. READ FCS SHEETS FROM BSA EXCEL
% -------------------------------------------------------
[~, sheetNames] = xlsfinfo(BSA_excel);
nFCS = numel(sheetNames);

for i = 1:nFCS

    M = readmatrix(BSA_excel,'Sheet',sheetNames{i});

    % -------------------------------
    % Column mapping (as you specified)
    % -------------------------------
    Time_min   = M(:,1);
    P_EV       = M(:,3);    % EV power
    P_BC       = M(:,6);    % Battery charging
    P_BD       = M(:,7);    % Battery discharging
    SOC        = M(:,8);    % SOC
    P_Grid     = M(:,9);    % Grid power
    E_Grid     = M(:,10);   % Grid energy

    % Remove NaNs
    valid = ~isnan(P_EV);
    Time_min = Time_min(valid);
    P_EV     = P_EV(valid);
    P_BC     = P_BC(valid);
    P_BD     = P_BD(valid);
    SOC      = SOC(valid);
    P_Grid   = P_Grid(valid);
    E_Grid   = E_Grid(valid);

    % EV energy (E_D,EV^h,FPM)
    E_EV_day = sum(P_EV) * dt;

    % -------------------------------
    % Table to export
    % -------------------------------
    T = table( ...
        Time_min, ...
        P_EV, ...
        P_BC, ...
        P_BD, ...
        SOC, ...
        P_Grid, ...
        E_Grid, ...
        'VariableNames', { ...
        'Time_min', ...
        'P_EV_kW', ...
        'P_BC_kW', ...
        'P_BD_kW', ...
        'SOC_pct', ...
        'P_Grid_kW', ...
        'E_Grid_kWh' });

    sheetOut = sprintf('FCS_%d_FPM', i);
    writetable(T, outputExcel, 'Sheet', sheetOut, 'Range', 'A1');

    % -------------------------------
    % FCS summary (right side)
    % -------------------------------
    summaryFCS = {
        'EV_Energy_kWh',      E_EV_day;
        'Grid_Energy_kWh',    sum(E_Grid);
        'Battery_Charge_kWh', dt*sum(P_BC);
        'Battery_Discharge_kWh', dt*sum(P_BD)
    };

    writecell(summaryFCS, outputExcel, ...
        'Sheet', sheetOut, 'Range', 'J2');
end

disp('✔ FPM FINAL RESULTS EXPORTED SUCCESSFULLY');
end
