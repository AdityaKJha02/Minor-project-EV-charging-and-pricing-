function [P_PV_avg, P_PV_max, E_avg, E_max, hours, G_avg, G_max] = ...
    PV_Power_Model_Avg_Max(excelFile, N_PV)
% ============================================================
% PV_Power_Model_Avg_Max
% ============================================================
% Calculates PV power using average and maximum hourly irradiance
%
% EXCEL FORMAT:
% Row 1  : Hours (1–24)
% Row 34 : Average irradiance G_avg(h) (W/m^2)
% Row 38 : Maximum irradiance G_max(h) (W/m^2)
%
% INPUTS:
% excelFile       : Excel file path
% N_PV            : Number of PV modules
% P_module_max    : Max power per module (W)
%
% OUTPUTS:
% P_PV_avg        : PV power using average irradiance (W)
% P_PV_max        : PV power using max irradiance (W)
% E_avg           : Daily PV energy (avg) (kWh)
% E_max           : Daily PV energy (max) (kWh)
% hours           : Hour vector (1–24)
% G_avg           : Average irradiance profile
% G_max           : Maximum irradiance profile
% ============================================================

    % ---- Constant ----
    G_ref = 1000;   % W/m^2
    P_module_max = 125;   %W

    % ---- Read Excel safely ----
    T = readtable(excelFile, 'VariableNamingRule', 'preserve');

    % ---- Extract hours ----
    hours = table2array(T(1, 2:end));

    % ---- Extract irradiance rows ----
    G_avg = table2array(T(34, 2:end));
    G_max = table2array(T(38, 2:end));

    % ---- Ensure numeric ----
    G_avg = double(G_avg);
    G_max = double(G_max);

    % ---- Handle missing values ----
    G_avg(isnan(G_avg)) = 0;
    G_max(isnan(G_max)) = 0;

    % ---- Scale if normalized ----
    if max(G_avg) <= 1
        G_avg = G_avg * 1000;
    end
    if max(G_max) <= 1
        G_max = G_max * 1000;
    end

    % ---- PV power calculation (W) ----
    P_PV_avg = N_PV .* P_module_max .* (G_avg ./ G_ref);
    P_PV_max = N_PV .* P_module_max .* (G_max ./ G_ref);

    % ---- Daily energy (kWh) ----
    E_avg = sum(P_PV_avg) / 1000;
    E_max = sum(P_PV_max) / 1000;

end
