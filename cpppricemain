
fprintf('\n========================================\n');
fprintf('   CPP BASE PRICE SEARCH STARTED\n');
fprintf('========================================\n');

%% ============================
% USER SETTINGS
%% ============================

base_start = 8.8;      % Rs/kWh
base_end   = 60;     % Rs/kWh
step       = .1;

PM_DES = 0.25;

battery_kWh = 1600;

timeGrid_min = (0:10:1430)';

%% ============================
% FILES
%% ============================

EV_file        = 'EVFCS_10min_Arrival_Power.xlsx';
RTP_file       = 'RTP_grid_cost_10min.xlsx';
TEMP_BSA_file  = 'TEMP_BSA_FPM.xlsx';
CPP_excel      = 'CPP_Detailed_Results_AllFCS.xlsx';

sheetNames_EV = {
 'EVFCS_1_6_2'
 'EVFCS_2_24_21'
 'EVFCS_3_15_22'
 'EVFCS_4_13_12'
 'EVFCS_5_7_18'
 'EVFCS_6_10_15'
};

nFCS = numel(sheetNames_EV);

%% ============================
% STATION DATA
%% ============================

N_co = [13 10 2 10 6 8];

PV_module_area = 2.6;

for i = 1:nFCS
    A_fc(i) = Area_calc(N_co(i));
    N_pv(i) = round(0.85 * A_fc(i) / PV_module_area);
end

for i = 1:nFCS
    M = readmatrix(EV_file,'Sheet',sheetNames_EV{i});
    beta_SS(i) = max(M(:,7));   % EV charging power
end

%% ============================
% BASE PRICE LOOP
%% ============================

for base_price = base_start:step:base_end

    OUT = cpp_revenue_energy_block( ...
        base_price, ...
        timeGrid_min, ...
        EV_file, RTP_file, ...
        sheetNames_EV, ...
        TEMP_BSA_file, CPP_excel, ...
        battery_kWh, N_co, beta_SS, N_pv);

    if OUT.PM >= PM_DES

        fprintf('\n====================================\n');
        fprintf(' CPP BASE PRICE FOUND\n');
        fprintf(' Base Price  = %.2f Rs/kWh\n', base_price);
        fprintf(' ProfitMargin= %.4f\n', OUT.PM);
        fprintf(' Revenue     = %.2f Rs\n', OUT.Revenue);
        fprintf(' Cost        = %.2f Rs\n', OUT.Cd);
        fprintf('====================================\n');

        break;

    end

end
