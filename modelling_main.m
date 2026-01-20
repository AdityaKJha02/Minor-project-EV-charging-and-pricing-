clc;
clear;

% =====================================================
% INPUTS
% =====================================================
excelFile   = 'June_Irradiance.xlsx';
outputExcel = 'PV_Results_6_NPV.xlsx';

N_co = [5, 2, 1, 1, 1, 2];
PV_module_max=550;
PV_module_area=2.6;
A_fc = zeros(1, length(N_co));
for i = 1:length(N_co)
    A_fc(i) = Area_calc(N_co(i));
end

N_pv=zeros(1,length(N_co));
for i = 1:length(N_co)
    N_pv(i) = round(0.85*A_fc(i)/PV_module_area);
end
disp(N_pv)
nCases = length(N_pv);

% =====================================================
% PRE-ALLOCATE STORAGE
% =====================================================
P_avg_all = cell(1, nCases);
P_max_all = cell(1, nCases);
E_avg_all = zeros(1, nCases);
E_max_all = zeros(1, nCases);

% =====================================================
% FUNCTION CALL LOOP
% =====================================================
for i = 1:nCases

    [P_avg, P_max, E_avg, E_max, hours, G_avg, G_max] = ...
        PV_Power_Model_Avg_Max(excelFile, N_pv(i));

    P_avg_all{i} = P_avg;
    P_max_all{i} = P_max;
    E_avg_all(i) = E_avg;
    E_max_all(i) = E_max;

end

% =====================================================
% DISPLAY ENERGY RESULTS
% =====================================================
disp('Average-day PV Energy (kWh):')
disp(E_avg_all)

disp('Maximum-day PV Energy (kWh):')
disp(E_max_all)

% =====================================================
% FIGURE: AVG & MAX IN SAME SUBPLOT (6 SUBPLOTS)
% =====================================================
figure('Name','PV Power Output (Avg vs Max Irradiance)','NumberTitle','off');

for i = 1:nCases
    subplot(3,2,i)

    plot(hours, P_avg_all{i}/1000, '-o','LineWidth',1.5)
    hold on
    plot(hours, P_max_all{i}/1000, '--s','LineWidth',1.5)
    hold off

    xlabel('Hour')
    ylabel('PV Power (kW)')
    title(['NPV = ', num2str(N_pv(i))])
    legend('Average Irradiance','Maximum Irradiance','Location','best')
    grid on
end

% =====================================================
% WRITE RESULTS TO EXCEL (6 SHEETS)
% =====================================================
for i = 1:nCases

    % ---- Hourly PV power table ----
    T = table( ...
        hours(:), ...
        P_avg_all{i}(:)/1000, ...
        P_max_all{i}(:)/1000, ...
        'VariableNames', {'Hour','PV_Power_Avg_kW','PV_Power_Max_kW'});

    sheetName = ['NPV_', num2str(N_pv(i))];

    % ---- Write hourly data ----
    writetable(T, outputExcel, 'Sheet', sheetName, 'Range', 'A1');

    % ---- Write energy summary ----
    energySummary = {
        'Total_Avg_Energy_kWh', E_avg_all(i);
        'Total_Max_Energy_kWh', E_max_all(i)
    };

    writecell(energySummary, outputExcel, ...
        'Sheet', sheetName, 'Range', 'E2');

end

disp('✔ Graphs generated and Excel file with 6 sheets created successfully.');
