clc; clear; close all;

%% Add MATPOWER path
addpath(genpath('C:\MATLAB\matpower'));
define_constants;

%% Load base case
mpc_base = loadcase('case69');   % Ensure this file exists

%% File names
files.base = 'ieee_69_load_PQ.xlsx';

files.FCS.FPM = 'IEEE69_Load_With_FCS_FPM.xlsx';
files.FCS.DPM = 'IEEE69_Load_With_FCS_DPM.xlsx';
files.FCS.TOU = 'IEEE69_Load_With_FCS_TOU.xlsx';
files.FCS.CPP = 'IEEE69_Load_With_FCS_CPP.xlsx';

files.ALL.FPM = 'IEEE69_Load_FPM.xlsx';
files.ALL.DPM = 'IEEE69_Load_DPM.xlsx';
files.ALL.TOU = 'IEEE69_Load_TOU.xlsx';
files.ALL.CPP = 'IEEE69_Load_CPP.xlsx';

pricing = {'FPM','DPM','TOU','CPP'};

%% Initialize results
minV = struct();
sysLoss = struct();

%% Function handle
run_24h = @(file) runLoadFlow24h(file, mpc_base);

%% ================================
%% BASE CASE
%% ================================
[minV.base, sysLoss.base] = run_24h(files.base);

%% ================================
%% LOOP FOR EACH PRICING METHOD
%% ================================
for k = 1:length(pricing)

    p = pricing{k};

    % Base + FCS
    [minV.FCS.(p), sysLoss.FCS.(p)] = run_24h(files.FCS.(p));

    % Base + FCS + SPV + BESS
    [minV.ALL.(p), sysLoss.ALL.(p)] = run_24h(files.ALL.(p));
end

%% ================================
%% PLOTTING
%% ================================

bus = 1:69;
hours = 1:24;

for k = 1:length(pricing)

    p = pricing{k};

    %% -------- Voltage Plot --------
    figure;
    plot(bus, minV.base, 'k', 'LineWidth', 2); hold on;
    plot(bus, minV.FCS.(p), 'r', 'LineWidth', 2);
    plot(bus, minV.ALL.(p), 'b', 'LineWidth', 2);

    xlabel('Bus Number','FontWeight','bold','FontSize',12);
    ylabel('Minimum Voltage (p.u.)','FontWeight','bold','FontSize',12);
    title(['Minimum Bus Voltage - ', p],'FontWeight','bold','FontSize',14);

    legend('Base','Base + FCS','Base + FCS + SPV + BESS', ...
        'Location','southwest');   % ✅ FIXED

    %grid on;
    set(gca,'FontWeight','bold');

    %% -------- System Loss Plot --------
    figure;
    plot(hours, sysLoss.base, 'k', 'LineWidth', 2); hold on;
    plot(hours, sysLoss.FCS.(p), 'r', 'LineWidth', 2);
    plot(hours, sysLoss.ALL.(p), 'b', 'LineWidth', 2);

    xlabel('Time (Hour)','FontWeight','bold','FontSize',12);
    ylabel('System Loss (kW)','FontWeight','bold','FontSize',12);
    title(['System Loss (24h) - ', p],'FontWeight','bold','FontSize',14);

    legend('Base','Base + FCS','Base + FCS + SPV + BESS', ...
        'Location','southeast');   % ✅ FIXED

    %grid on;
    set(gca,'FontWeight','bold');

end

%% ================================
%% EXPORT RESULTS TO EXCEL
%% ================================

for k = 1:length(pricing)

    p = pricing{k};

    % Voltage data
    T_voltage = table(bus', minV.base, minV.FCS.(p), minV.ALL.(p), ...
        'VariableNames', {'Bus','Base','FCS','FCS_SPV_BESS'});

    writetable(T_voltage, ['Voltage_',p,'.xlsx']);

    % Loss data
    T_loss = table(hours', sysLoss.base', sysLoss.FCS.(p)', sysLoss.ALL.(p)', ...
        'VariableNames', {'Hour','Base','FCS','FCS_SPV_BESS'});

    writetable(T_loss, ['Loss_',p,'.xlsx']);

end

disp('✅ Analysis Completed Successfully');