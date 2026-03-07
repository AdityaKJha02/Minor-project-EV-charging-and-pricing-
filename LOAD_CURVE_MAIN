clc; clear; close all;

%% ===============================
% FILES
%% ===============================

file_FPM = 'EVFCS_10min_Arrival_Power_fp (3).xlsx';
file_PED = 'EVFCS_10min_Arrival_Power_DPM_FINAL.xlsx';
file_TOU = 'EVFCS_10min_Arrival_Power_TOU_FINAL.xlsx';
file_CPP = 'EVFCS_10min_Arrival_Power_CPP_FINAL.xlsx';

%% ===============================
% SHEET NAMES
%% ===============================

sheetNames_FPM = {
'EVFCS_1_6_2'
'EVFCS_2_24_21'
'EVFCS_3_15_22'
'EVFCS_4_13_12'
'EVFCS_5_7_18'
'EVFCS_6_10_15'
};

sheetNames_PED = {
'EVFCS_1_2_6'
'EVFCS_2_24_21'
'EVFCS_3_15_22'
'EVFCS_4_12_13'
'EVFCS_5_7_18'
'EVFCS_6_10_15'
};

sheetNames_TOU = {
'EVFCS_1_6_2'
'EVFCS_2_21_24'
'EVFCS_3_15_22'
'EVFCS_4_13_12'
'EVFCS_5_18_7'
'EVFCS_6_10_15'
};

sheetNames_CPP = {
'EVFCS_1_2_6'
'EVFCS_2_24_21'
'EVFCS_3_15_22'
'EVFCS_4_13_12'
'EVFCS_5_7_18'
'EVFCS_6_15_10'
};

nFCS = 6;

time = (0:143)*10/60;   % hours

%% =========================================================
% DELETE OLD FIGURES
%% =========================================================

oldFiles = {
'Figure_FPM.png'
'Figure_PED.png'
'Figure_TOU.png'
'Figure_CPP.png'
'Figure_FPM_vs_PED.png'
'Figure_FPM_vs_TOU.png'
};

for i = 1:length(oldFiles)
    if exist(oldFiles{i},'file')
        delete(oldFiles{i});
    end
end

fprintf('Old images deleted.\n');

%% =========================================================
% FUNCTION TO PLOT ONE PRICING METHOD
%% =========================================================

function plotPricing(file, sheets, time, titleName, saveName)

figure

for i = 1:6
    
    subplot(2,3,i)
    
    M = readmatrix(file,'Sheet',sheets{i});
    
    P = M(2:145,7);
    
    plot(time,P,'LineWidth',2)
    
    grid on
    xlabel('Time (hours)')
    ylabel('Power (kW)')
    
    title(['FCS ' num2str(i)])
    
end

sgtitle(titleName)

saveas(gcf,saveName)

fprintf('Saved %s\n',saveName);

end

%% =========================================================
% GENERATE FIGURES
%% =========================================================

plotPricing(file_FPM,sheetNames_FPM,time,...
'FPM Charging Power Profiles','Figure_FPM.png')

plotPricing(file_PED,sheetNames_PED,time,...
'PED Charging Power Profiles','Figure_PED.png')

plotPricing(file_TOU,sheetNames_TOU,time,...
'TOU Charging Power Profiles','Figure_TOU.png')

plotPricing(file_CPP,sheetNames_CPP,time,...
'CPP Charging Power Profiles','Figure_CPP.png')
