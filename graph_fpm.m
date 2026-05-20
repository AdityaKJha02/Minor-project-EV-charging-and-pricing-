clc;
clear;
close all;

% File name
filename = 'Hourly_Average_All_Sheets_dpm.xlsx';

% Get sheet names
[~, sheetNames] = xlsfinfo(filename);

numStations = length(sheetNames);

% Colors
colors = lines(numStations);

figure;
hold on;

for i = 1:numStations
    % Read data from each sheet
    data = readmatrix(filename, 'Sheet', sheetNames{i});
    
    % Extract data
    hour = data(:,1);
    avg_power = data(:,2);
    
    % Custom legend name: FCS1, FCS2, ..., FCS6
    legendName = ['FCS' num2str(i)];
    
    % Plot
    plot(hour, avg_power, '-o', ...
        'Color', colors(i,:), ...
        'LineWidth', 1.8, ...
        'DisplayName', legendName);
end

% Labels and title
xlabel('Time (hour)');
ylabel('Average Power (kW)');
title('DPM Power Comparison of FCS');

% Show legend
legend('show', 'Location', 'best');

%grid on;
hold off;