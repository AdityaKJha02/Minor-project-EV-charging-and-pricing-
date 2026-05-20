clc;
clear;
close all;

% File name
filename = 'Min voltage level.xlsx';

% Get sheet names
[~, sheetNames] = xlsfinfo(filename);

numStations = length(sheetNames);

% Colors
colors = lines(numStations);

figure;
hold on;

for i = 1:4
    % Read data from each sheet
    data = readmatrix(filename, 'Sheet', sheetNames{i});
    
    % Extract data
    hour = data(:,1);
    Min_voltage = data(:,27);
    
    % Custom legend name: FCS1, FCS2, ..., FCS6
    legendName = sheetNames{i};
    
    % Plot
    plot(hour, Min_voltage, '-o', ...
        'Color', colors(i,:), ...
        'LineWidth', 1.8, ...
        'DisplayName', legendName);
end

% Labels and title
xlabel('Bus number');
ylabel('Minimum voltage in PU');
title('Minimun voltage comparison in pricing');

% Show legend
legend('show', 'Location', 'best');

%grid on;
hold off;