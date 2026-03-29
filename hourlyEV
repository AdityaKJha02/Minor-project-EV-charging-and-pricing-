clc; clear; close all;

%% ================= INPUT FILE =================
filename = 'EVFCS_10min_Arrival_Power_TOU_FINAL.xlsx';

[~, sheetNames] = xlsfinfo(filename);

ncs = 6;          % number of FCS
nhour = 24;
slots_per_hour = 6;

%% ================= STORAGE =================
All_FCS = zeros(nhour, ncs);

%% ================= LOOP THROUGH SHEETS =================
for s = 1:ncs
    
    data = readcell(filename, 'Sheet', sheetNames{s});
    
    % Extract EV column (Column E)
    EV_col = data(2:145, 5);
    
    EV_count_hourly = zeros(nhour,1);
    
    for h = 1:nhour
        
        idx1 = (h-1)*slots_per_hour + 1;
        idx2 = h*slots_per_hour;
        
        slots = EV_col(idx1:idx2);
        EV_list = {};
        
        for i = 1:length(slots)
            
            entry = slots{i};
            
            % Robust empty check
            if isempty(entry) || ...
               (ischar(entry) && isempty(strtrim(entry))) || ...
               (isstring(entry) && strlength(entry)==0)
                continue;
            end
            
            entry = string(entry);
            parts = split(entry, ',');
            
            EV_list = [EV_list; parts];
        end
        
        % Remove empty
        EV_list = EV_list(~cellfun('isempty', EV_list));
        
        % Unique EVs
        unique_EVs = unique(EV_list);
        
        EV_count_hourly(h) = length(unique_EVs);
        
    end
    
    % Store results
    All_FCS(:, s) = EV_count_hourly;
    
end

%% ================= CREATE OUTPUT FILE =================
outputFile = 'EV_Count_All_FCS.xlsx';

Hour = (0:23)';

% Write each FCS sheet
for s = 1:ncs
    T = table(Hour, All_FCS(:,s), ...
        'VariableNames', {'Hour','Number_of_EVs'});
    
    sheetName = ['FCS_', num2str(s)];
    writetable(T, outputFile, 'Sheet', sheetName);
end

%% ================= TOTAL SHEET =================
Total_EV = sum(All_FCS, 2);

T_total = table(Hour, Total_EV, ...
    'VariableNames', {'Hour','Total_EVs'});

writetable(T_total, outputFile, 'Sheet', 'TOTAL');

disp('====================================');
disp('✅ Excel file created with 7 sheets');
disp('6 FCS sheets + 1 TOTAL sheet');
disp(['Saved as: ', outputFile]);
disp('====================================');
