clc;
clear;

%% ================= INPUT =================
battery_sizes = [1100, 1300, 1200, 1500]; % [TOU, DPM, CPP, FP]

files = { ...
    'EVFCS_FCS_Summary_TOU_FINAL.xlsx', ...
    'EVFCS_FCS_Summary_DPM_FINAL.xlsx', ...
    'EVFCS_FCS_Summary_CPP_FINAL.xlsx', ...
    'EVFCS_FCS_Summary_fp (2).xlsx'};

sheets = {'FCS_1','FCS_2','FCS_3','FCS_4','FCS_5','FCS_6'};
step_size = 25;   % change to 50, 25, etc.
%% ================= PROCESS =================
for f = 1:length(files)

    file = files{f};
    total_battery = battery_sizes(f);

    energy = zeros(6,1);

    %% ---- Read B2 ----
    for i = 1:6
        val = readmatrix(file, 'Sheet', sheets{i}, 'Range', 'B2');
        if isempty(val) || all(isnan(val))
    val = 0;
else
    val = val(1); % take first value if array
end
        energy(i) = val;
    end

    total_energy = sum(energy);

    if total_energy == 0
        warning('All energy values are zero in %s', file);
        alloc_final = zeros(6,1);
    else
        %% ---- Step 1: proportional ----
        alloc = (energy / total_energy) * total_battery;

        %% ---- Step 2: floor to 100 ----
alloc_floor = floor(alloc / step_size) * step_size;
        %% ---- Step 3: distribute remainder ----
        remainder = total_battery - sum(alloc_floor);

        % fractional parts (priority for distribution)
        frac = alloc - alloc_floor;

        [~, idx] = sort(frac, 'descend');

        alloc_final = alloc_floor;

        i = 1;
        while remainder > 0
            alloc_final(idx(i)) = alloc_final(idx(i)) + step_size;
remainder = remainder - step_size;
            i = mod(i,6) + 1;
        end
    end

    %% ---- Write to Excel ----
    for i = 1:6
        writematrix(alloc_final(i), file, ...
            'Sheet', sheets{i}, 'Range', 'M2');
    end

    fprintf('Processed: %s\n', file);
end

disp('Battery allocation completed!');

%% ================= CREATE SUMMARY FILE =================

summary_matrix = zeros(6,4); % 6 rows (FCS), 4 cols (schemes)

for f = 1:length(files)

    file = files{f};

    for i = 1:6
        val = readmatrix(file, 'Sheet', sheets{i}, 'Range', 'M2');

        if isempty(val) || all(isnan(val))
            val = 0;
        else
            val = val(1);
        end

        summary_matrix(i,f) = val;
    end
end

%% Convert to table
T_summary = array2table(summary_matrix, ...
    'VariableNames', {'TOU','DPM','CPP','FPM'}, ...
    'RowNames', sheets);

%% Write to Excel
output_summary_file = 'Battery_Distribution_Summary.xlsx';

writetable(T_summary, output_summary_file, ...
    'WriteRowNames', true);

disp('Summary file created: Battery_Distribution_Summary.xlsx');
