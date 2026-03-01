function generateFCS_Summary( ...
        arrivalPowerFile, timelineFile, sheetNames_EV)

    outputExcel = 'EVFCS_FCS_Summary.xlsx';

    nFCS = numel(sheetNames_EV);

    for i = 1:nFCS
        sheet = sheetNames_EV{i};

        %% ================================
        % FILE 1: 10-min Arrival Power
        % ================================
        Tpow = readtable(arrivalPowerFile, 'Sheet', sheet);

        activePorts = Tpow{:,4};
        portCounts  = zeros(height(Tpow),1);

        for k = 1:height(Tpow)
            val = string(activePorts{k});
            if strlength(val) == 0 || ismissing(val)
                portCounts(k) = 0;
            else
                portCounts(k) = numel(strsplit(val, ','));
            end
        end

        max_ports_used = max(portCounts);

        %% Restrict to first 144 slots
Tpow_144 = Tpow(1:144, :);

power_kW = Tpow_144{:,7};
activePorts_144 = Tpow_144{:,4};

% Recompute port counts for 144 slots only
portCounts = zeros(height(Tpow_144),1);
for k = 1:height(Tpow_144)
    val = string(activePorts_144{k});
    if strlength(val) == 0 || ismissing(val)
        portCounts(k) = 0;
    else
        portCounts(k) = numel(strsplit(val, ','));
    end
end

max_ports_used = max(portCounts);

max_power     = max(power_kW);
avg_power_144 = mean(power_kW, 'omitnan');

        %% ================================
        % FILE 2: EV Timeline
        % ================================
        Ttime = readtable(timelineFile, 'Sheet', sheet);

% Extract arrival column (Column E = 5)
arrival_min = Ttime{:,5};

% Logical index for arrival <= 1430
valid_idx = arrival_min <= 1430;

% Filter table
T_filtered = Ttime(valid_idx, :);

% Now compute statistics only on filtered rows

max_port_index = max(T_filtered{:,4});

avg_wait_min   = mean(T_filtered{:,6}, 'omitnan');
max_wait_min   = max(T_filtered{:,6});

avg_charge_min = mean(T_filtered{:,8}, 'omitnan');
max_charge_min = max(T_filtered{:,8});

nEVs = height(T_filtered);

total_energy_kWh = sum(T_filtered{:,12}, 'omitnan');
avg_energy_kWh   = mean(T_filtered{:,12}, 'omitnan');
        %% ================================
        % SUMMARY TABLE
        % ================================
        Summary = table( ...
            nEVs, ...
            total_energy_kWh, ...
            max_power, ...
            avg_power_144, ...
            max_port_index, ...
            avg_wait_min, ...
            max_wait_min, ...
            avg_charge_min, ...
            max_charge_min, ...
            avg_energy_kWh, ...
            'VariableNames', { ...
            'Num_EVs', ...
            'Total_Energy_Sold_kWh', ...
                'Max_Power_kW', ...
                'Avg_Power_144slots_kW', ...
                'Max_Port_Index', ...
                'Avg_Wait_min', ...
                'Max_Wait_min', ...
                'Avg_Charge_min', ...
                'Max_Charge_min', ...
                'Avg_Energy_kWh'});

        %% ================================
        % WRITE TO ONE EXCEL (MULTI-SHEET)
        % ================================
        writetable(Summary, outputExcel, ...
            'Sheet', sprintf('FCS_%d', i), ...
            'Range', 'A1');

        fprintf('✔ Written summary for FCS %d\n', i);
    end

    fprintf('\n✔ Final output file created: %s\n', outputExcel);
end
