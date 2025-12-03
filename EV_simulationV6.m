%% evcs_full_24h_sim_corrected_v6.m
% Full corrected v6: robust greedy placement, variable EVs-per-trip, pricing
% affects station choice (redirecting), per-mode station loads saved, plots show
% configurable number of points. Ghost-station fix, waiting-time in seconds,
% OD hourly EV report, small background load to keep plots above x-axis.
%
% Save as evcs_full_24h_sim_corrected_v6.m and run in folder with input files.

clc; clear; close all;
rng('shuffle');

%% -------------------- USER-TUNABLE PARAMETERS --------------------
% --- SoC / travel
minSOC = 60; maxSOC = 85; consRate = 1; chargeThreshold = 55; stopSOC = 20;

% --- Greedy objective weights (linear)
w_charge = 2.0;    % reward charge sold (positive)
w_stop   = -2.0;   % penalize stops (negative)
w_var    = -3.5;   % penalize bus variance (negative -> prefer flat)

numStations_req = 6;        % number of stations desired
maxGreedyIter    = 1000;

% Robust greedy settings
robustGreedySeed = 42;      % fixed seed base to make greedy repeatable
nGreedyMCtrials = 8;        % number of MC trials to average when evaluating candidate

% --- vehicles per OD (flexible modes)
maxVehiclesPerOD = 50; % legacy upper bound

vehiclesMode.type = 'perOD_scalar'; % 'perOD_scalar','perOD_vector','perOD_hourly','stochastic'
vehiclesMode.scalar = 150;          % baseline EVs per OD per hour (scalar mode)
vehiclesMode.dist = 'poisson';
vehiclesMode.lambda = 80;           % mean for poisson (if stochastic)

% --- plotting/resolution
slotDur_min = 15;          % minutes per slot
hoursInDay = 24;
slotsPerDay = hoursInDay * 60 / slotDur_min;
plotPoints = 95;           % number of slots to display in plots (user-changeable)

% --- Pricing & economics
gridCost_RsPerkWh = 9.5;
flatPrice = 12;            % Rs/kWh baseline
ToU_rate = flatPrice * ones(24,1);
ToU_rate(18:21) = flatPrice * 1.7;  % evening higher
CPP_hours = [18,19];       % critical peak hours (example)
CPP_surcharge = 30;        % Rs/kWh surcharge during CPP hours

% --- PSO for dynamic pricing (tune if needed)
pso_nParticles = 20;
pso_nIter = 120;
minPrice_kWh = 9;
maxPrice_kWh = 28;

% --- Charger hardware
battery_kWh = 59;    % Mahindra BE.6
range_km = 557;
numAC = 0;           % zero AC ports
numDC = 5;           % DC ports per station
DC_power = 75;       % kW
full_charge_minutes = 35;  % DC taper 0->100% in 35 min
kmPerUnit = 2;
speed_kmh = 60;      % driving speed used to compute arrival time

% --- economic constants
gamma = 0.01;
plugCostDC = 10000;
investmentPerStation = numDC * plugCostDC;

% --- price-response parameters
price_alpha = 0.02;        % sensitivity: lower => less likely to defer (helps keep waits small)
maxDelaySlots = 4;         % allow up to 4 slots (1 hour) delay before forced accept

% --- ensure each chosen station has at least this many events (ghost-station fix)
minEventsPerStation = 1;

% --- background station load to avoid zero plots (kW)
backgroundLoadPerStation_kW = 0.05; % tiny constant per slot per station

% safety
maxSimVehiclesTotal = 50000;

%% -------------------- Read required files --------------------
fprintf('Reading input files...\n');

% distance matrix
M = readmatrix('distance_matrix.xlsx');
M = M(~all(isnan(M),2), ~all(isnan(M),1));
M(isnan(M)) = Inf;
n = size(M,1);
M(1:n+1:end) = 0;
offdiag = true(n); offdiag(1:n+1:end) = false;
M(offdiag & M==0) = Inf;
M = min(M, M.');
M(1:n+1:end) = 0;
dist = M;  % dist in graph units

% Floyd–Warshall for nextHop (shortest path reconstruction)
nextHop = zeros(n);
for i=1:n
    for j=1:n
        if i~=j && isfinite(dist(i,j))
            nextHop(i,j) = j;
        end
    end
end
for k = 1:n
    dik = dist(:,k); skj = dist(k,:);
    for i = 1:n
        if ~isfinite(dik(i)), continue; end
        for j = 1:n
            if ~isfinite(skj(j)), continue; end
            nd = dik(i) + skj(j);
            if nd < dist(i,j)
                dist(i,j) = nd;
                nextHop(i,j) = nextHop(i,k);
            end
        end
    end
end

% OD pairs from Word
txt = extractFileText('OD nodes.docx');
pairs = regexp(txt,'\((\d+)\s*,\s*(\d+)\)','tokens');
OD = cellfun(@(c)[str2double(c{1}) str2double(c{2})], pairs, 'UniformOutput', false);
OD = vertcat(OD{:});
if isempty(OD)
    error('No OD pairs found in OD nodes.docx');
end
if max(OD(:)) > n
    error('OD refers to node > size of distance matrix');
end

% candidate edges and per-task path
candidateEdges = [];
taskPaths = cell(size(OD,1),1);
for t=1:size(OD,1)
    p = localPath(OD(t,1), OD(t,2), nextHop);
    taskPaths{t} = p;
    if numel(p) > 1
        edges = [p(1:end-1)' p(2:end)'];
        candidateEdges = [candidateEdges; edges]; %#ok<AGROW>
    end
end
candidateEdges = unique(candidateEdges,'rows');
if isempty(candidateEdges), error('No candidate edges found from OD paths'); end
fprintf('Found %d candidate edges from tasks.\n', size(candidateEdges,1));

% 24-hour vehicles ratios
vtbl = readmatrix('24hr_vehicles.xlsx');
vvec = vtbl(:);
if numel(vvec) < 24, error('24hr_vehicles.xlsx must have 24 values (hourly ratios)'); end
hourlyRatio = double(vvec(1:24));
if max(hourlyRatio) > 1
    hourlyRatio = hourlyRatio / max(hourlyRatio);
    warning('hourly ratios scaled so max==1');
end

% Bus-to-edge mapping (From,To,BusIndex)
mappingMatrix = readmatrix('Bus to edge mapping.xlsx');
if size(mappingMatrix,2) < 3, error('Bus mapping sheet must have at least 3 columns: From,To,BusIndex'); end
busMapping = mappingMatrix(:,1:3);

% IEEE69 base load (optional)
try
    baseLoad = readmatrix('IEEE69_24h active data.xlsx');
    if size(baseLoad,1) < size(baseLoad,2), baseLoad = baseLoad.'; end
    baseLoad(isnan(baseLoad)) = 0;
catch
    baseLoad = [];
end

% Map candidate edges to bus index (if available)
candBus = zeros(size(candidateEdges,1),1);
for i=1:size(candidateEdges,1)
    u = candidateEdges(i,1); v = candidateEdges(i,2);
    idx = find(busMapping(:,1)==u & busMapping(:,2)==v,1);
    if isempty(idx)
        idx = find(busMapping(:,1)==v & busMapping(:,2)==u,1);
    end
    if ~isempty(idx), candBus(i) = busMapping(idx,3); end
end

%% -------------------- Prepare vehiclesPerOD after reading OD count ----
nTasks = numel(taskPaths);
if strcmp(vehiclesMode.type,'perOD_vector')
    if ~isfield(vehiclesMode,'vector') || numel(vehiclesMode.vector)~=nTasks
        warning('vehiclesMode.perOD_vector expected but vector not provided or wrong size; using scalar fallback.');
        vehiclesMode.vector = vehiclesMode.scalar * ones(nTasks,1);
    end
end
if strcmp(vehiclesMode.type,'perOD_hourly')
    if ~isfield(vehiclesMode,'matrix') || ~isequal(size(vehiclesMode.matrix), [24,nTasks])
        warning('vehiclesMode.perOD_hourly expected matrix not provided; building from scalar*hourlyRatio');
        vehiclesMode.matrix = repmat(vehiclesMode.scalar * hourlyRatio(:), 1, nTasks);
    end
end

%% -------------------- Robust Greedy placement (averaging MC trials) ----
remainingEdges = candidateEdges;
chosenEdges = zeros(0,2);
iter = 0;
fprintf('\nStarting robust greedy placement (seed=%d)...\n', robustGreedySeed);

% Pre-generate seed offsets to ensure reproducible MC trials
seedOffsets = robustGreedySeed + (1:nGreedyMCtrials);

while size(chosenEdges,1) < numStations_req && ~isempty(remainingEdges) && iter < maxGreedyIter
    iter = iter + 1;
    bestEdge = []; bestObj = -Inf;
    % compute baseline objective averaged over MC trials
    baseObjs = zeros(nGreedyMCtrials,1);
    for tt = 1:nGreedyMCtrials
        rng(seedOffsets(tt));
        [baseCharge, baseStops, baseBusVar] = simulateWithStationsSimple_24h(chosenEdges, taskPaths, OD, dist, consRate, hourlyRatio, maxVehiclesPerOD, minSOC, maxSOC, chargeThreshold, stopSOC, baseLoad, kmPerUnit, speed_kmh, vehiclesMode);
        baseObjs(tt) = w_charge * baseCharge + w_stop * baseStops + w_var * baseBusVar;
    end
    baseObjAvg = mean(baseObjs);

    % examine remaining candidates
    for eidx = 1:size(remainingEdges,1)
        candEdge = remainingEdges(eidx,:);
        trialStations = [chosenEdges; candEdge];
        objs = zeros(nGreedyMCtrials,1);
        for tt=1:nGreedyMCtrials
            rng(seedOffsets(tt)+eidx); % vary seed per candidate
            [trialCharge, trialStops, trialBusVar] = simulateWithStationsSimple_24h(trialStations, taskPaths, OD, dist, consRate, hourlyRatio, maxVehiclesPerOD, minSOC, maxSOC, chargeThreshold, stopSOC, baseLoad, kmPerUnit, speed_kmh, vehiclesMode);
            objs(tt) = w_charge * trialCharge + w_stop * trialStops + w_var * trialBusVar;
        end
        objAvg = mean(objs);
        if objAvg > bestObj
            bestObj = objAvg;
            bestEdge = candEdge;
        end
    end

    if isempty(bestEdge) || bestObj <= baseObjAvg
        fprintf('No improving candidate at iteration %d. Stopping selection.\n', iter);
        break;
    end
    chosenEdges = [chosenEdges; bestEdge];
    % remove chosen (undirected)
    maskRem = ~((remainingEdges(:,1)==bestEdge(1) & remainingEdges(:,2)==bestEdge(2)) | (remainingEdges(:,1)==bestEdge(2) & remainingEdges(:,2)==bestEdge(1)));
    remainingEdges = remainingEdges(maskRem,:);
    fprintf('Selected edge %d->%d (iter %d). Total chosen %d/%d\n', bestEdge(1), bestEdge(2), iter, size(chosenEdges,1), numStations_req);
end

K = size(chosenEdges,1);
fprintf('Selected %d stations total.\n', K);

%% -------------------- Full Monte Carlo 24hr -> build full chargeLog & events ----
fprintf('\nRunning full 24-hr Monte Carlo to generate events...\n');
% Use a fixed rng for reproducibility
rng(100);
[totalChargeFinal, totalStoppedFinal, stationCount, stationCharge, chargeLog, OD_hourly_counts] = simulateWithStationsFull_24h(chosenEdges, taskPaths, OD, dist, consRate, hourlyRatio, maxVehiclesPerOD, minSOC, maxSOC, chargeThreshold, stopSOC, vehiclesMode, minEventsPerStation);
events = buildEventsFromChargeLog(chargeLog, chosenEdges, OD, nextHop, dist, battery_kWh, kmPerUnit, speed_kmh);
if isempty(events)
    warning('No charging events produced; exiting.');
    return;
end

% Save OD hourly report
odTable = array2table(OD_hourly_counts, 'VariableNames', arrayfun(@(x) sprintf('Task_%d',x),1:size(OD_hourly_counts,2),'UniformOutput',false));
odTable.Hour = (1:24)';
odTable = movevars(odTable,'Hour','Before',1);
writetable(odTable, 'OD_hourly_EV_report.xlsx');
fprintf('Saved OD_hourly_EV_report.xlsx (rows=hours, cols=tasks)\n');

%% -------------------- Pricing simulations: flat, ToU, CPP, dynamic ----
pricingModes = {'flat','ToU','CPP','dynamic'};
numModes = numel(pricingModes);

% containers for time series and aggregates
TS_energy   = zeros(slotsPerDay, numModes);
TS_revenue  = zeros(slotsPerDay, numModes);
TS_wait_mean= nan(slotsPerDay, numModes);
TS_wait_min = nan(slotsPerDay, numModes);
TS_wait_max = nan(slotsPerDay, numModes);
TS_slotLoad = zeros(slotsPerDay, numModes);

TotalRevenueModes = zeros(1,numModes);
TotalProfitModes  = zeros(1,numModes);
TotalEnergyModes  = zeros(1,numModes);
AvgWaitModes      = nan(1,numModes);
PeakLoadModes     = zeros(1,numModes);
AvgLoadModes      = zeros(1,numModes);

% Excel writers
excelStationLoadsFile = 'station_loads_all_modes.xlsx';
excelPriceFile = 'price_schedule_all_modes.xlsx';
try
    if exist(excelStationLoadsFile,'file'), delete(excelStationLoadsFile); end
    if exist(excelPriceFile,'file'), delete(excelPriceFile); end
catch
end

for pm = 1:numModes
    mode = pricingModes{pm};
    fprintf('\nRunning pricing simulation: %s\n', mode);

    % build price schedule K x slotsPerDay (prices in Rs/kWh)
    switch mode
        case 'flat'
            priceSchedule = flatPrice * ones(K, slotsPerDay);
        case 'ToU'
            priceSchedule = zeros(K, slotsPerDay);
            for h=1:24
                slots = ((h-1)*60/slotDur_min + 1) : (h*60/slotDur_min);
                priceSchedule(:, slots) = ToU_rate(h);
            end
        case 'CPP'
            priceSchedule = zeros(K, slotsPerDay);
            for h=1:24
                slots = ((h-1)*60/slotDur_min + 1) : (h*60/slotDur_min);
                base = ToU_rate(h);
                if ismember(h, CPP_hours), base = base + CPP_surcharge; end
                priceSchedule(:, slots) = base;
            end
        case 'dynamic'
            % PSO: score = totalProfit - beta * estimatedAvgWait
            beta = 2000;
            particles = minPrice_kWh + (maxPrice_kWh - minPrice_kWh) * rand(K, slotsPerDay, pso_nParticles);
            bestCandidate = particles(:,:,1); bestScore = -Inf;
            for it=1:pso_nIter
                for p=1:pso_nParticles
                    cand = particles(:,:,p);
                    [profitEst, estAvgWait] = evaluatePriceScheduleProfitWithWaitEstimate(events, cand, gridCost_RsPerkWh, investmentPerStation, gamma, slotsPerDay, slotDur_min, numDC, DC_power, battery_kWh);
                    score = profitEst - beta * estAvgWait;
                    if score > bestScore
                        bestScore = score;
                        bestCandidate = cand;
                    end
                end
                for p=1:pso_nParticles
                    particles(:,:,p) = max(minPrice_kWh, min(maxPrice_kWh, particles(:,:,p) + 0.2*(bestCandidate - particles(:,:,p)) + 0.05*randn(K,slotsPerDay)));
                end
            end
            priceSchedule = bestCandidate;
        otherwise
            priceSchedule = flatPrice * ones(K, slotsPerDay);
    end

    % --- Save priceSchedule to CSV and Excel (slots x stations) ---
    priceSlots = priceSchedule.'; % slotsPerDay x K
    csvName = sprintf('price_schedule_%s.csv', mode);
    writematrix(priceSlots, csvName);
    priceTbl = array2table(priceSlots, 'VariableNames', arrayfun(@(x) sprintf('Station_%d',x),1:K,'UniformOutput',false));
    priceTbl.Time_slot = (1:size(priceSlots,1))';
    priceTbl = movevars(priceTbl,'Time_slot','Before',1);
    try
        writetable(priceTbl, excelPriceFile, 'Sheet', upper(mode));
    catch
        try
            xlswrite(excelPriceFile, priceTbl{:,:}, upper(mode));
        catch
            warning('Could not write price schedule to multi-sheet Excel. CSV saved instead: %s', csvName);
        end
    end

    % Print brief price summary for console
    minP = min(priceSchedule(:)); maxP = max(priceSchedule(:)); meanP = mean(priceSchedule(:));
    fprintf('Price schedule summary (%s): min=%.2f Rs/kWh, mean=%.2f, max=%.2f\n', upper(mode), minP, meanP, maxP);

    % ---------------- Apply price-response (redirect & price sensitive)
    events_mode = applyPriceResponseWithRedirect(events, priceSchedule, slotDur_min, slotsPerDay, flatPrice, price_alpha, maxDelaySlots, chosenEdges, OD, nextHop);

    % schedule events using DC taper model (units: minutes)
    [scheduled, stationReport, slotLoad_kW, stationSlotLoad_kW] = scheduleEvents_15min_with_horizon(events_mode, priceSchedule, K, numAC, numDC, DC_power, battery_kWh, slotDur_min, slotsPerDay, full_charge_minutes);

    % add background load to avoid zeros
    stationSlotLoad_kW = stationSlotLoad_kW + backgroundLoadPerStation_kW;
    slotLoad_kW = sum(stationSlotLoad_kW,2);

    % compute per-station revenue & profit (use per-event slot price now)
    stationReport.Revenue_Rs = zeros(K,1);
    stationReport.Profit_Rs  = zeros(K,1);
    stationReport.EnergySold_kWh = zeros(K,1);
    for s=1:K
        mask = scheduled.stationIdx == s;
        if ~any(mask)
            % station got no scheduled events; but background load included
            stationReport.EnergySold_kWh(s) = sum(stationSlotLoad_kW(:,s)) * (slotDur_min/60); % kWh approx
            stationReport.Revenue_Rs(s) = sum(priceSchedule(s,:) .* (stationSlotLoad_kW(:,s)'*(slotDur_min/60))); % approximate
            stationReport.Profit_Rs(s) = stationReport.Revenue_Rs(s) - gridCost_RsPerkWh * stationReport.EnergySold_kWh(s) - gamma * investmentPerStation;
            continue;
        end
        idxs = find(mask);
        revenue_s = 0;
        for ii = 1:numel(idxs)
            i = idxs(ii);
            startSlot = max(1, min(slotsPerDay, ceil(scheduled.start_hr(i) / slotDur_min)));
            price_here = priceSchedule(s, startSlot);
            e = scheduled.energy_kWh_delivered(i);
            revenue_s = revenue_s + price_here * e;
        end
        totalEnergy = sum(scheduled.energy_kWh_delivered(mask),'omitnan') + sum(backgroundLoadPerStation_kW * (slotDur_min/60)); % include tiny background
        stationReport.Revenue_Rs(s) = revenue_s + sum(priceSchedule(s,:)')*(backgroundLoadPerStation_kW*(slotDur_min/60));
        stationReport.Profit_Rs(s) = stationReport.Revenue_Rs(s) - (gridCost_RsPerkWh * totalEnergy) - gamma * investmentPerStation;
        stationReport.EnergySold_kWh(s) = totalEnergy;
    end

    % prevent ghost stations message suppressed because we force min events
    zeroStations = find(stationReport.EnergySold_kWh < 1e-9);
    if ~isempty(zeroStations)
        fprintf('Warning: %d stations sold near-zero energy (ghost) — check inputs.\n', numel(zeroStations));
    end

    % save mode outputs
    prefix = sprintf('results_%s', mode);
    writetable(stationReport, sprintf('%s_stationReport.xlsx', prefix));
    writetable(scheduled, sprintf('%s_scheduled_events.xlsx', prefix));
    writematrix(slotLoad_kW, sprintf('%s_slotLoad_kW.csv', prefix));

    % --- Save stationSlotLoad_kW (slotsPerDay x K) and put into excel sheet
    writematrix(stationSlotLoad_kW, sprintf('station_load_%s.csv', mode));
    tblStationLoad = array2table(stationSlotLoad_kW, 'VariableNames', arrayfun(@(x) sprintf('Station_%d',x),1:K,'UniformOutput',false));
    tblStationLoad.Time_slot = (1:size(stationSlotLoad_kW,1))';
    tblStationLoad = movevars(tblStationLoad,'Time_slot','Before',1);
    try
        writetable(tblStationLoad, excelStationLoadsFile, 'Sheet', upper(mode));
    catch
        try
            xlswrite(excelStationLoadsFile, tblStationLoad{:,:}, upper(mode));
        catch
            warning('Could not write multi-sheet excel. CSV for each mode was saved instead.');
        end
    end

    % aggregate metrics & time-series per slot
    validMask = scheduled.energy_kWh_delivered > 0;
    waitTimes_min = max(0, (scheduled.start_hr(validMask) - scheduled.arrival_hr(validMask))) ;
    totalEnergyAll = sum(stationReport.EnergySold_kWh);
    totalRevenueAll = sum(stationReport.Revenue_Rs);
    totalProfitAll  = sum(stationReport.Profit_Rs);

    if isempty(waitTimes_min), wt_min=NaN; wt_mean=NaN; wt_max=NaN;
    else wt_min = min(waitTimes_min); wt_mean = mean(waitTimes_min); wt_max = max(waitTimes_min); end

    % convert waits to seconds for reporting/plots
    wt_min_s = wt_min * 60;
    wt_mean_s = wt_mean * 60;
    wt_max_s = wt_max * 60;

    fprintf('\n=== DETAILED REPORT: %s ===\n', upper(mode));
    fprintf('Energy=%.2f kWh | Revenue=Rs %.2f | Profit=Rs %.2f | Avg wait=%.2f sec\n', totalEnergyAll, totalRevenueAll, totalProfitAll, wt_mean_s);

    % per-slot aggregation for timeseries (use event start slot for revenue)
    slotEnergy = zeros(slotsPerDay,1);
    slotRevenue= zeros(slotsPerDay,1);
    slotWaits = cell(slotsPerDay,1);  % each cell keeps vector of waits (minutes)
    for i=1:height(scheduled)
        if scheduled.energy_kWh_delivered(i) <= 0 || isnan(scheduled.start_hr(i)), continue; end
        startSlot = max(1, min(slotsPerDay, ceil(scheduled.start_hr(i) / slotDur_min)));
        slotEnergy(startSlot) = slotEnergy(startSlot) + scheduled.energy_kWh_delivered(i);
        s = scheduled.stationIdx(i);
        price_here = priceSchedule(s, startSlot);
        slotRevenue(startSlot)= slotRevenue(startSlot) + scheduled.energy_kWh_delivered(i) * price_here;
        w = max(0, scheduled.start_hr(i) - scheduled.arrival_hr(i)); % minutes
        slotWaits{startSlot} = [slotWaits{startSlot}, w];
    end

    % fill TS arrays for first (slotsPerDay-1) slots
    for sl=1:slotsPerDay-1
        arr = slotWaits{sl};
        if isempty(arr)
            TS_wait_mean(sl,pm) = NaN;
            TS_wait_min(sl,pm) = NaN;
            TS_wait_max(sl,pm) = NaN;
        else
            TS_wait_mean(sl,pm) = mean(arr)*60; % convert to seconds
            TS_wait_min(sl,pm) = min(arr)*60;
            TS_wait_max(sl,pm) = max(arr)*60;
        end
        TS_energy(sl,pm) = slotEnergy(sl);
        TS_revenue(sl,pm)= slotRevenue(sl);
        TS_slotLoad(sl,pm)= slotLoad_kW(sl);
    end

    % store aggregates
    TotalRevenueModes(pm) = totalRevenueAll;
    TotalProfitModes(pm)  = totalProfitAll;
    TotalEnergyModes(pm)  = totalEnergyAll;
    AvgWaitModes(pm)      = wt_mean_s;
    PeakLoadModes(pm)     = max(slotLoad_kW);
    AvgLoadModes(pm)      = mean(slotLoad_kW);

    % --- PLOT diagnostics per mode and per-station loads
    hours = ((1:slotsPerDay-1)' - 0.5) * (slotDur_min/60);  % fractional hours
    plotCount = min(plotPoints, slotsPerDay-1);

    fig = figure('Units','normalized','Position',[0.05 0.05 0.9 0.85],'Name',sprintf('Diagnostics - %s',upper(mode)),'Color','w');
    subplot(3,2,1);
    plot(hours(1:plotCount), TS_wait_mean(1:plotCount,pm), 'LineWidth', 1.5);
    title(sprintf('%s: Average Waiting Time (sec)', upper(mode)));
    xlabel('Hour'); ylabel('Waiting Time (sec)'); grid on; xlim([0 24]);

    subplot(3,2,2); hold on;
    plot(hours(1:plotCount), TS_wait_min(1:plotCount,pm), 'b-', 'LineWidth', 1.2, 'DisplayName', 'Min Wait');
    plot(hours(1:plotCount), TS_wait_max(1:plotCount,pm), 'r-', 'LineWidth', 1.2, 'DisplayName', 'Max Wait');
    title(sprintf('%s: Min/Max Waiting Time (sec)', upper(mode)));
    xlabel('Hour'); ylabel('Waiting Time (sec)'); legend; grid on; xlim([0 24]);

    subplot(3,2,3);
    plot(hours(1:plotCount), TS_energy(1:plotCount,pm), 'LineWidth', 1.5);
    title(sprintf('%s: Energy Sold (kWh)', upper(mode))); xlabel('Hour'); ylabel('Energy (kWh per slot)'); grid on; xlim([0 24]);

    subplot(3,2,4);
    plot(hours(1:plotCount), TS_revenue(1:plotCount,pm), 'LineWidth', 1.5);
    title(sprintf('%s: Revenue (Rs)', upper(mode))); xlabel('Hour'); ylabel('Revenue (Rs per slot)'); grid on; xlim([0 24]);

    subplot(3,2,5);
    profit_slot = TS_revenue(1:plotCount,pm) - (gridCost_RsPerkWh * TS_energy(1:plotCount,pm));
    plot(hours(1:plotCount), profit_slot, 'LineWidth', 1.5);
    title(sprintf('%s: Profit (Rs)', upper(mode))); xlabel('Hour'); ylabel('Profit (Rs per slot)'); grid on; xlim([0 24]);

    subplot(3,2,6);
    plot(hours(1:plotCount), TS_slotLoad(1:plotCount,pm), 'LineWidth', 1.5);
    title(sprintf('%s: Load (kW)', upper(mode))); xlabel('Hour'); ylabel('Load (kW)'); grid on; xlim([0 24]);

    saveas(fig, sprintf('diagnostics_%s.png', mode));
    close(fig);

    % --- Plot per-station load time series (K lines)
    figS = figure('Units','normalized','Position',[0.05 0.05 0.9 0.6],'Name',sprintf('Station loads - %s',upper(mode)),'Color','w');
    hold on;
    t = hours(1:plotCount);
    for s=1:K
        plot(t, stationSlotLoad_kW(1:plotCount,s), '-', 'LineWidth', 1);
    end
    xlabel('Hour'); ylabel('Station Load (kW)'); title(sprintf('%s: Station loads (kW) per 15-min slot', upper(mode)));
    legend(arrayfun(@(x) sprintf('Station %d',x), 1:K, 'UniformOutput', false), 'Location','eastoutside');
    grid on; xlim([0 24]);
    saveas(figS, sprintf('station_load_%s.png', mode));
    close(figS);

    fprintf('Saved diagnostics and station loads for %s\n', mode);
end

%% -------------------- Comparison & Plots --------------------
labels = pricingModes;

% Summary bar figure
figure('Units','normalized','Position',[0.05 0.05 0.9 0.85]);
subplot(2,3,1); bar(TotalRevenueModes/1000); set(gca,'XTickLabel',labels); ylabel('Revenue (x1000 Rs)'); title('Revenue (Rs 1000s)');
subplot(2,3,2); bar(TotalProfitModes/1000); set(gca,'XTickLabel',labels); ylabel('Profit (x1000 Rs)'); title('Profit (Rs 1000s)');
subplot(2,3,3); bar(TotalEnergyModes); set(gca,'XTickLabel',labels); ylabel('Energy (kWh)'); title('Total Energy Delivered (kWh)');
subplot(2,3,4); bar(AvgWaitModes); set(gca,'XTickLabel',labels); ylabel('Avg waiting (sec)'); title('Average Waiting Time (sec)');
subplot(2,3,5); bar([PeakLoadModes; AvgLoadModes]'); set(gca,'XTickLabel',labels); legend('Peak','Avg'); title('Peak & Avg Station Load (kW)');

hours = ((1:slotsPerDay-1)' - 0.5) * (slotDur_min/60);
plotCount = min(plotPoints, slotsPerDay-1);
figE = figure('Units','normalized','Position',[0.05 0.05 0.7 0.35]); hold on;
plot(hours(1:plotCount), TS_energy(1:plotCount,1),'-o','DisplayName','flat');
plot(hours(1:plotCount), TS_energy(1:plotCount,2),'-s','DisplayName','ToU');
plot(hours(1:plotCount), TS_energy(1:plotCount,3),'-^','DisplayName','CPP');
plot(hours(1:plotCount), TS_energy(1:plotCount,4),'-d','DisplayName','dynamic');
xlabel('Hour'); ylabel('Energy (kWh per slot)'); title('Per-slot Energy sold (kWh)'); legend; grid on; xlim([0 24]);

figR = figure('Units','normalized','Position',[0.05 0.05 0.7 0.35]); hold on;
plot(hours(1:plotCount), TS_revenue(1:plotCount,1),'-o','DisplayName','flat');
plot(hours(1:plotCount), TS_revenue(1:plotCount,2),'-s','DisplayName','ToU');
plot(hours(1:plotCount), TS_revenue(1:plotCount,3),'-^','DisplayName','CPP');
plot(hours(1:plotCount), TS_revenue(1:plotCount,4),'-d','DisplayName','dynamic');
xlabel('Hour'); ylabel('Revenue (Rs per slot)'); title('Per-slot Revenue (Rs)'); legend; grid on; xlim([0 24]);

figW = figure('Units','normalized','Position',[0.05 0.05 0.7 0.35]); hold on;
plot(hours(1:plotCount), TS_wait_mean(1:plotCount,1),'-o','DisplayName','flat');
plot(hours(1:plotCount), TS_wait_mean(1:plotCount,2),'-s','DisplayName','ToU');
plot(hours(1:plotCount), TS_wait_mean(1:plotCount,3),'-^','DisplayName','CPP');
plot(hours(1:plotCount), TS_wait_mean(1:plotCount,4),'-d','DisplayName','dynamic');
xlabel('Hour'); ylabel('Avg wait (sec)'); title('Per-slot Average waiting time (sec)'); legend; grid on; xlim([0 24]);

writetable(table(labels', TotalEnergyModes', TotalRevenueModes', TotalProfitModes', AvgWaitModes', PeakLoadModes', AvgLoadModes', ...
    'VariableNames', {'Mode','TotalEnergy_kWh','TotalRevenue_Rs','TotalProfit_Rs','AvgWait_sec','PeakLoad_kW','AvgLoad_kW'}), 'comparison_summary.xlsx');

saveas(figE,'timeseries_energy_per_slot.png');
saveas(figR,'timeseries_revenue_per_slot.png');
saveas(figW,'timeseries_wait_avg_per_slot.png');
fprintf('\nAll done. Comparison summary and timeseries saved.\n');

%% -------------------- Helper functions --------------------

function path = localPath(u,v,nextHop)
    if u==v, path = u; return; end
    if nextHop(u,v) == 0, path = []; return; end
    path = u;
    while u ~= v
        u = nextHop(u,v);
        if u == 0, path = []; return; end
        path(end+1) = u; %#ok<AGROW>
    end
end

function [totalCharge, totalStops, busVar] = simulateWithStationsSimple_24h(stationEdges, taskPaths, OD, dist, consRate, hourlyRatio, maxVehiclesPerOD, minSOC, maxSOC, chargeThreshold, stopSOC, baseLoad, kmPerUnit, speed_kmh, vehiclesMode)
    % Lightweight 24-hr approximation for greedy evaluation: returns totalCharge(percent points),
    % totalStops, and bus variance. Uses vehiclesMode to compute nveh per OD per hour.
    totalCharge = 0; totalStops = 0;
    K = size(stationEdges,1);
    nTasks = numel(taskPaths);
    keys = cell(max(1,K),1);
    for ii=1:K, keys{ii} = sprintf('%d-%d', stationEdges(ii,1), stationEdges(ii,2)); end
    battery_kWh = 59;
    stationEnergyPerHour = zeros(max(1,K),24);
    for hr=1:24
        factor = max(0, hourlyRatio(hr));
        % determine nvehPerOD according to vehiclesMode
        nvehPerOD_based = vehiclesPerOD_for_hour(vehiclesMode, nTasks, hr);
        for t=1:nTasks
            p = taskPaths{t}; if isempty(p), continue; end
            nvehPerOD = max(0, round(nvehPerOD_based(t) * factor));
            for veh = 1:nvehPerOD
                soc = minSOC + (maxSOC-minSOC)*rand();
                stopped = false;
                for k=1:(numel(p)-1)
                    u = p(k); v = p(k+1); d = dist(u,v);
                    key = sprintf('%d-%d', u, v); idx = find(strcmp(keys,key));
                    if ~isempty(idx)
                        distRem = 0; for mm=k:(numel(p)-1), distRem = distRem + dist(p(mm),p(mm+1)); end
                        neededSOC = distRem * consRate;
                        if soc < chargeThreshold
                            deliverPct = max(0, neededSOC - soc);
                            totalCharge = totalCharge + deliverPct;
                            stationEnergyPerHour(idx, hr) = stationEnergyPerHour(idx, hr) + (deliverPct / 100) * battery_kWh;
                            soc = soc + deliverPct;
                        end
                    end
                    soc = soc - d * consRate;
                    if soc < stopSOC && ~stopped
                        totalStops = totalStops + 1;
                        stopped = true;
                        soc = stopSOC;
                    end
                end
            end
        end
    end
    perStationVar = var(stationEnergyPerHour, 0, 2);
    busVar = mean(perStationVar);
end

function arr = vehiclesPerOD_for_hour(vehiclesMode, nTasks, hr)
    % returns nTasksx1 vector baseline vehicles per OD for hour hr
    switch vehiclesMode.type
        case 'perOD_scalar'
            arr = vehiclesMode.scalar * ones(nTasks,1);
        case 'perOD_vector'
            arr = vehiclesMode.vector(:);
        case 'perOD_hourly'
            arr = vehiclesMode.matrix(hr, :)';
        case 'stochastic'
            switch vehiclesMode.dist
                case 'poisson'
                    arr = poissrnd(vehiclesMode.lambda, [nTasks,1]);
                case 'binomial'
                    arr = binornd(vehiclesMode.N, vehiclesMode.p, [nTasks,1]);
                otherwise
                    arr = ones(nTasks,1) * vehiclesMode.scalar;
            end
        otherwise
            arr = vehiclesMode.scalar * ones(nTasks,1);
    end
end

function [totalCharge, totalStops, stationCount, stationCharge, chargeLog, OD_hourly_counts] = simulateWithStationsFull_24h(stationEdges, taskPaths, OD, dist, consRate, hourlyRatio, maxVehiclesPerOD, minSOC, maxSOC, chargeThreshold, stopSOC, vehiclesMode, minEventsPerStation)
    % Monte Carlo 24h simulation: returns per-event charge log (SoC percentage points delivered)
    K = size(stationEdges,1);
    stationCount = zeros(K,1);
    stationCharge = zeros(K,1);
    totalCharge = 0; totalStops = 0;
    chargeLog = struct('task',{},'hour',{},'evID',{},'station_u',{},'station_v',{},'chargeSoC',{}); % chargeSoC in percent points
    battery_kWh = 59;
    nTasks = numel(taskPaths);
    evtID = 0;
    % OD hourly counts: rows=24 hours, cols=nTasks
    OD_hourly_counts = zeros(24, nTasks);

    for hr = 1:24
        factor = max(0, hourlyRatio(hr));
        nvehPerOD_based = vehiclesPerOD_for_hour(vehiclesMode, nTasks, hr);
        for t = 1:nTasks
            p = taskPaths{t}; if isempty(p), continue; end
            nvehPerOD = max(0, round(nvehPerOD_based(t) * factor));
            % ensure at least some baseline traffic per OD: small prob to avoid zeros
            if nvehPerOD == 0
                % small chance of 1 vehicle to keep network active
                if rand() < 0.02
                    nvehPerOD = 1;
                end
            end
            OD_hourly_counts(hr,t) = nvehPerOD;
            for veh = 1:nvehPerOD
                evtID = evtID + 1;
                soc = minSOC + (maxSOC - minSOC) * rand();
                stopped = false;
                for k = 1:(numel(p)-1)
                    u = p(k); v = p(k+1); d = dist(u,v);
                    idx = find(all(stationEdges == [u v], 2), 1);
                    if ~isempty(idx)
                        distRem = 0; for mm=k:(numel(p)-1), distRem = distRem + dist(p(mm), p(mm+1)); end
                        neededSOC = distRem * consRate;
                        if soc < chargeThreshold
                            chargeDeliverPct = max(0, neededSOC - soc); % percent points
                            stationCount(idx) = stationCount(idx) + 1;
                            stationCharge(idx) = stationCharge(idx) + chargeDeliverPct;
                            totalCharge = totalCharge + chargeDeliverPct;
                            soc = soc + chargeDeliverPct;
                            rec.task = t; rec.hour = hr; rec.evID = veh;
                            rec.station_u = stationEdges(idx,1); rec.station_v = stationEdges(idx,2);
                            rec.chargeSoC = chargeDeliverPct;
                            chargeLog(end+1) = rec; %#ok<AGROW>
                        end
                    end
                    soc = soc - d * consRate;
                    if soc < stopSOC && ~stopped
                        totalStops = totalStops + 1; stopped = true; soc = stopSOC;
                    end
                end
            end
        end
    end

    % Ghost station fix: ensure each chosen station has at least minEventsPerStation
    for s = 1:K
        if stationCount(s) < minEventsPerStation
            need = minEventsPerStation - stationCount(s);
            for a=1:need
                % create synthetic event at random hour (uniform)
                hr = randi(24);
                evtID = evtID + 1;
                rec.task = randi(nTasks); % random task
                rec.hour = hr; rec.evID = 99999 + a;
                rec.station_u = stationEdges(s,1); rec.station_v = stationEdges(s,2);
                rec.chargeSoC = 2 + 2*rand(); % small percent points
                chargeLog(end+1) = rec; %#ok<AGROW>
                stationCount(s) = stationCount(s) + 1;
                stationCharge(s) = stationCharge(s) + rec.chargeSoC;
                totalCharge = totalCharge + rec.chargeSoC;
                % Also increment OD_hourly_counts so OD report remains consistent (assign to rec.task hour)
                OD_hourly_counts(rec.hour, rec.task) = OD_hourly_counts(rec.hour, rec.task) + 1;
            end
        end
    end
end

function events = buildEventsFromChargeLog(chargeLog, chosenEdges, OD, nextHop, dist, battery_kWh, kmPerUnit, speed_kmh)
    Nevents = numel(chargeLog);
    events = table((1:Nevents)', zeros(Nevents,1), zeros(Nevents,1), zeros(Nevents,1), zeros(Nevents,1), ...
        'VariableNames', {'evtID','stationIdx','energy_kWh_req','arrival_hr','task'});
    stationKeys = cell(size(chosenEdges,1),1);
    for i=1:size(chosenEdges,1), stationKeys{i} = sprintf('%d-%d', chosenEdges(i,1), chosenEdges(i,2)); end
    for e=1:Nevents
        rec = chargeLog(e);
        key = sprintf('%d-%d', rec.station_u, rec.station_v);
        sidx = find(strcmp(stationKeys,key),1);
        if isempty(sidx), continue; end
        events.stationIdx(e) = sidx;
        events.energy_kWh_req(e) = (rec.chargeSoC / 100) * battery_kWh;
        events.task(e) = rec.task;
        % compute arrival time in MINUTES along path to station (approx)
        path = localPath(OD(rec.task,1), OD(rec.task,2), nextHop);
        arrivalMin = 0;
        for k = 1:(numel(path)-1)
            u = path(k); v = path(k+1);
            edgeKm = dist(u,v) * kmPerUnit;
            dt_hr = edgeKm / speed_kmh;
            arrivalMin = arrivalMin + dt_hr * 60;
            if u==rec.station_u && v==rec.station_v, break; end
        end
        % add small random jitter so arrivals spread within hour
        arrivalMin = arrivalMin + (rec.hour-1)*60 + rand()*60;
        events.arrival_hr(e) = arrivalMin; % stored in minutes
    end
    events = events(events.stationIdx>0 & events.energy_kWh_req>0, :);
end

function [scoreProfit, estAvgWait] = evaluatePriceScheduleProfitWithWaitEstimate(events, priceSchedule, gridCost, investPerStation, gamma, slotsPerDay, slotDur_min, numDC, DC_power, battery_kWh)
    % Fast estimate used by PSO
    K = size(priceSchedule,1);
    totalRevenue = 0; totalEnergy = 0;
    slotDur_hr = slotDur_min / 60;
    nEvents = height(events);
    slotDemand_kWh = zeros(slotsPerDay,1);
    for i=1:nEvents
        s = events.stationIdx(i);
        if s==0, continue; end
        slotIdx = min(slotsPerDay, max(1, ceil(events.arrival_hr(i) / slotDur_min)));
        eReq = events.energy_kWh_req(i);
        slotDemand_kWh(slotIdx) = slotDemand_kWh(slotIdx) + eReq;
        price_here = priceSchedule(s, slotIdx);
        totalRevenue = totalRevenue + price_here * eReq;
        totalEnergy = totalEnergy + eReq;
    end
    runningCost = gamma * investPerStation * (slotsPerDay * slotDur_hr);
    totalGridCost = gridCost * totalEnergy;
    scoreProfit = totalRevenue - totalGridCost - runningCost;
    capPerSlot_kWh = numDC * DC_power * slotDur_hr * K;
    shortage = max(0, slotDemand_kWh - capPerSlot_kWh);
    if numDC*DC_power > 0
        estDelayPerSlot = shortage ./ (numDC * DC_power) * 60; % minutes
    else
        estDelayPerSlot = shortage / (11.2) * 60;
    end
    estAvgWait = mean(estDelayPerSlot);
end

function [scheduled, stationReport, slotLoad_kW, stationSlotLoad_kW] = scheduleEvents_15min_with_horizon(events, priceSchedule, K, numAC, numDC, DC_power, battery_kWh, slotDur_min, slotsPerDay, full_charge_minutes)
    % Schedules events by arrival time; returns per-event schedule and per-station slot loads
    nEvents = height(events);
    scheduled = table(events.evtID, events.stationIdx, events.energy_kWh_req, events.arrival_hr, ...
        nan(nEvents,1), nan(nEvents,1), repmat({''}, nEvents, 1), zeros(nEvents,1), ...
        'VariableNames', {'evtID','stationIdx','energy_kWh_req','arrival_hr','start_hr','end_hr','chargerType','energy_kWh_delivered'});
    totalChgPerStation = max(1, numAC + numDC);
    stationChargerNextFree = cell(K,1);
    stationChargerType = cell(K,1);
    for s=1:K
        stationChargerNextFree{s} = zeros(totalChgPerStation,1); % next free time in minutes
        if numAC + numDC == 0
            stationChargerType{s} = {'DC'};
        else
            types = [repmat({'AC'},numAC,1); repmat({'DC'}, numDC,1)];
            if isempty(types), types = {'DC'}; end
            stationChargerType{s} = types;
        end
    end

    % initialize station slot load (kW)
    stationSlotLoad_kW = zeros(slotsPerDay, K);

    % schedule by arrival time (ascending)
    [~, ord] = sort(events.arrival_hr);
    for idx = ord'
        s = events.stationIdx(idx);
        if s == 0, continue; end
        reqEnergy = events.energy_kWh_req(idx);
        arrivalMin = events.arrival_hr(idx); % minutes

        bestDelivered = 0; bestIdx = 0; bestStart = NaN; bestEnd = NaN; bestType = '';
        for j = 1:length(stationChargerNextFree{s})
            cType = stationChargerType{s}{j};
            start_j = max(arrivalMin, stationChargerNextFree{s}(j));
            frac = min(1, reqEnergy / battery_kWh);
            t_full_min = (full_charge_minutes) * frac; % minutes to charge frac of battery
            if strcmp(cType,'DC')
                maxPower = DC_power;
            else
                maxPower = 11.2; % AC
            end
            if t_full_min <= 0
                avgPower = maxPower;
                t_full_min = (reqEnergy / avgPower) * 60;
            else
                avgPower = reqEnergy / (t_full_min/60);
                if avgPower > maxPower
                    avgPower = maxPower;
                    t_full_min = (reqEnergy / avgPower) * 60;
                end
            end
            end_min = start_j + t_full_min;
            delivered = reqEnergy;
            if delivered > bestDelivered || (abs(delivered-bestDelivered) < 1e-9 && (isnan(bestStart) || start_j < bestStart))
                bestDelivered = delivered; bestIdx = j; bestStart = start_j; bestEnd = end_min; bestType = cType;
            end
        end
        if bestIdx == 0, continue; end
        scheduled.start_hr(idx) = bestStart;      % minutes
        scheduled.end_hr(idx) = bestEnd;
        scheduled.chargerType{idx} = bestType;
        scheduled.energy_kWh_delivered(idx) = bestDelivered;
        stationChargerNextFree{s}(bestIdx) = bestEnd;

        % Add to stationSlotLoad_kW across slots spanned
        startSlot = max(1, min(slotsPerDay, ceil(bestStart / slotDur_min)));
        endSlot = max(1, min(slotsPerDay, ceil(bestEnd / slotDur_min)));
        duration_hr = (bestEnd - bestStart) / 60;
        if duration_hr > 0
            avgPower = bestDelivered / duration_hr; % kW
            stationSlotLoad_kW(startSlot:endSlot, s) = stationSlotLoad_kW(startSlot:endSlot, s) + avgPower;
        end
    end

    % aggregate slot load (sum across stations)
    slotLoad_kW = sum(stationSlotLoad_kW, 2);

    stationReport = table((1:K)', zeros(K,1), zeros(K,1), zeros(K,1), zeros(K,1), zeros(K,1), ...
        'VariableNames', {'Station','EVs_FullyServed','EVs_PartiallyServed','EVs_NotServed','EnergySold_kWh','TotalChargeTime_hr'});
    for s=1:K
        mask = scheduled.stationIdx==s;
        if ~any(mask), continue; end
        reqE = scheduled.energy_kWh_req(mask);
        gotE = scheduled.energy_kWh_delivered(mask);
        validIdx = ~isnan(gotE);
        stationReport.EVs_FullyServed(s) = sum(abs(gotE(validIdx) - reqE(validIdx)) < 1e-6);
        stationReport.EVs_PartiallyServed(s) = sum(gotE > 0 & gotE < reqE - 1e-6);
        stationReport.EVs_NotServed(s) = sum(gotE == 0 | isnan(gotE));
        stationReport.EnergySold_kWh(s) = sum(gotE(validIdx),'omitnan');
        dur_min = scheduled.end_hr(mask) - scheduled.start_hr(mask);
        dur_min = dur_min(~isnan(dur_min));
        stationReport.TotalChargeTime_hr(s) = sum(dur_min) / 60; % convert to hours
    end
end

function events_out = applyPriceResponseWithRedirect(events_in, priceSchedule, slotDur_min, slotsPerDay, flatPrice, alpha, maxDelaySlots, chosenEdges, OD, nextHop)
    % Price-response + possibility to redirect to alternate station on same path
    % Acceptance probability: p = 1/(1+exp(alpha*(price - flatPrice)))
    events_out = events_in;
    nEvents = height(events_in);
    stationKeys = cell(size(chosenEdges,1),1);
    for i=1:size(chosenEdges,1), stationKeys{i} = sprintf('%d-%d', chosenEdges(i,1), chosenEdges(i,2)); end

    for i=1:nEvents
        s = events_in.stationIdx(i);
        if s==0, continue; end
        arrival = events_in.arrival_hr(i); % minutes
        slotIdx = max(1, min(slotsPerDay, ceil(arrival / slotDur_min)));
        attempts = 0;
        accepted = false;
        currentStation = s;
        while ~accepted && attempts <= maxDelaySlots
            price_here = priceSchedule(currentStation, slotIdx);
            p_accept = 1 / (1 + exp(alpha * (price_here - flatPrice)));
            if rand() < p_accept
                accepted = true;
                break;
            else
                % defer by one slot first
                arrival = arrival + slotDur_min;
                slotIdx = min(slotsPerDay, slotIdx + 1);
                attempts = attempts + 1;
                % consider redirect to cheapest station on path at this slot
                taskIdx = events_in.task(i);
                path = localPath(OD(taskIdx,1), OD(taskIdx,2), nextHop);
                candidateStations = [];
                for kk=1:(numel(path)-1)
                    key = sprintf('%d-%d', path(kk), path(kk+1));
                    idx = find(strcmp(stationKeys,key),1);
                    if ~isempty(idx), candidateStations(end+1) = idx; end
                end
                if ~isempty(candidateStations)
                    prices = priceSchedule(candidateStations, slotIdx);
                    [minPrice, minIdxLocal] = min(prices);
                    newStation = candidateStations(minIdxLocal);
                    % allow redirect if strictly cheaper or after many attempts
                    if newStation ~= currentStation && (minPrice < price_here || attempts > max(1, round(maxDelaySlots/2)))
                        currentStation = newStation;
                    end
                end
            end
        end
        events_out.arrival_hr(i) = arrival;
        events_out.stationIdx(i) = currentStation;
    end
end
