%% evcs_full_24h_sim_corrected_v10.m
% 24-hour EVCS simulation with 4 pricing schemes:
% flat / ToU / CPP / dynamic (PSO).
%
% Key points:
%   - 1-hour slots, continuous-time charging (minutes)
%   - robust greedy siting
%   - Monte-Carlo OD EV generation
%   - Dynamic pricing uses PSO + price-based rerouting of EVs
%   - Rerouting ONLY in dynamic mode (via scheduler)
%   - Grid capacity = K stations × 5 DC × 75 kW (<= 2250 kW)
%   - NO ghost EVs
%   - Axes limits auto-scaled so graphs don’t clip

clc; clear; close all;
rng('shuffle');

%% -------------------- USER-TUNABLE PARAMETERS --------------------
% --- SoC / travel
minSOC          = 60;
maxSOC          = 85;
consRate        = 1;     % %SOC per "distance unit"
chargeThreshold = 55;
stopSOC         = 20;

% --- Greedy objective weights (linear)
w_charge = 2.0;    % reward charge sold (positive)
w_stop   = -2.0;   % penalize stops (negative)
w_var    = -3.5;   % penalize bus variance (negative -> prefer flat)

numStations_req = 6;        % number of stations desired
maxGreedyIter    = 1000;

% Robust greedy settings
robustGreedySeed = 42;      % fixed seed base to make greedy repeatable
nGreedyMCtrials  = 8;       % number of MC trials to average per candidate

% --- vehicles per OD (flexible modes)
maxVehiclesPerOD    = 50;   % legacy upper bound (still used in helpers)
vehiclesMode.type   = 'perOD_scalar'; % 'perOD_scalar','perOD_vector','perOD_hourly','stochastic'
vehiclesMode.scalar = 150;           % baseline EVs per OD per hour
vehiclesMode.dist   = 'poisson';
vehiclesMode.lambda = 80;            % mean for poisson (if stochastic)

% --- Time resolution: 1-hour slots, charging continuous (minutes)
slotDur_min = 60;          % 1 hour per slot
hoursInDay  = 24;
slotsPerDay = hoursInDay;  % exactly 24 slots
plotPoints  = slotsPerDay; % show all 24 hours

% --- Pricing & economics
gridCost_RsPerkWh = 9.5;
flatPrice         = 12;     % Rs/kWh baseline

% ToU: same pattern as before, but directly hourly
ToU_rate = flatPrice * ones(24,1);
ToU_rate(18:21) = flatPrice * 1.7;  % evening higher (18–21 h)

CPP_hours     = [18,19];    % critical peak hours
CPP_surcharge = 30;         % Rs/kWh surcharge during CPP hours

% --- Feeder/grid capacity (will be set after K is known)
gridCapacity_kW = NaN;      % placeholder; updated after greedy siting

% --- PSO for dynamic pricing
pso_nParticles = 20;
pso_nIter      = 120;
minPrice_kWh   = 9;
maxPrice_kWh   = 28;

% Keep profit dominant but include soft penalties
betaWait       = 500;    % penalty on waiting (estimated)
betaPeak       = 2000;   % penalty on peak above gridCapacity

% --- Charger hardware
battery_kWh         = 59;    % Mahindra BE.6
range_km            = 557;
numAC               = 0;     % AC ports per station
numDC               = 5;     % DC ports per station
DC_power            = 75;    % kW (nameplate)
full_charge_minutes = 35;    % DC taper 0->100% in 35 min
kmPerUnit           = 2;     % km / "distance unit"
speed_kmh           = 60;    % driving speed for arrival time

% --- economic constants
gamma       = 0.01;
plugCostDC  = 10000;
investmentPerStation = numDC * plugCostDC;

% --- dynamic dispatch attraction parameter (minutes per Rs)
lambdaPrice_min = 2.0;   % higher -> price matters more vs waiting

% --- ghost station fix now disabled (no fake events)
minEventsPerStation = 0;

% --- background station load (kW) if you want
backgroundLoadPerStation_kW = 0.0; % keep 0 so station-load plots are clean

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
dist = M;  % graph units

% Floyd–Warshall for nextHop (shortest path reconstruction)
nextHop = zeros(n);
for i = 1:n
    for j = 1:n
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
                dist(i,j)    = nd;
                nextHop(i,j) = nextHop(i,k);
            end
        end
    end
end

% OD pairs from Word
txt   = extractFileText('OD nodes.docx');
pairs = regexp(txt,'\((\d+)\s*,\s*(\d+)\)','tokens');
OD    = cellfun(@(c)[str2double(c{1}) str2double(c{2})], pairs, 'UniformOutput', false);
OD    = vertcat(OD{:});
if isempty(OD), error('No OD pairs found in OD nodes.docx'); end
if max(OD(:)) > n, error('OD refers to node > size of distance matrix'); end

% candidate edges and per-task path
candidateEdges = [];
taskPaths      = cell(size(OD,1),1);
for t = 1:size(OD,1)
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
if numel(vvec) < 24, error('24hr_vehicles.xlsx must have 24 values'); end
hourlyRatio = double(vvec(1:24));
if max(hourlyRatio) > 1
    hourlyRatio = hourlyRatio / max(hourlyRatio);
    warning('hourly ratios scaled so max==1');
end

% Bus-to-edge mapping (From,To,BusIndex)
mappingMatrix = readmatrix('Bus to edge mapping.xlsx');
if size(mappingMatrix,2) < 3
    error('Bus mapping sheet must have at least 3 columns: From,To,BusIndex');
end
busMapping = mappingMatrix(:,1:3);

% IEEE69 base load (optional)
try
    baseLoad = readmatrix('IEEE69_24h active data.xlsx');
    if size(baseLoad,1) < size(baseLoad,2), baseLoad = baseLoad.'; end
    baseLoad(isnan(baseLoad)) = 0;
catch
    baseLoad = [];
end

% Map candidate edges to bus index (if available) -- not used further now
candBus = zeros(size(candidateEdges,1),1);
for i = 1:size(candidateEdges,1)
    u = candidateEdges(i,1); v = candidateEdges(i,2);
    idx = find(busMapping(:,1)==u & busMapping(:,2)==v,1);
    if isempty(idx), idx = find(busMapping(:,1)==v & busMapping(:,2)==u,1); end
    if ~isempty(idx), candBus(i) = busMapping(idx,3); end
end

%% -------------------- Prepare vehiclesPerOD after reading OD count ----
nTasks = numel(taskPaths);
if strcmp(vehiclesMode.type,'perOD_vector')
    if ~isfield(vehiclesMode,'vector') || numel(vehiclesMode.vector)~=nTasks
        warning('vehiclesMode.perOD_vector wrong size; using scalar fallback.');
        vehiclesMode.vector = vehiclesMode.scalar * ones(nTasks,1);
    end
end
if strcmp(vehiclesMode.type,'perOD_hourly')
    if ~isfield(vehiclesMode,'matrix') || ~isequal(size(vehiclesMode.matrix), [24,nTasks])
        warning('vehiclesMode.perOD_hourly matrix not provided; building from scalar*hourlyRatio');
        vehiclesMode.matrix = repmat(vehiclesMode.scalar * hourlyRatio(:), 1, nTasks);
    end
end

%% -------------------- Robust Greedy placement (MC averaged) ----
remainingEdges = candidateEdges;
chosenEdges    = zeros(0,2);
iter = 0;
fprintf('\nStarting robust greedy placement (seed=%d)...\n', robustGreedySeed);

seedOffsets = robustGreedySeed + (1:nGreedyMCtrials);

while size(chosenEdges,1) < numStations_req && ~isempty(remainingEdges) && iter < maxGreedyIter
    iter = iter + 1;
    bestEdge = []; bestObj = -Inf;

    % baseline with current chosenEdges
    baseObjs = zeros(nGreedyMCtrials,1);
    for tt = 1:nGreedyMCtrials
        rng(seedOffsets(tt));
        [baseCharge, baseStops, baseBusVar] = simulateWithStationsSimple_24h( ...
            chosenEdges, taskPaths, OD, dist, consRate, hourlyRatio, ...
            maxVehiclesPerOD, minSOC, maxSOC, chargeThreshold, stopSOC, ...
            baseLoad, kmPerUnit, speed_kmh, vehiclesMode);
        baseObjs(tt) = w_charge*baseCharge + w_stop*baseStops + w_var*baseBusVar;
    end
    baseObjAvg = mean(baseObjs);

    % try each remaining candidate edge
    for eidx = 1:size(remainingEdges,1)
        candEdge = remainingEdges(eidx,:);
        trialStations = [chosenEdges; candEdge];
        objs = zeros(nGreedyMCtrials,1);
        for tt = 1:nGreedyMCtrials
            rng(seedOffsets(tt)+eidx);
            [trialCharge, trialStops, trialBusVar] = simulateWithStationsSimple_24h( ...
                trialStations, taskPaths, OD, dist, consRate, hourlyRatio, ...
                maxVehiclesPerOD, minSOC, maxSOC, chargeThreshold, stopSOC, ...
                baseLoad, kmPerUnit, speed_kmh, vehiclesMode);
            objs(tt) = w_charge*trialCharge + w_stop*trialStops + w_var*trialBusVar;
        end
        objAvg = mean(objs);
        if objAvg > bestObj
            bestObj  = objAvg;
            bestEdge = candEdge;
        end
    end

    if isempty(bestEdge) || bestObj <= baseObjAvg
        fprintf('No improving candidate at iteration %d. Stopping.\n', iter);
        break;
    end

    chosenEdges = [chosenEdges; bestEdge];
    % remove chosen (undirected)
    maskRem = ~((remainingEdges(:,1)==bestEdge(1) & remainingEdges(:,2)==bestEdge(2)) | ...
                (remainingEdges(:,1)==bestEdge(2) & remainingEdges(:,2)==bestEdge(1)));
    remainingEdges = remainingEdges(maskRem,:);
    fprintf('Selected edge %d->%d (iter %d). Total chosen %d/%d\n', ...
        bestEdge(1), bestEdge(2), iter, size(chosenEdges,1), numStations_req);
end

K = size(chosenEdges,1);
fprintf('Selected %d stations total.\n', K);

% ---- Set realistic feeder capacity now : K * 5 DC * 75 kW ----
gridCapacity_kW = K * numDC * DC_power;  % <= 2250 kW for K=6

%% -------------------- Full Monte Carlo 24h -> events ----
fprintf('\nRunning full 24-hr Monte Carlo to generate events...\n');
rng(100);
[totalChargeFinal, totalStoppedFinal, stationCount, stationCharge, ...
    chargeLog, OD_hourly_counts] = simulateWithStationsFull_24h( ...
        chosenEdges, taskPaths, OD, dist, consRate, hourlyRatio, ...
        maxVehiclesPerOD, minSOC, maxSOC, chargeThreshold, stopSOC, ...
        vehiclesMode, minEventsPerStation);

events = buildEventsFromChargeLog(chargeLog, chosenEdges, OD, nextHop, ...
                                  dist, battery_kWh, kmPerUnit, speed_kmh);
if isempty(events)
    warning('No charging events produced; exiting.');
    return;
end

% Save OD hourly report
odTable = array2table(OD_hourly_counts, ...
    'VariableNames', arrayfun(@(x) sprintf('Task_%d',x),1:size(OD_hourly_counts,2),'UniformOutput',false));
odTable.Hour = (1:24)';
odTable = movevars(odTable,'Hour','Before',1);
writetable(odTable, 'OD_hourly_EV_report.xlsx');
fprintf('Saved OD_hourly_EV_report.xlsx\n');

%% -------------------- Pricing simulations --------------------
pricingModes = {'flat','ToU','CPP','dynamic'};
numModes     = numel(pricingModes);

TS_energy    = zeros(slotsPerDay, numModes);
TS_revenue   = zeros(slotsPerDay, numModes);
TS_wait_mean = nan(slotsPerDay, numModes);
TS_wait_min  = nan(slotsPerDay, numModes);
TS_wait_max  = nan(slotsPerDay, numModes);
TS_slotLoad  = zeros(slotsPerDay, numModes);

TotalRevenueModes = zeros(1,numModes);
TotalProfitModes  = zeros(1,numModes);
TotalEnergyModes  = zeros(1,numModes);
AvgWaitModes      = nan(1,numModes);
PeakLoadModes     = zeros(1,numModes);
AvgLoadModes      = zeros(1,numModes);

excelStationLoadsFile = 'station_loads_all_modes.xlsx';
excelPriceFile        = 'price_schedule_all_modes.xlsx';
try
    if exist(excelStationLoadsFile,'file'), delete(excelStationLoadsFile); end
    if exist(excelPriceFile,'file'), delete(excelPriceFile); end
catch
end

for pm = 1:numModes
    mode = pricingModes{pm};
    fprintf('\nRunning pricing simulation: %s\n', mode);

    %% price schedule K x slotsPerDay (Rs/kWh)
    switch mode
        case 'flat'
            priceSchedule = flatPrice * ones(K, slotsPerDay);

        case 'ToU'
            priceSchedule = repmat(ToU_rate(:)', K, 1);

        case 'CPP'
            priceSchedule = zeros(K, slotsPerDay);
            for h = 1:24
                base = ToU_rate(h);
                if ismember(h, CPP_hours), base = base + CPP_surcharge; end
                priceSchedule(:,h) = base;
            end

        case 'dynamic'
            % PSO over Kx24 prices
            particles = minPrice_kWh + (maxPrice_kWh-minPrice_kWh)* ...
                        rand(K, slotsPerDay, pso_nParticles);
            bestCandidate = particles(:,:,1);
            bestScore     = -Inf;

            for it = 1:pso_nIter
                for p = 1:pso_nParticles
                    cand = particles(:,:,p);
                    [profitEst, estAvgWait, peakDemand_kW] = ...
                        evaluatePriceScheduleProfitWithWaitEstimate( ...
                            events, cand, gridCost_RsPerkWh, ...
                            investmentPerStation, gamma, slotsPerDay, ...
                            slotDur_min, numDC, DC_power, battery_kWh, ...
                            gridCapacity_kW);

                    overLimit = max(0, peakDemand_kW - gridCapacity_kW);
                    score = profitEst - betaWait*estAvgWait ...
                                      - betaPeak*(overLimit.^2);

                    if score > bestScore
                        bestScore     = score;
                        bestCandidate = cand;
                    end
                end

                % simple global-best update + noise
                for p = 1:pso_nParticles
                    particles(:,:,p) = particles(:,:,p) ...
                        + 0.2*(bestCandidate-particles(:,:,p)) ...
                        + 0.05*randn(K,slotsPerDay);
                    particles(:,:,p) = max(minPrice_kWh, ...
                        min(maxPrice_kWh, particles(:,:,p)));
                end
            end
            priceSchedule = bestCandidate;

        otherwise
            priceSchedule = flatPrice * ones(K, slotsPerDay);
    end

    %% save priceSchedule
    priceSlots = priceSchedule.'; % slotsPerDay x K
    csvName    = sprintf('price_schedule_%s.csv', mode);
    writematrix(priceSlots, csvName);
    priceTbl = array2table(priceSlots, ...
        'VariableNames', arrayfun(@(x) sprintf('Station_%d',x),1:K,'UniformOutput',false));
    priceTbl.Time_slot = (1:size(priceSlots,1))';
    priceTbl = movevars(priceTbl,'Time_slot','Before',1);
    try
        writetable(priceTbl, excelPriceFile, 'Sheet', upper(mode));
    catch
        try
            xlswrite(excelPriceFile, priceTbl{:,:}, upper(mode));
        catch
            warning('Could not write price schedule to multi-sheet Excel. CSV saved: %s', csvName);
        end
    end

    minP  = min(priceSchedule(:));
    maxP  = max(priceSchedule(:));
    meanP = mean(priceSchedule(:));
    fprintf('Price summary (%s): min=%.2f mean=%.2f max=%.2f Rs/kWh\n', ...
        upper(mode),minP,meanP,maxP);

    %% scheduling & dynamic rerouting
    events_mode = events;       % same arrivals for all
    if strcmp(mode,'dynamic')
        lambdaPrice = lambdaPrice_min;   % price matters
    else
        lambdaPrice = 0;                 % only waiting matters
    end

    [scheduled, stationReport, slotLoad_kW, stationSlotLoad_kW] = ...
        scheduleEvents_with_dynamic(events_mode, priceSchedule, K, ...
            numAC, numDC, DC_power, battery_kWh, slotDur_min, ...
            slotsPerDay, full_charge_minutes, mode, lambdaPrice);

    %% add background then enforce grid capacity
    stationSlotLoad_kW = stationSlotLoad_kW + backgroundLoadPerStation_kW;
    slotLoad_kW        = sum(stationSlotLoad_kW,2);

    rawPeak = max(slotLoad_kW);
    if rawPeak > gridCapacity_kW + 1e-6
        scale = gridCapacity_kW / rawPeak;
        stationSlotLoad_kW = stationSlotLoad_kW * scale;
        slotLoad_kW        = slotLoad_kW * scale;
        fprintf('Mode %s: peak %.3f kW scaled to %.3f kW\n', mode, rawPeak, gridCapacity_kW);
    end

    %% station revenue/profit (approx)
    stationReport.Revenue_Rs     = zeros(K,1);
    stationReport.Profit_Rs      = zeros(K,1);
    stationReport.EnergySold_kWh = zeros(K,1);

    for s = 1:K
        mask = scheduled.stationIdx==s;
        if ~any(mask), continue; end
        idxs = find(mask);
        revenue_s = 0;
        for ii = 1:numel(idxs)
            i = idxs(ii);
            if isnan(scheduled.start_hr(i)), continue; end
            startSlot  = max(1,min(slotsPerDay,ceil(scheduled.start_hr(i)/slotDur_min)));
            price_here = priceSchedule(s,startSlot);
            e          = scheduled.energy_kWh_delivered(i);
            revenue_s  = revenue_s + price_here*e;
        end
        totalEnergy = sum(scheduled.energy_kWh_delivered(mask),'omitnan');
        stationReport.Revenue_Rs(s) = revenue_s;
        stationReport.Profit_Rs(s)  = stationReport.Revenue_Rs(s) ...
            - gridCost_RsPerkWh*totalEnergy - gamma*investmentPerStation;
        stationReport.EnergySold_kWh(s) = totalEnergy;
    end

    zeroStations = find(stationReport.EnergySold_kWh < 1e-9);
    if ~isempty(zeroStations)
        fprintf('Stations with near-zero energy (likely off OD paths): ');
        fprintf('%d ', zeroStations); fprintf('\n');
    end

    %% save per-mode outputs
    prefix = sprintf('results_%s',mode);
    writetable(stationReport, sprintf('%s_stationReport.xlsx', prefix));
    writetable(scheduled,      sprintf('%s_scheduled_events.xlsx', prefix));
    writematrix(slotLoad_kW,   sprintf('%s_slotLoad_kW.csv', prefix));

    writematrix(stationSlotLoad_kW, sprintf('station_load_%s.csv', mode));
    tblStationLoad = array2table(stationSlotLoad_kW, ...
        'VariableNames', arrayfun(@(x) sprintf('Station_%d',x),1:K,'UniformOutput',false));
    tblStationLoad.Time_slot = (1:size(stationSlotLoad_kW,1))';
    tblStationLoad = movevars(tblStationLoad,'Time_slot','Before',1);
    try
        writetable(tblStationLoad, excelStationLoadsFile, 'Sheet', upper(mode));
    catch
        try
            xlswrite(excelStationLoadsFile, tblStationLoad{:,:}, upper(mode));
        catch
            warning('Could not write multi-sheet station loads. CSV saved per mode.');
        end
    end

    %% time-series metrics
    validMask = scheduled.energy_kWh_delivered > 0;
    waitTimes_min = max(0, scheduled.start_hr(validMask) - ...
                           scheduled.arrival_hr(validMask));
    totalEnergyAll  = sum(stationReport.EnergySold_kWh);
    totalRevenueAll = sum(stationReport.Revenue_Rs);
    totalProfitAll  = sum(stationReport.Profit_Rs);

    if isempty(waitTimes_min)
        wt_min=NaN; wt_mean=NaN; wt_max=NaN;
    else
        wt_min  = min(waitTimes_min);
        wt_mean = mean(waitTimes_min);
        wt_max  = max(waitTimes_min);
    end

    wt_min_s  = wt_min*60;
    wt_mean_s = wt_mean*60;
    wt_max_s  = wt_max*60;

    fprintf('\n=== REPORT %s ===\n', upper(mode));
    fprintf('Energy=%.2f kWh | Revenue=Rs %.2f | Profit=Rs %.2f | Avg wait=%.2f sec\n', ...
        totalEnergyAll,totalRevenueAll,totalProfitAll,wt_mean_s);

    slotEnergy  = zeros(slotsPerDay,1);
    slotRevenue = zeros(slotsPerDay,1);
    slotWaits   = cell(slotsPerDay,1);

    for i = 1:height(scheduled)
        if scheduled.energy_kWh_delivered(i)<=0 || isnan(scheduled.start_hr(i)), continue; end
        startSlot = max(1,min(slotsPerDay,ceil(scheduled.start_hr(i)/slotDur_min)));
        slotEnergy(startSlot)  = slotEnergy(startSlot) + scheduled.energy_kWh_delivered(i);
        sIdx                  = scheduled.stationIdx(i);
        price_here            = priceSchedule(sIdx,startSlot);
        slotRevenue(startSlot)= slotRevenue(startSlot) + ...
                                scheduled.energy_kWh_delivered(i)*price_here;
        w = max(0,scheduled.start_hr(i)-scheduled.arrival_hr(i));
        slotWaits{startSlot} = [slotWaits{startSlot}, w];
    end

    for sl = 1:slotsPerDay
        arr = slotWaits{sl};
        if isempty(arr)
            TS_wait_mean(sl,pm)=NaN;
            TS_wait_min(sl,pm) =NaN;
            TS_wait_max(sl,pm) =NaN;
        else
            TS_wait_mean(sl,pm)=mean(arr)*60;
            TS_wait_min(sl,pm) =min(arr)*60;
            TS_wait_max(sl,pm) =max(arr)*60;
        end
        TS_energy(sl,pm)   = slotEnergy(sl);
        TS_revenue(sl,pm)  = slotRevenue(sl);
        TS_slotLoad(sl,pm) = slotLoad_kW(sl);
    end

    TotalRevenueModes(pm) = totalRevenueAll;
    TotalProfitModes(pm)  = totalProfitAll;
    TotalEnergyModes(pm)  = totalEnergyAll;
    AvgWaitModes(pm)      = wt_mean_s;
    PeakLoadModes(pm)     = max(slotLoad_kW);
    AvgLoadModes(pm)      = mean(slotLoad_kW);

    %% diagnostics plots per mode (with auto y-limits)
    hours = (1:slotsPerDay)';
    plotCount = min(plotPoints, slotsPerDay);

    maxWaitMean = max(TS_wait_mean(:,pm),[],'omitnan');
    maxWaitMax  = max(TS_wait_max(:,pm),[],'omitnan');
    maxEnergy   = max(TS_energy(:,pm));
    maxRevenue  = max(TS_revenue(:,pm));
    maxLoad     = max(TS_slotLoad(:,pm));

    fig = figure('Units','normalized','Position',[0.05 0.05 0.9 0.85], ...
        'Name',sprintf('Diagnostics - %s',upper(mode)),'Color','w');

    subplot(3,2,1);
    plot(hours(1:plotCount), TS_wait_mean(1:plotCount,pm),'LineWidth',1.5);
    title(sprintf('%s: Average Waiting Time (sec)',upper(mode)));
    xlabel('Hour'); ylabel('Waiting Time (sec)'); grid on; xlim([1 24]);
    ylim([0 max(maxWaitMean,1)*1.2]);

    subplot(3,2,2); hold on;
    plot(hours(1:plotCount), TS_wait_min(1:plotCount,pm),'b-','LineWidth',1.2,'DisplayName','Min');
    plot(hours(1:plotCount), TS_wait_max(1:plotCount,pm),'r-','LineWidth',1.2,'DisplayName','Max');
    title(sprintf('%s: Min/Max Waiting Time (sec)',upper(mode)));
    xlabel('Hour'); ylabel('Waiting Time (sec)'); legend; grid on; xlim([1 24]);
    ylim([0 max(maxWaitMax,1)*1.2]);

    subplot(3,2,3);
    plot(hours(1:plotCount), TS_energy(1:plotCount,pm),'LineWidth',1.5);
    title(sprintf('%s: Energy Sold (kWh)',upper(mode)));
    xlabel('Hour'); ylabel('Energy (kWh/slot)'); grid on; xlim([1 24]);
    ylim([0 max(maxEnergy,1)*1.2]);

    subplot(3,2,4);
    plot(hours(1:plotCount), TS_revenue(1:plotCount,pm),'LineWidth',1.5);
    title(sprintf('%s: Revenue (Rs)',upper(mode)));
    xlabel('Hour'); ylabel('Revenue (Rs/slot)'); grid on; xlim([1 24]);
    ylim([0 max(maxRevenue,1)*1.2]);

    subplot(3,2,5);
    profit_slot = TS_revenue(1:plotCount,pm) - gridCost_RsPerkWh*TS_energy(1:plotCount,pm);
    plot(hours(1:plotCount),profit_slot,'LineWidth',1.5);
    title(sprintf('%s: Profit (Rs)',upper(mode)));
    xlabel('Hour'); ylabel('Profit (Rs/slot)'); grid on; xlim([1 24]);
    ylim([min(0,min(profit_slot))*1.2, max(max(profit_slot),1)*1.2]);

    subplot(3,2,6);
    plot(hours(1:plotCount),TS_slotLoad(1:plotCount,pm),'LineWidth',1.5);
    title(sprintf('%s: Load (kW)',upper(mode)));
    xlabel('Hour'); ylabel('Load (kW)'); grid on; xlim([1 24]);
    ylim([0 max(maxLoad,gridCapacity_kW)*1.1]);
    yline(gridCapacity_kW,'r--','Cap');

    saveas(fig,sprintf('diagnostics_%s.png',mode));
    close(fig);

    % station load plot (zoomed to station capacity)
    figS = figure('Units','normalized','Position',[0.05 0.05 0.9 0.6], ...
        'Name',sprintf('Station loads - %s',upper(mode)),'Color','w');
    hold on;
    for s = 1:K
        plot(hours(1:plotCount),stationSlotLoad_kW(1:plotCount,s),'-','LineWidth',1);
    end
    xlabel('Hour'); ylabel('Station Load (kW)');
    title(sprintf('%s: Station loads (kW) per hour slot',upper(mode)));
    legend(arrayfun(@(x) sprintf('Station %d',x),1:K,'UniformOutput',false),'Location','eastoutside');
    grid on; xlim([1 24]); ylim([0 numDC*DC_power*1.2]);   % ~0–375 kW per station
    saveas(figS,sprintf('station_load_%s.png',mode));
    close(figS);

    fprintf('Saved diagnostics and station loads for %s\n',mode);
end

%% -------------------- Comparison & Plots --------------------
labels = pricingModes;

figure('Units','normalized','Position',[0.05 0.05 0.9 0.85]);

subplot(2,3,1);
bar(TotalRevenueModes/1000); set(gca,'XTickLabel',labels);
ylabel('Revenue (x1000 Rs)'); title('Revenue (Rs 1000s)');
ylim([0 max(TotalRevenueModes/1000)*1.2]);

subplot(2,3,2);
bar(TotalProfitModes/1000); set(gca,'XTickLabel',labels);
ylabel('Profit (x1000 Rs)'); title('Profit (Rs 1000s)');
ylim([min(0,min(TotalProfitModes/1000))*1.2, max(TotalProfitModes/1000)*1.2]);

subplot(2,3,3);
bar(TotalEnergyModes); set(gca,'XTickLabel',labels);
ylabel('Energy (kWh)'); title('Total Energy Delivered (kWh)');
ylim([0 max(TotalEnergyModes)*1.2]);

subplot(2,3,4);
bar(AvgWaitModes); set(gca,'XTickLabel',labels);
ylabel('Average waiting (sec)'); title('Average Waiting Time (sec)');
ylim([0 max(AvgWaitModes)*1.2]);

subplot(2,3,5);
bar([PeakLoadModes; AvgLoadModes]'); set(gca,'XTickLabel',labels);
legend('Peak','Avg'); title('Peak & Avg EVCS Load (kW)');
ylim([0 max([PeakLoadModes AvgLoadModes])*1.2]);
yline(gridCapacity_kW,'r--','Cap');

hours = (1:slotsPerDay)';
plotCount = min(plotPoints, slotsPerDay);

figE = figure('Units','normalized','Position',[0.05 0.05 0.7 0.35]); hold on;
plot(hours(1:plotCount),TS_energy(1:plotCount,1),'-o','DisplayName','flat');
plot(hours(1:plotCount),TS_energy(1:plotCount,2),'-s','DisplayName','ToU');
plot(hours(1:plotCount),TS_energy(1:plotCount,3),'-^','DisplayName','CPP');
plot(hours(1:plotCount),TS_energy(1:plotCount,4),'-d','DisplayName','dynamic');
xlabel('Hour'); ylabel('Energy (kWh/slot)'); title('Per-slot Energy Sold');
legend; grid on; xlim([1 24]); ylim([0 max(TS_energy,[],'all')*1.2]);
saveas(figE,'timeseries_energy_per_slot.png');

figR = figure('Units','normalized','Position',[0.05 0.05 0.7 0.35]); hold on;
plot(hours(1:plotCount),TS_revenue(1:plotCount,1),'-o','DisplayName','flat');
plot(hours(1:plotCount),TS_revenue(1:plotCount,2),'-s','DisplayName','ToU');
plot(hours(1:plotCount),TS_revenue(1:plotCount,3),'-^','DisplayName','CPP');
plot(hours(1:plotCount),TS_revenue(1:plotCount,4),'-d','DisplayName','dynamic');
xlabel('Hour'); ylabel('Revenue (Rs/slot)'); title('Per-slot Revenue');
legend; grid on; xlim([1 24]); ylim([0 max(TS_revenue,[],'all')*1.2]);
saveas(figR,'timeseries_revenue_per_slot.png');

figW = figure('Units','normalized','Position',[0.05 0.05 0.7 0.35]); hold on;
plot(hours(1:plotCount),TS_wait_mean(1:plotCount,1),'-o','DisplayName','flat');
plot(hours(1:plotCount),TS_wait_mean(1:plotCount,2),'-s','DisplayName','ToU');
plot(hours(1:plotCount),TS_wait_mean(1:plotCount,3),'-^','DisplayName','CPP');
plot(hours(1:plotCount),TS_wait_mean(1:plotCount,4),'-d','DisplayName','dynamic');
xlabel('Hour'); ylabel('Avg wait (sec)'); title('Per-slot Average Waiting Time');
legend; grid on; xlim([1 24]); ylim([0 max(TS_wait_mean,[],'all')*1.2]);
saveas(figW,'timeseries_wait_avg_per_slot.png');

summaryTbl = table(labels', TotalEnergyModes', TotalRevenueModes', ...
    TotalProfitModes', AvgWaitModes', PeakLoadModes', AvgLoadModes', ...
    'VariableNames', {'Mode','TotalEnergy_kWh','TotalRevenue_Rs', ...
    'TotalProfit_Rs','AvgWait_sec','PeakLoad_kW','AvgLoad_kW'});
writetable(summaryTbl,'comparison_summary.xlsx');

fprintf('\nAll done. Summary saved to comparison_summary.xlsx.\n');

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

function [totalCharge,totalStops,busVar] = simulateWithStationsSimple_24h( ...
    stationEdges, taskPaths, OD, dist, consRate, hourlyRatio, ...
    maxVehiclesPerOD, minSOC, maxSOC, chargeThreshold, stopSOC, ...
    baseLoad, kmPerUnit, speed_kmh, vehiclesMode)

    totalCharge = 0; totalStops = 0;
    K = size(stationEdges,1);
    nTasks = numel(taskPaths);
    keys = cell(max(1,K),1);
    for ii=1:K, keys{ii} = sprintf('%d-%d',stationEdges(ii,1),stationEdges(ii,2)); end
    battery_kWh = 59;
    stationEnergyPerHour = zeros(max(1,K),24);

    for hr = 1:24
        factor = max(0,hourlyRatio(hr));
        nvehPerOD_based = vehiclesPerOD_for_hour(vehiclesMode,nTasks,hr);
        for t = 1:nTasks
            p = taskPaths{t}; if isempty(p), continue; end
            nvehPerOD = max(0, round(nvehPerOD_based(t)*factor));
            for veh = 1:nvehPerOD
                soc = minSOC + (maxSOC-minSOC)*rand();
                stopped=false;
                for k = 1:(numel(p)-1)
                    u=p(k); v=p(k+1); d=dist(u,v);
                    key = sprintf('%d-%d',u,v);
                    idx = find(strcmp(keys,key));
                    if ~isempty(idx)
                        distRem=0;
                        for mm=k:(numel(p)-1)
                            distRem = distRem + dist(p(mm),p(mm+1));
                        end
                        neededSOC = distRem*consRate;
                        if soc < chargeThreshold
                            deliverPct = max(0,neededSOC-soc);
                            totalCharge = totalCharge + deliverPct;
                            stationEnergyPerHour(idx,hr) = ...
                                stationEnergyPerHour(idx,hr) + (deliverPct/100)*battery_kWh;
                            soc = soc + deliverPct;
                        end
                    end
                    soc = soc - d*consRate;
                    if soc < stopSOC && ~stopped
                        totalStops = totalStops + 1;
                        stopped=true; soc=stopSOC;
                    end
                end
            end
        end
    end
    perStationVar = var(stationEnergyPerHour,0,2);
    busVar = mean(perStationVar);
end

function arr = vehiclesPerOD_for_hour(vehiclesMode,nTasks,hr)
    switch vehiclesMode.type
        case 'perOD_scalar'
            arr = vehiclesMode.scalar * ones(nTasks,1);
        case 'perOD_vector'
            arr = vehiclesMode.vector(:);
        case 'perOD_hourly'
            arr = vehiclesMode.matrix(hr,:)';
        case 'stochastic'
            switch vehiclesMode.dist
                case 'poisson'
                    arr = poissrnd(vehiclesMode.lambda,[nTasks,1]);
                case 'binomial'
                    arr = binornd(vehiclesMode.N,vehiclesMode.p,[nTasks,1]);
                otherwise
                    arr = vehiclesMode.scalar * ones(nTasks,1);
            end
        otherwise
            arr = vehiclesMode.scalar * ones(nTasks,1);
    end
end

function [totalCharge,totalStops,stationCount,stationCharge, ...
          chargeLog,OD_hourly_counts] = simulateWithStationsFull_24h( ...
    stationEdges, taskPaths, OD, dist, consRate, hourlyRatio, ...
    maxVehiclesPerOD, minSOC, maxSOC, chargeThreshold, stopSOC, ...
    vehiclesMode, minEventsPerStation) %#ok<INUSD>

    K = size(stationEdges,1);
    stationCount  = zeros(K,1);
    stationCharge = zeros(K,1);
    totalCharge   = 0;
    totalStops    = 0;
    chargeLog = struct('task',{},'hour',{},'evID',{}, ...
                       'station_u',{},'station_v',{},'chargeSoC',{});
    battery_kWh = 59;
    nTasks = numel(taskPaths);
    evtID  = 0;
    OD_hourly_counts = zeros(24,nTasks);

    for hr = 1:24
        factor = max(0,hourlyRatio(hr));
        nvehPerOD_based = vehiclesPerOD_for_hour(vehiclesMode,nTasks,hr);
        for t = 1:nTasks
            p = taskPaths{t}; if isempty(p), continue; end
            nvehPerOD = max(0,round(nvehPerOD_based(t)*factor));
            if nvehPerOD==0 && rand()<0.02
                nvehPerOD = 1;
            end
            OD_hourly_counts(hr,t) = nvehPerOD;
            for veh = 1:nvehPerOD
                evtID = evtID + 1;
                soc   = minSOC + (maxSOC-minSOC)*rand();
                stopped=false;
                for k = 1:(numel(p)-1)
                    u=p(k); v=p(k+1); d=dist(u,v);
                    idx = find(all(stationEdges==[u v],2),1);
                    if ~isempty(idx)
                        distRem=0;
                        for mm=k:(numel(p)-1)
                            distRem = distRem + dist(p(mm),p(mm+1));
                        end
                        neededSOC = distRem*consRate;
                        if soc < chargeThreshold
                            chargeDeliverPct = max(0,neededSOC-soc);
                            stationCount(idx)  = stationCount(idx)+1;
                            stationCharge(idx) = stationCharge(idx)+chargeDeliverPct;
                            totalCharge        = totalCharge+chargeDeliverPct;
                            soc = soc + chargeDeliverPct;
                            rec.task      = t;
                            rec.hour      = hr;
                            rec.evID      = veh;
                            rec.station_u = stationEdges(idx,1);
                            rec.station_v = stationEdges(idx,2);
                            rec.chargeSoC = chargeDeliverPct;
                            chargeLog(end+1) = rec; %#ok<AGROW>
                        end
                    end
                    soc = soc - d*consRate;
                    if soc < stopSOC && ~stopped
                        totalStops = totalStops + 1;
                        stopped=true; soc=stopSOC;
                    end
                end
            end
        end
    end
end

function events = buildEventsFromChargeLog(chargeLog,chosenEdges,OD,nextHop, ...
        dist,battery_kWh,kmPerUnit,speed_kmh)

    Nevents = numel(chargeLog);
    events = table((1:Nevents)', zeros(Nevents,1), zeros(Nevents,1), ...
                   zeros(Nevents,1), zeros(Nevents,1), ...
                   'VariableNames', {'evtID','stationIdx','energy_kWh_req', ...
                                     'arrival_hr','task'});
    stationKeys = cell(size(chosenEdges,1),1);
    for i=1:size(chosenEdges,1)
        stationKeys{i} = sprintf('%d-%d',chosenEdges(i,1),chosenEdges(i,2));
    end

    for e = 1:Nevents
        rec = chargeLog(e);
        key = sprintf('%d-%d',rec.station_u,rec.station_v);
        sidx = find(strcmp(stationKeys,key),1);
        if isempty(sidx), continue; end
        events.stationIdx(e)     = sidx;
        events.energy_kWh_req(e) = (rec.chargeSoC/100)*battery_kWh;
        events.task(e)           = rec.task;

        path = localPath(OD(rec.task,1),OD(rec.task,2),nextHop);
        arrivalMin = 0;
        for k=1:(numel(path)-1)
            u=path(k); v=path(k+1);
            edgeKm = dist(u,v)*kmPerUnit;
            dt_hr  = edgeKm/speed_kmh;
            arrivalMin = arrivalMin + dt_hr*60;
            if u==rec.station_u && v==rec.station_v, break; end
        end
        arrivalMin = arrivalMin + (rec.hour-1)*60 + rand()*60;
        events.arrival_hr(e) = arrivalMin;
    end
    events = events(events.stationIdx>0 & events.energy_kWh_req>0,:);
end

function [scoreProfit,estAvgWait,peakDemand_kW] = ...
    evaluatePriceScheduleProfitWithWaitEstimate(events,priceSchedule, ...
        gridCost,investPerStation,gamma,slotsPerDay,slotDur_min, ...
        numDC,DC_power,battery_kWh,gridCapacity_kW) %#ok<INUSD>

    K = size(priceSchedule,1);
    totalRevenue = 0; totalEnergy = 0;
    slotDur_hr   = slotDur_min/60;
    nEvents      = height(events);
    slotDemand_kWh = zeros(slotsPerDay,1);

    for i = 1:nEvents
        s = events.stationIdx(i);
        if s==0, continue; end
        slotIdx = min(slotsPerDay,max(1,ceil(events.arrival_hr(i)/slotDur_min)));
        eReq = events.energy_kWh_req(i);
        slotDemand_kWh(slotIdx) = slotDemand_kWh(slotIdx)+eReq;
        price_here = priceSchedule(s,slotIdx);
        totalRevenue = totalRevenue + price_here*eReq;
        totalEnergy  = totalEnergy  + eReq;
    end

    runningCost   = gamma*investPerStation*(slotsPerDay*slotDur_hr);
    totalGridCost = gridCost*totalEnergy;
    scoreProfit   = totalRevenue - totalGridCost - runningCost;

    slotDemand_kW = slotDemand_kWh / slotDur_hr;
    peakDemand_kW = max(slotDemand_kW);

    chargerCap_kW  = numDC*DC_power*K;
    effCap_kW      = min(chargerCap_kW,gridCapacity_kW);
    capPerSlot_kWh = effCap_kW*slotDur_hr;

    shortage = max(0,slotDemand_kWh - capPerSlot_kWh);
    if numDC*DC_power > 0
        estDelayPerSlot = shortage ./ (numDC*DC_power*K) * 60; % minutes
    else
        estDelayPerSlot = shortage / 11.2 * 60;
    end
    estAvgWait = mean(estDelayPerSlot);
end

function [scheduled,stationReport,slotLoad_kW,stationSlotLoad_kW] = ...
    scheduleEvents_with_dynamic(events,priceSchedule,K, ...
        numAC,numDC,DC_power,battery_kWh,slotDur_min,slotsPerDay, ...
        full_charge_minutes,mode,lambdaPrice)

    nEvents = height(events);
    scheduled = table(events.evtID,events.stationIdx,events.energy_kWh_req, ...
        events.arrival_hr, nan(nEvents,1), nan(nEvents,1), ...
        repmat({''},nEvents,1), zeros(nEvents,1), ...
        'VariableNames',{'evtID','stationIdx','energy_kWh_req', ...
                         'arrival_hr','start_hr','end_hr', ...
                         'chargerType','energy_kWh_delivered'});

    totalChgPerStation = max(1,numAC+numDC);
    stationChargerNextFree = cell(K,1);
    stationChargerType     = cell(K,1);
    for s = 1:K
        stationChargerNextFree{s} = zeros(totalChgPerStation,1);
        if numAC+numDC==0
            stationChargerType{s} = {'DC'};
        else
            types = [repmat({'AC'},numAC,1); repmat({'DC'},numDC,1)];
            if isempty(types), types = {'DC'}; end
            stationChargerType{s} = types;
        end
    end

    stationSlotLoad_kW = zeros(slotsPerDay,K);

    % process events in chronological order
    [~,ord] = sort(events.arrival_hr);
    for idx = ord'
        reqEnergy  = events.energy_kWh_req(idx);
        arrivalMin = events.arrival_hr(idx);
        if reqEnergy <= 0
            continue;
        end

        if strcmp(mode,'dynamic') && lambdaPrice > 0
            % ---------- Dynamic: choose best station+charger ----------
            bestCost   = Inf;
            bestStation= 0;
            bestChIdx  = 0;
            bestStart  = NaN;
            bestEnd    = NaN;
            bestType   = '';

            for s = 1:K
                for j = 1:length(stationChargerNextFree{s})
                    cType   = stationChargerType{s}{j};
                    start_j = max(arrivalMin, stationChargerNextFree{s}(j));

                    frac = min(1, reqEnergy/battery_kWh);
                    t_full_min = full_charge_minutes * frac;

                    if strcmp(cType,'DC'), maxPower = DC_power;
                    else, maxPower = 11.2; end

                    if t_full_min <= 0
                        avgPower   = maxPower;
                        t_full_min = (reqEnergy/avgPower)*60;
                    else
                        avgPower = reqEnergy/(t_full_min/60);
                        if avgPower > maxPower
                            avgPower   = maxPower;
                            t_full_min = (reqEnergy/avgPower)*60;
                        end
                    end

                    end_j = start_j + t_full_min;
                    wait_j = max(0, start_j - arrivalMin);

                    slotIdx = max(1,min(slotsPerDay,ceil(start_j/slotDur_min)));
                    price_here = priceSchedule(s,slotIdx);

                    cost = wait_j + lambdaPrice * price_here;

                    if cost < bestCost
                        bestCost   = cost;
                        bestStation= s;
                        bestChIdx  = j;
                        bestStart  = start_j;
                        bestEnd    = end_j;
                        bestType   = cType;
                    end
                end
            end

            if bestStation==0
                continue;
            end
            s = bestStation; chIdx = bestChIdx;
            startTime = bestStart; endTime = bestEnd; cType = bestType;
            events.stationIdx(idx) = s;  % update actual chosen station
        else
            % ---------- Non-dynamic: use pre-assigned station ----------
            s = events.stationIdx(idx);
            if s==0, continue; end

            bestDelivered = 0;
            bestIdx = 0; bestStart = NaN; bestEnd = NaN; bestType = '';
            for j = 1:length(stationChargerNextFree{s})
                cType   = stationChargerType{s}{j};
                start_j = max(arrivalMin, stationChargerNextFree{s}(j));

                frac      = min(1, reqEnergy/battery_kWh);
                t_full_min= full_charge_minutes * frac;

                if strcmp(cType,'DC'), maxPower = DC_power;
                else, maxPower = 11.2; end

                if t_full_min <= 0
                    avgPower   = maxPower;
                    t_full_min = (reqEnergy/avgPower)*60;
                else
                    avgPower = reqEnergy/(t_full_min/60);
                    if avgPower > maxPower
                        avgPower   = maxPower;
                        t_full_min = (reqEnergy/avgPower)*60;
                    end
                end
                end_min  = start_j + t_full_min;
                delivered = reqEnergy;

                if delivered>bestDelivered || (abs(delivered-bestDelivered)<1e-9 && ...
                                              (isnan(bestStart)||start_j<bestStart))
                    bestDelivered = delivered;
                    bestIdx  = j;
                    bestStart= start_j;
                    bestEnd  = end_min;
                    bestType = cType;
                end
            end
            if bestIdx==0, continue; end
            chIdx     = bestIdx;
            startTime = bestStart;
            endTime   = bestEnd;
            cType     = bestType;
        end

        % record event schedule
        scheduled.start_hr(idx)            = startTime;
        scheduled.end_hr(idx)              = endTime;
        scheduled.chargerType{idx}         = cType;
        scheduled.energy_kWh_delivered(idx)= reqEnergy;
        stationChargerNextFree{s}(chIdx)   = endTime;

        % add to slot load
        startSlot = max(1,min(slotsPerDay,ceil(startTime/slotDur_min)));
        endSlot   = max(1,min(slotsPerDay,ceil(endTime  /slotDur_min)));
        duration_hr = (endTime-startTime)/60;
        if duration_hr > 0
            avgPower = reqEnergy / duration_hr;
            stationSlotLoad_kW(startSlot:endSlot,s) = ...
                stationSlotLoad_kW(startSlot:endSlot,s) + avgPower;
        end
    end

    slotLoad_kW = sum(stationSlotLoad_kW,2);

    % station-level report (counts & energy; revenue added later)
    stationReport = table((1:K)', zeros(K,1), zeros(K,1), zeros(K,1), ...
        zeros(K,1), zeros(K,1), ...
        'VariableNames',{'Station','EVs_FullyServed','EVs_PartiallyServed', ...
                         'EVs_NotServed','EnergySold_kWh','TotalChargeTime_hr'});

    for s = 1:K
        mask = scheduled.stationIdx == s;
        if ~any(mask), continue; end
        reqE = scheduled.energy_kWh_req(mask);
        gotE = scheduled.energy_kWh_delivered(mask);
        validIdx = ~isnan(gotE);
        stationReport.EVs_FullyServed(s) = sum(abs(gotE(validIdx)-reqE(validIdx))<1e-6);
        stationReport.EVs_PartiallyServed(s) = sum(gotE>0 & gotE<reqE-1e-6);
        stationReport.EVs_NotServed(s)   = sum(gotE==0 | isnan(gotE));
        stationReport.EnergySold_kWh(s)  = sum(gotE(validIdx),'omitnan');
        dur_min = scheduled.end_hr(mask) - scheduled.start_hr(mask);
        dur_min = dur_min(~isnan(dur_min));
        stationReport.TotalChargeTime_hr(s) = sum(dur_min)/60;
    end
end
