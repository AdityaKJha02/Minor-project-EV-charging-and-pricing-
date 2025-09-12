%% EV Charging Station Placement & Simulation (Greedy, avoids dead stations)
% Full end-to-end script. Paste into a new .m file and run.

clc; clear; close all;
rng('shuffle');

%% -------------------- USER TUNABLE PARAMETERS --------------------
numEVs = 10;           % EVs simulated per task
minSOC = 60;            % initial SoC lower bound (%)
maxSOC = 85;            % initial SoC upper bound (%)
consRate = 1;           % %SoC consumed per unit distance (distance units as in matrix)
chargeThreshold = 55;   % EVs < this % are eligible for charging
stopSOC = 20;           % EV stops if SoC < stopSOC
numStations = 6;        % number of stations to place
penalty = 6;            % weight for stopped EVs in greedy objective (tuneable)
maxGreedyIter = 1000;   % safety cap for greedy loops
% ----------------------------------------------------------------

%% ---------- 1) Read & sanitize distance matrix ----------
assumeUndirected = true;
zeroMeansNoEdge  = true;

M = readmatrix('distance_matrix.xlsx');
M = M(~all(isnan(M),2), ~all(isnan(M),1));
M(isnan(M)) = Inf;
n = size(M,1);
M(1:n+1:end) = 0;
if zeroMeansNoEdge
    offdiag = true(n); offdiag(1:n+1:end) = false;
    M(offdiag & M==0) = Inf;
end
if assumeUndirected
    M = min(M, M.');
    M(1:n+1:end) = 0;
end

%% ---------- 2) Floyd–Warshall w/ path reconstruction ----------
dist = M;
nextHop = zeros(n);

for i = 1:n
    for j = 1:n
        if i~=j && isfinite(dist(i,j))
            nextHop(i,j) = j;
        end
    end
end

for k = 1:n
    dik = dist(:,k);
    skj = dist(k,:);
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

% local path function defined at end (localPath)

%% ---------- 3) Read OD pairs from Word ----------
txt = extractFileText('OD nodes.docx');
pairs = regexp(txt,'\((\d+)\s*,\s*(\d+)\)','tokens');
OD = cellfun(@(c)[str2double(c{1}) str2double(c{2})], pairs, 'UniformOutput', false);
OD = vertcat(OD{:});

maxNode = max(OD(:));
if maxNode > n
    error('OD references node %d but matrix is only %dx%d.', maxNode, n, n);
end

%% ---------- 4) Build candidate edge list from OD paths ----------
candidateEdges = [];
for t = 1:size(OD,1)
    p = localPath(OD(t,1), OD(t,2), nextHop);
    if numel(p) > 1
        edges = [p(1:end-1)' p(2:end)'];
        candidateEdges = [candidateEdges; edges]; %#ok<AGROW>
    end
end
candidateEdges = unique(candidateEdges,'rows');
m = size(candidateEdges,1);
if m == 0
    error('No edges found from OD paths - nothing to place stations on.');
end
fprintf('Found %d candidate edges from tasks.\n', m);

%% ---------- 5) Greedy iterative placement avoiding dead stations ----------
% We'll iteratively pick one station at a time. At each step test all remaining
% candidates and pick the one that yields the best objective:
%   objective = totalChargeSold - penalty * totalStoppedEVs
% After selecting an edge, we permanently add it and continue until we select
% numStations or no candidate gives positive marginal improvement.

chosenEdges = zeros(0,2);
remainingEdges = candidateEdges;
selectedFlags = false(m,1);

% For speed, precompute per-task path lists
taskPaths = cell(size(OD,1),1);
for t = 1:size(OD,1)
    taskPaths{t} = localPath(OD(t,1), OD(t,2), nextHop);
end

iter = 0;
while size(chosenEdges,1) < numStations && ~isempty(remainingEdges) && iter < maxGreedyIter
    iter = iter + 1;
    bestEdge = [];
    bestScore = -Inf;
    bestMetrics = [];
    % Test each remaining edge (simulate with chosenEdges + candidate)
    for eidx = 1:size(remainingEdges,1)
        candEdge = remainingEdges(eidx,:);
        trialStations = [chosenEdges; candEdge];
        [trialCharge, trialStops] = simulateWithStationsSimple(trialStations, taskPaths, OD, ...
            dist, consRate, numEVs, minSOC, maxSOC, chargeThreshold, stopSOC);
        score = trialCharge - penalty * trialStops;
        % prefer edges that also serve at least one EV (we'll ensure later)
        if score > bestScore
            bestScore = score;
            bestEdge = candEdge;
            bestMetrics = [trialCharge, trialStops]; %#ok<AGROW>
        end
    end

    % If bestEdge yields no improvement over having no extra station (i.e., if adding it doesn't increase objective),
    % we stop early.
    % To check baseline objective with current chosenEdges:
    [baseCharge, baseStops] = simulateWithStationsSimple(chosenEdges, taskPaths, OD, ...
            dist, consRate, numEVs, minSOC, maxSOC, chargeThreshold, stopSOC);
    baseScore = baseCharge - penalty * baseStops;
    if isempty(bestEdge)
        break;
    end
    if bestScore <= baseScore
        fprintf('No candidate provides improvement at iteration %d. Stopping selection.\n', iter);
        break;
    end

    % Add bestEdge permanently
    chosenEdges = [chosenEdges; bestEdge];
    % remove that edge from remainingEdges
    remainingEdges = setdiff(remainingEdges, bestEdge, 'rows','stable');
    fprintf('Selected station %d: edge %d->%d | trialCharge=%.2f trialStops=%d score=%.2f\n', ...
        size(chosenEdges,1), bestEdge(1), bestEdge(2), bestMetrics(1), bestMetrics(2), bestScore);
end

% If fewer than desired stations chosen, warn
if size(chosenEdges,1) < numStations
    fprintf('Note: only %d stations selected (requested %d). No beneficial candidates remain.\n', ...
        size(chosenEdges,1), numStations);
end

%% ---------- 6) Final full simulation with chosenEdges (detailed logs) ----------
[totalChargeFinal, totalStoppedFinal, stationCount, stationCharge, chargeLog] = ...
    simulateWithStationsFull(chosenEdges, taskPaths, OD, dist, consRate, numEVs, minSOC, maxSOC, chargeThreshold, stopSOC);

%% ---------- 7) Clean-up: remove any dead stations (should be rare with greedy) ----------
deadIdx = find(stationCount==0);
if ~isempty(deadIdx)
    fprintf('\nRemoving %d dead stations (no EVs served):\n', numel(deadIdx));
    for i = deadIdx'
        fprintf('  Dead station at edge %d->%d\n', chosenEdges(i,1), chosenEdges(i,2));
    end
    % Remove them
    chosenEdges(deadIdx,:) = [];
    stationCount(deadIdx) = [];
    stationCharge(deadIdx) = [];
end
%% ---------- 8) Reporting ----------
fprintf('\n--- Final Simulation Report (Greedy selection, no dead stations) ---\n');
for s = 1:size(chosenEdges,1)
    fprintf('Station %d at edge %d->%d: EVs served = %d | Charge sold (SoC-pts) = %.2f\n', ...
        s, chosenEdges(s,1), chosenEdges(s,2), stationCount(s), stationCharge(s));
end
fprintf('Total EVs served = %d\n', sum(stationCount));
fprintf('Total charge sold (SoC-pts) = %.2f\n', totalChargeFinal);
fprintf('Total EVs stopped (SoC < %.1f%%) = %d\n', stopSOC, totalStoppedFinal);

% Save detailed charge dataset
% chargeLog is a table-like struct array: fields task, evID, station_u, station_v, chargeSoC
% Convert to table for convenience
if ~isempty(chargeLog)
    T = struct2table(chargeLog);
    writetable(T,'charge_log.csv');
    fprintf('Detailed charge log saved to charge_log.csv (rows = charge events).\n');
else
    fprintf('No charging events recorded; charge log is empty.\n');
end

%% ------------------- Local helper functions -------------------

function path = localPath(u,v,nextHop)
    % returns node sequence from u to v using nextHop table
    if u==v
        path = u;
        return;
    end
    if nextHop(u,v)==0
        path = [];
        return;
    end
    path = u;
    while u ~= v
        u = nextHop(u,v);
        if u==0, path = []; return; end
        path(end+1) = u; %#ok<AGROW>
    end
end

function [totalCharge, totalStops] = simulateWithStationsSimple(stationEdges, taskPaths, OD, ...
        dist, consRate, numEVs, minSOC, maxSOC, chargeThreshold, stopSOC)
    % Lightweight simulator used inside greedy loop. Returns only aggregate metrics.
    % stationEdges: Kx2 array of edges [u v]
    totalCharge = 0;
    totalStops = 0;
    K = size(stationEdges,1);

    % Precompute keys for station edges
    keys = cell(K,1);
    for ii=1:K
        keys{ii} = sprintf('%d-%d', stationEdges(ii,1), stationEdges(ii,2));
    end

    for t = 1:length(taskPaths)
        p = taskPaths{t};
        if isempty(p), continue; end

        % initialize EV SoCs for this task
        socEVs = minSOC + (maxSOC-minSOC).*rand(numEVs,1);
        % track whether each EV has already stopped
        hasStopped = false(numEVs,1);

        for k = 1:(numel(p)-1)
            u = p(k); v = p(k+1);
            d = dist(u,v);
            key = sprintf('%d-%d', u,v);

            % If station present
            idx = find(strcmp(keys, key));
            if ~isempty(idx)
                % compute remaining distance from here to finish
                distRem = 0;
                for mm = k:(numel(p)-1)
                    distRem = distRem + dist(p(mm), p(mm+1));
                end
                neededSOC = distRem * consRate;

                eligible = socEVs < chargeThreshold;
                chargeDeliver = zeros(numEVs,1);
                chargeDeliver(eligible) = max(0, neededSOC - socEVs(eligible));

                totalCharge = totalCharge + sum(chargeDeliver);
                socEVs(eligible) = socEVs(eligible) + chargeDeliver(eligible);
            end

            % travel consumption
            socEVs = socEVs - d*consRate;

            % detect new stops (unique per EV)
            newlyStopped = (socEVs < stopSOC) & ~hasStopped;
            totalStops = totalStops + sum(newlyStopped);

            % mark them as stopped permanently
            hasStopped(newlyStopped) = true;

            % clamp SoC of stopped EVs to stopSOC
            socEVs(hasStopped) = stopSOC;
        end
    end
end


function [totalCharge, totalStops, stationCount, stationCharge, chargeLog] = ...
        simulateWithStationsFull(stationEdges, taskPaths, OD, dist, consRate, numEVs, minSOC, maxSOC, chargeThreshold, stopSOC)
    % Full simulator that returns per-station counts and a detailed charge log.
    K = size(stationEdges,1);
    stationCount = zeros(K,1);
    stationCharge = zeros(K,1);
    totalCharge = 0;
    totalStops = 0;
    chargeLog = struct('task',{},'evID',{},'station_u',{},'station_v',{},'chargeSoC',{});
    keys = cell(K,1);
    for ii=1:K, keys{ii} = sprintf('%d-%d', stationEdges(ii,1), stationEdges(ii,2)); end

    for t = 1:length(taskPaths)
        p = taskPaths{t};
        if isempty(p), continue; end

        % initialize EV SoCs for this task
        socEVs = minSOC + (maxSOC-minSOC).*rand(numEVs,1);

        for k = 1:(numel(p)-1)
            u = p(k); v = p(k+1);
            d = dist(u,v);
            key = sprintf('%d-%d', u,v);
            % station at this edge?
            idx = find(strcmp(keys, key));
            if ~isempty(idx)
                % compute remaining distance from here to finish
                distRem = 0;
                for mm = k:(numel(p)-1)
                    distRem = distRem + dist(p(mm), p(mm+1));
                end
                neededSOC = distRem * consRate;
                eligible = socEVs < chargeThreshold;

                % charge delivered just enough to finish the task
                chargeDeliver = zeros(numEVs,1);
                chargeDeliver(eligible) = max(0, neededSOC - socEVs(eligible));

                served = find(chargeDeliver > 0);
                if ~isempty(served)
                    stationCount(idx) = stationCount(idx) + numel(served);
                    stationCharge(idx) = stationCharge(idx) + sum(chargeDeliver(served));
                    totalCharge = totalCharge + sum(chargeDeliver(served));
                    % update socs
                    socEVs(served) = socEVs(served) + chargeDeliver(served);
                    % record each charge event in chargeLog
                    for evi = 1:numel(served)
                        evID = served(evi);
                        rec.task = t;
                        rec.evID = evID;
                        rec.station_u = stationEdges(idx,1);
                        rec.station_v = stationEdges(idx,2);
                        rec.chargeSoC = chargeDeliver(evID);
                        chargeLog(end+1) = rec; %#ok<AGROW>
                    end
                end
            end

            % travel edge
            socEVs = socEVs - d*consRate;

            % stopped EVs
            stopped = find(socEVs < stopSOC);
            if ~isempty(stopped)
                totalStops = totalStops + numel(stopped);
                % clamp them so they do not continue
                socEVs(stopped) = stopSOC;
            end
        end
    end
end

%% ------------------ Charging Infrastructure Simulation with Arrival Times ------------------
% Parameters
battery_kWh = 59;       % Mahindra BE 6C battery capacity (kWh)
range_km    = 557;      % range for full charge
AC_power    = 11.2;     % kW per AC charger
DC_power    = 140;      % kW per DC charger
numAC       = 10;       % AC chargers per station
numDC       = 1;        % DC chargers per station
timeWindow_hr = 1.0;    % 1 hour horizon
kmPerUnit   = 2;      % each graph unit = 1 km
speed_kmh   = 180;      % EV driving speed

% Precompute consumption rate in kWh per km
energyPerKm = battery_kWh / range_km;   % ~0.106 kWh/km

% Reconstruct detailed arrival schedule for chargeLog
Nevents = numel(chargeLog);
events = table((1:Nevents)', zeros(Nevents,1), zeros(Nevents,1), ...
    zeros(Nevents,1), zeros(Nevents,1), ...
    'VariableNames', {'evtID','stationIdx','energy_kWh_req','arrival_hr','reqDuration_hr'});

stationKeys = cell(size(chosenEdges,1),1);
for i=1:size(chosenEdges,1)
    stationKeys{i} = sprintf('%d-%d', chosenEdges(i,1), chosenEdges(i,2));
end

for e = 1:Nevents
    rec = chargeLog(e);
    key = sprintf('%d-%d', rec.station_u, rec.station_v);
    sidx = find(strcmp(stationKeys,key),1);
    if isempty(sidx), continue; end

    % Required energy
    energy_kWh = (rec.chargeSoC/100) * battery_kWh;
    events.stationIdx(e) = sidx;
    events.energy_kWh_req(e) = energy_kWh;

    % Arrival time: based on path up to this station
    path = localPath(OD(rec.task,1), OD(rec.task,2), nextHop);
    arrivalTime = 0;
    for k = 1:(numel(path)-1)
        u = path(k); v = path(k+1);
        edgeKm = dist(u,v)*kmPerUnit;
        dt = edgeKm/speed_kmh; % hours
        arrivalTime = arrivalTime + dt;
        if u==rec.station_u && v==rec.station_v
            break; % reached charging edge
        end
    end
    events.arrival_hr(e) = arrivalTime;
end

% Sort events by arrival time
events = sortrows(events,'arrival_hr');

% Initialize charger availability
K = size(chosenEdges,1);
stationChargerNextFree = cell(K,1);
for s = 1:K
    stationChargerNextFree{s} = zeros(numAC+numDC,1); % all free at t=0
end

% Scheduling table
scheduled = table(events.evtID, events.stationIdx, events.energy_kWh_req, ...
    events.arrival_hr, zeros(height(events),1), zeros(height(events),1), ...
    strings(height(events),1), zeros(height(events),1), ...
    'VariableNames', {'evtID','stationIdx','energy_kWh_req','arrival_hr','start_hr','end_hr','chargerType','energy_kWh_delivered'});

for i=1:height(events)
    s = events.stationIdx(i);
    if s==0, continue; end

    % Check chargers at this station
    nextFree = stationChargerNextFree{s};
    if isempty(nextFree), continue; end

    % Assign charger type: >10 kWh → DC else AC
    if events.energy_kWh_req(i) > 10
        [startT, localIdx] = min(nextFree(numAC+1:end));
        chargerType = "DC";
        power = DC_power;
        globalIdx = numAC+localIdx;
    else
        [startT, localIdx] = min(nextFree(1:numAC));
        chargerType = "AC";
        power = AC_power;
        globalIdx = localIdx;
    end

    % Start time = max(arrival, charger availability)
    start_hr = max(events.arrival_hr(i), startT);
    dur_hr   = events.energy_kWh_req(i)/power;
    end_hr   = start_hr + dur_hr;

    % Delivered energy within time window
    if start_hr >= timeWindow_hr
        delivered = 0; end_hr = NaN;
    else
        actualEnd = min(end_hr, timeWindow_hr);
        delivered = power*(actualEnd-start_hr);
        end_hr = actualEnd;
    end

    % Save
    scheduled.start_hr(i) = start_hr;
    scheduled.end_hr(i) = end_hr;
    scheduled.chargerType(i) = chargerType;
    scheduled.energy_kWh_delivered(i) = delivered;

    % Update charger availability
    stationChargerNextFree{s}(globalIdx) = end_hr;
end

% Add charging time column
scheduled.chargeTime_min = (scheduled.end_hr - scheduled.start_hr)*60;

% Save to CSV
writetable(scheduled,'charger_schedule_with_arrival.csv');
fprintf('Detailed schedule saved to charger_schedule_with_arrival.csv (with arrival times).\n');

%% ------------------ Corrected arrival-aware scheduling + per-station report ------------------
% Assumes events table already computed with fields:
%   events.stationIdx, events.energy_kWh_req, events.arrival_hr
% (if you used a different name, adjust accordingly)

% Remove events that are not mapped to any chosen station
events = events(events.stationIdx>0,:);

if isempty(events)
    warning('No charging events with valid station indices. Aborting scheduling.');
else
    % --- 1) Set simulation horizon dynamically ---
    % default: at least 1 hour, or extend to cover last arrival + 1 hour buffer
    postArrivalBuffer_hr = 0.5; % you can change this
    timeWindow_hr = max(1.0, max(events.arrival_hr) + postArrivalBuffer_hr);
    fprintf('Scheduling horizon (timeWindow_hr) = %.2f hr (max arrival = %.2f hr)\n', timeWindow_hr, max(events.arrival_hr));

    % --- 2) Setup chargers per station (cell arrays) ---
    K = size(chosenEdges,1);
    totalChgPerStation = numAC + numDC;
    stationChargerNextFree = cell(K,1);
    stationChargerPower = cell(K,1);
    for s = 1:K
        stationChargerNextFree{s} = zeros(totalChgPerStation,1);                        % all free at t=0
        stationChargerPower{s} = [repmat(AC_power, numAC, 1); repmat(DC_power, numDC, 1)];
    end

    % --- 3) Sort events by arrival time (earliest first) for realistic processing ---
    events = sortrows(events,'arrival_hr');

    % --- 4) Prepare scheduled table (chargerType as cellstr) ---
    nEvents = height(events);
    scheduled = table(events.evtID, events.stationIdx, events.energy_kWh_req, events.arrival_hr, ...
        nan(nEvents,1), nan(nEvents,1), repmat({''},nEvents,1), zeros(nEvents,1), ...
        'VariableNames', {'evtID','stationIdx','energy_kWh_req','arrival_hr','start_hr','end_hr','chargerType','energy_kWh_delivered'});

    % --- 5) Schedule each event: pick charger that gives max delivered energy within horizon ---
    for i = 1:nEvents
        s = events.stationIdx(i);
        reqEnergy = events.energy_kWh_req(i);
        arrivalT = events.arrival_hr(i);

        % skip if no chargers at station
        powers = stationChargerPower{s};
        nextFree = stationChargerNextFree{s};
        if isempty(powers), continue; end

        bestDelivered = 0;
        bestIdx = 0;
        bestStart = NaN;
        bestEnd = NaN;
        bestType = '';

        % evaluate each charger at station
        for j = 1:numel(powers)
            power_j = powers(j);
            % earliest possible start on this charger
            start_j = max(arrivalT, nextFree(j));
            if start_j >= timeWindow_hr
                % charger unavailable within horizon — cannot deliver anything
                continue;
            end
            % full duration to satisfy this event on this charger (hours)
            dur_full = reqEnergy / power_j;
            end_full = start_j + dur_full;
            % delivered within horizon (cap at timeWindow_hr)
            actual_end = min(end_full, timeWindow_hr);
            delivered_j = max(0, power_j * (actual_end - start_j));
            delivered_j = min(delivered_j, reqEnergy);

            % choose charger maximizing delivered energy (tie-breaker: earliest start, then higher power)
            if delivered_j > bestDelivered || ...
               (abs(delivered_j-bestDelivered) < 1e-9 && (start_j < bestStart || (start_j==bestStart && power_j > (bestIdx>0)*powers(bestIdx))))
                bestDelivered = delivered_j;
                bestIdx = j;
                bestStart = start_j;
                bestEnd = actual_end;
                if j <= numAC
                    bestType = 'AC';
                else
                    bestType = 'DC';
                end
                bestFullEnd = end_full; % real completion time (may exceed horizon)
            end
        end

        % If nothing deliverable within horizon -> not served
        if bestIdx == 0 || bestDelivered <= 0
            scheduled.start_hr(i) = NaN;
            scheduled.end_hr(i) = NaN;
            scheduled.chargerType{i} = '';
            scheduled.energy_kWh_delivered(i) = 0;
            continue;
        end

        % Assign event to chosen charger
        scheduled.start_hr(i) = bestStart;
        scheduled.end_hr(i)   = bestEnd;   % note: <= timeWindow_hr
        scheduled.chargerType{i} = bestType;
        scheduled.energy_kWh_delivered(i) = bestDelivered;

        % Update charger busy-until to the *real* completion time (prevents double-booking across arrival times)
        stationChargerNextFree{s}(bestIdx) = bestFullEnd;
    end

    % --- 6) Post-process scheduling results: compute times and per-station metrics ---
    % Clean up chargeTime: convert NaNs to 0 for summations
    chargeTime_hr = scheduled.end_hr - scheduled.start_hr;
    chargeTime_hr(~isfinite(chargeTime_hr) | chargeTime_hr < 0) = 0;
    scheduled.chargeTime_hr = chargeTime_hr;
    scheduled.chargeTime_min = chargeTime_hr * 60;

    % Per-station report
    stationReport = table((1:K)', zeros(K,1), zeros(K,1), zeros(K,1), ...
        zeros(K,1), zeros(K,1), zeros(K,1), ...
        'VariableNames', {'Station','EVs_FullyServed','EVs_PartiallyServed','EVs_NotServed', ...
                          'EnergySold_kWh','TotalChargeTime_hr','AvgChargeTime_min'});

    for s = 1:K
        mask = scheduled.stationIdx==s;
        if ~any(mask), continue; end

        reqE = scheduled.energy_kWh_req(mask);
        gotE = scheduled.energy_kWh_delivered(mask);
        tmin = scheduled.chargeTime_min(mask);

        stationReport.EVs_FullyServed(s) = sum( abs(gotE - reqE) < 1e-6 );
        stationReport.EVs_PartiallyServed(s) = sum( gotE > 0 & gotE < reqE - 1e-6 );
        stationReport.EVs_NotServed(s) = sum( gotE == 0 );

        stationReport.EnergySold_kWh(s) = nansum(gotE);
        stationReport.TotalChargeTime_hr(s) = nansum(tmin)/60;
        if sum(gotE>0) > 0
            stationReport.AvgChargeTime_min(s) = nanmean(tmin(gotE>0));
        else
            stationReport.AvgChargeTime_min(s) = 0;
        end
    end

    % --- 7) Print & save ---
    fprintf('\n=== Station-wise Service Report (horizon = %.2f hr) ===\n', timeWindow_hr);
    for s = 1:K
        fprintf(['Station %d (edge %d->%d): Fully = %d | Partially = %d | Not = %d | ' ...
                 'Energy sold = %.2f kWh | Total time = %.2f hr | Avg time = %.1f min/EV\n'], ...
            s, chosenEdges(s,1), chosenEdges(s,2), ...
            stationReport.EVs_FullyServed(s), stationReport.EVs_PartiallyServed(s), stationReport.EVs_NotServed(s), ...
            stationReport.EnergySold_kWh(s), stationReport.TotalChargeTime_hr(s), stationReport.AvgChargeTime_min(s));
    end

    writetable(stationReport,'station_service_report.csv');
    writetable(scheduled,'charger_schedule_with_arrival.csv'); % includes arrival & charge times
    fprintf('Saved station_service_report.csv and charger_schedule_with_arrival.csv\n');
end

%% ------------------ Station-to-Bus Mapping (Iterative Greedy) ------------------
% Load IEEE69 bus load profile
baseLoad = readmatrix('IEEE69_24h active data.xlsx');
if size(baseLoad,1) < size(baseLoad,2)
    baseLoad = baseLoad.'; % ensure rows = buses, cols = hours
end
baseLoad(isnan(baseLoad)) = 0;

% Focus on time window (12–2 pm = hours 12 and 13)
baseLoad_win = baseLoad(:,12:13); % [69 x 2]
[nbus, H] = size(baseLoad_win);

K = size(chosenEdges,1); % number of stations

% Charger load profile (constant across hours)
stationLoadProfiles = repmat([numAC*AC_power + numDC*DC_power, numAC*AC_power + numDC*DC_power], K, 1);
stationLoadProfiles_MW = stationLoadProfiles / 1000; % convert to MW

% Exclude generator buses = rows where all hours are zero
validBuses = find(any(baseLoad_win>0,2));
fprintf('Valid buses (non-generator) = %d out of %d\n', numel(validBuses), nbus);

% Initialize assignment
stationToBus = zeros(K,1);
usedBuses = false(nbus,1);

for s = 1:K
    bestBus = -1; bestScore = Inf;
    for b = validBuses(:)'
        if usedBuses(b), continue; end % already taken
        % Compute disturbance metric: variance increase
        baseProf = baseLoad_win(b,:);
        newProf = baseProf + stationLoadProfiles_MW(s,:);
        score = var(newProf) - var(baseProf); % how much instability added
        if score < bestScore
            bestScore = score;
            bestBus = b;
        end
    end
    if bestBus==-1
        warning('No available bus found for station %d, skipping.', s);
    else
        stationToBus(s) = bestBus;
        usedBuses(bestBus) = true;
    end
end

%% ----------- Reporting ----------
fprintf('\n=== Station-to-Bus Mapping Report (Greedy Iterative) ===\n');
for s = 1:K
    b = stationToBus(s);
    if b==0, continue; end
    base = baseLoad_win(b,:);
    new  = base + stationLoadProfiles_MW(s,:);
    fprintf('Station %d at edge %d->%d → Bus %d | BaseLoad=%s MW | Updated=%s MW\n',...
        s, chosenEdges(s,1), chosenEdges(s,2), b, mat2str(base,3), mat2str(new,3));
end

% Save mapping
Tmap = table((1:K)', chosenEdges(:,1), chosenEdges(:,2), stationToBus, ...
    'VariableNames', {'Station','FromNode','ToNode','AssignedBus'});
writetable(Tmap,'station_bus_mapping.csv');
fprintf('Saved mapping to station_bus_mapping.csv\n');

%% ------------------ Dynamic Pricing & EV Assignment with Bounds, Rerouting & Reports ------------------

% --- Parameters (reuse existing if already defined earlier) ---
if ~exist('battery_kWh','var'); battery_kWh = 59; end
if ~exist('range_km','var');    range_km = 557; end
if ~exist('AC_power','var');    AC_power = 11.2; end
if ~exist('DC_power','var');    DC_power = 140; end
if ~exist('numAC','var');       numAC = 10; end
if ~exist('numDC','var');       numDC = 1; end
if ~exist('kmPerUnit','var');   kmPerUnit = 2; end
if ~exist('speed_kmh','var');   speed_kmh = 180; end
if ~exist('efficiencyFactor','var'); efficiencyFactor = 0.9; end
if ~exist('gamma','var');       gamma = 0.01; end
if ~exist('plugCostAC','var');  plugCostAC = 2000; end
if ~exist('plugCostDC','var');  plugCostDC = 10000; end
if ~exist('gridBasePrice','var'); gridBasePrice = 8.5; end

% >>> New user-defined station price bounds <<<
minPrice_kWh = 10;   % $/kWh
maxPrice_kWh = 15;   % $/kWh

investmentPerStation = numAC*plugCostAC + numDC*plugCostDC;

% --- Build events table (with task IDs) ---
Nevents = numel(chargeLog);
events = table((1:Nevents)', zeros(Nevents,1), zeros(Nevents,1), ...
               zeros(Nevents,1), zeros(Nevents,1), zeros(Nevents,1), ...
    'VariableNames', {'evtID','stationIdx','energy_kWh_req','arrival_hr','reqDuration_hr','task'});
stationKeys = cell(size(chosenEdges,1),1);
for i=1:size(chosenEdges,1)
    stationKeys{i} = sprintf('%d-%d', chosenEdges(i,1), chosenEdges(i,2));
end
for e = 1:Nevents
    rec = chargeLog(e);
    key = sprintf('%d-%d', rec.station_u, rec.station_v);
    sidx = find(strcmp(stationKeys,key),1);
    if isempty(sidx), continue; end
    energy_kWh = (rec.chargeSoC/100) * battery_kWh;
    events.stationIdx(e)     = sidx;
    events.energy_kWh_req(e) = energy_kWh;
    events.task(e)           = rec.task;
    path = localPath(OD(rec.task,1), OD(rec.task,2), nextHop);
    arrivalTime = 0;
    for k = 1:(numel(path)-1)
        u = path(k); v = path(k+1);
        edgeKm = dist(u,v) * kmPerUnit;
        dt = edgeKm / speed_kmh;
        arrivalTime = arrivalTime + dt;
        if u==rec.station_u && v==rec.station_v, break; end
    end
    events.arrival_hr(e) = arrivalTime;
end
events = events(events.stationIdx>0 & events.energy_kWh_req>0, :);
if isempty(events)
    warning('No valid events to run dynamic pricing/assignment. Aborting block.');
    return;
end

% --- Time discretization ---
slotDur = 15/60; % 15 min
timeWindow_hr = max(events.arrival_hr) + 0.5;
nSlots = max(1, ceil(timeWindow_hr / slotDur));
slotEdges = (0:nSlots) * slotDur;

% --- PSO optimization for station price schedules ---
K = size(chosenEdges,1);
nParticles = 30; nIter = 40;
particles = minPrice_kWh + (maxPrice_kWh-minPrice_kWh)*rand(K,nSlots,nParticles);
bestParticles = particles(:,:,1); bestScore = -Inf;

for it = 1:nIter
    for p = 1:nParticles
        cand = particles(:,:,p);
        totalProfit = 0;
        for s = 1:K
            for sl = 1:nSlots
                mask = events.arrival_hr >= slotEdges(sl) & events.arrival_hr < slotEdges(sl+1) & events.stationIdx==s;
                eSold = sum(events.energy_kWh_req(mask));
                price = cand(s,sl);
                revenue = price * eSold;
                gridCost = gridBasePrice * (eSold / efficiencyFactor);
                runningCost = gamma * investmentPerStation * slotDur;
                totalProfit = totalProfit + (revenue - gridCost - runningCost);
            end
        end
        if totalProfit > bestScore
            bestScore = totalProfit;
            bestParticles = cand;
        end
    end
    for p = 1:nParticles
        particles(:,:,p) = max(minPrice_kWh, min(maxPrice_kWh, ...
            particles(:,:,p) + 0.15*(bestParticles - particles(:,:,p)) + 0.02*randn(K,nSlots)));
    end
end
priceSchedule = bestParticles; % K x nSlots

% --- EV attraction & rerouting based on pricing ---
assignedStation = zeros(height(events),1);
rerouted = false(height(events),1);
for ee = 1:height(events)
    taskID = events.task(ee);
    basePath = localPath(OD(taskID,1), OD(taskID,2), nextHop);
    candStations = unique(events.stationIdx); bestScore = -Inf; chosen = events.stationIdx(ee);
    for s = candStations'
        slotIdx = min(nSlots, max(1, ceil(events.arrival_hr(ee)/slotDur)));
        price = priceSchedule(s,slotIdx);
        distExtra = dist(OD(taskID,1), chosenEdges(s,1));
        waitPenalty = rand*5; % placeholder for queuing delay
        attraction = -(distExtra + waitPenalty) - price*10; % combined score
        if attraction > bestScore
            bestScore = attraction; chosen = s;
        end
    end
    assignedStation(ee) = chosen;
    rerouted(ee) = (chosen ~= events.stationIdx(ee));
end
events.assignedStation = assignedStation;
events.rerouted = rerouted;

% --- Station-level summary ---
stationReport = table((1:K)', zeros(K,1), zeros(K,1), zeros(K,1), ...
    zeros(K,1), zeros(K,1), zeros(K,1), ...
    'VariableNames', {'Station','ServedEVs','ReroutedEVs','EnergySold_kWh','Revenue','Profit','AvgChargeTime_min'});
for s = 1:K
    mask = events.assignedStation==s;
    if ~any(mask), continue; end
    evs = sum(mask); reRoute = sum(events.rerouted(mask));
    eSold = sum(events.energy_kWh_req(mask));
    rev = mean(priceSchedule(s,:)) * eSold;
    cost = gridBasePrice*(eSold/efficiencyFactor) + gamma*investmentPerStation;
    prof = rev - cost;
    stationReport.ServedEVs(s) = evs;
    stationReport.ReroutedEVs(s) = reRoute;
    stationReport.EnergySold_kWh(s) = eSold;
    stationReport.Revenue(s) = rev;
    stationReport.Profit(s) = prof;
    stationReport.AvgChargeTime_min(s) = (eSold/AC_power)/evs*60; % approx
end

fprintf('\n=== Station-wise Rerouting Report ===\n');
disp(stationReport);
writetable(stationReport,'station_rerouting_report.csv');

% --- EV user-level report ---
userReport = table(unique(events.task), zeros(numel(unique(events.task)),1), zeros(numel(unique(events.task)),1), ...
    zeros(numel(unique(events.task)),1), zeros(numel(unique(events.task)),1), ...
    'VariableNames', {'Task','AvgPrice','ExtraDist','WaitTime','DriveTime'});
utasks = unique(events.task);
for i = 1:numel(utasks)
    t = utasks(i); mask = events.task==t;
    userReport.AvgPrice(i) = mean(arrayfun(@(ee) ...
        priceSchedule(events.assignedStation(ee), min(nSlots,ceil(events.arrival_hr(ee)/slotDur))), find(mask)));
    userReport.ExtraDist(i) = mean(rand(sum(mask),1)*2); % placeholder extra distance
    userReport.WaitTime(i) = mean(rand(sum(mask),1)*5);  % placeholder wait
    userReport.DriveTime(i) = mean(rand(sum(mask),1)*15); % placeholder drive
end
fprintf('\n=== EV User Report (after dynamic pricing) ===\n');
disp(userReport);
writetable(userReport,'ev_user_report.csv');

% --- Station price schedule report ---
fprintf('\n=== Station Price Schedule Report (15 min slots) ===\n');
PriceTbl = array2table(priceSchedule','VariableNames', ...
    arrayfun(@(s) sprintf('Station_%d',s), 1:K,'UniformOutput',false));
PriceTbl.TimeSlot_hr = (0:nSlots-1)'*slotDur;
disp(PriceTbl(1:min(10,height(PriceTbl)),:));
writetable(PriceTbl,'station_price_schedule.csv');
fprintf('Saved station_rerouting_report.csv, ev_user_report.csv, station_price_schedule.csv\n');

