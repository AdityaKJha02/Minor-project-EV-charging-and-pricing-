%% EV Charging Station Placement & Simulation (Greedy, avoids dead stations)
% Full end-to-end script. Paste into a new .m file and run.

clc; clear; close all;
rng('shuffle');

%% -------------------- USER TUNABLE PARAMETERS --------------------
numEVs = 20;           % EVs simulated per task
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
    % For speed, create string keys for station edges
    keys = cell(K,1);
    for ii=1:K, keys{ii} = sprintf('%d-%d', stationEdges(ii,1), stationEdges(ii,2)); end

    for t = 1:length(taskPaths)
        p = taskPaths{t};
        if isempty(p), continue; end
        socEVs = minSOC + (maxSOC-minSOC).*rand(numEVs,1);
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
            % travel
            socEVs = socEVs - d*consRate;
            % stopped EVs
            stopped = socEVs < stopSOC;
            totalStops = totalStops + sum(stopped);
            socEVs(stopped) = stopSOC;
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

%% ------------------ Charging Infrastructure Simulation (1-hour window) ------------------
% Assumptions & parameters
battery_kWh = 59;       % Mahindra BE 6C battery capacity (kWh)
AC_power    = 11.2;     % kW per AC charger
DC_power    = 140;      % kW per DC fast charger
numAC       = 10;        % AC chargers per station
numDC       = 2;        % DC chargers per station
timeWindow_hr = 1.0;    % 1 hour time horizon

% Require chargeLog from simulateWithStationsFull
if ~exist('chargeLog','var') || isempty(chargeLog)
    warning('chargeLog not found or empty. Re-running full simulation to obtain charge events...');
    [~,~,~,~,chargeLog] = simulateWithStationsFull(chosenEdges, taskPaths, OD, dist, ...
        consRate, numEVs, minSOC, maxSOC, chargeThreshold, stopSOC);
end

% Map station edges to their index in chosenEdges
K = size(chosenEdges,1);
stationKeys = cell(K,1);
for i=1:K, stationKeys{i} = sprintf('%d-%d', chosenEdges(i,1), chosenEdges(i,2)); end

if K==0
    error('No charging stations selected. Cannot simulate chargers.');
end

% Build list of charging events from chargeLog
Nevents = numel(chargeLog);
events = table((1:Nevents)', zeros(Nevents,1), zeros(Nevents,1), zeros(Nevents,1), ...
    'VariableNames', {'evtID','stationIdx','energy_kWh','reqDuration_hr'});

for e = 1:Nevents
    rec = chargeLog(e);
    key = sprintf('%d-%d', rec.station_u, rec.station_v);
    sidx = find(strcmp(stationKeys, key),1);
    if isempty(sidx)
        events.stationIdx(e) = 0;
        events.energy_kWh(e) = 0;
        events.reqDuration_hr(e) = Inf;
    else
        events.stationIdx(e) = sidx;
        energy_kWh = (rec.chargeSoC/100) * battery_kWh;
        events.energy_kWh(e) = energy_kWh;
        % request duration depends on charger type later
        % keep Inf here, will decide when assigning charger
        events.reqDuration_hr(e) = NaN;
    end
end

% Sort events by energy demand (largest first)
[~,order] = sort(events.energy_kWh,'descend');
events = events(order,:);

%% --------- Revised Scheduling: choose charger that MAXIMIZES delivered energy in window ---------
% Initialize charger availability per station
stationChargerNextFree = cell(K,1);
stationChargerPower = cell(K,1);
totalChgPerStation = numAC + numDC;
for s = 1:K
    stationChargerNextFree{s} = zeros(totalChgPerStation,1);  % all free at t=0
    % first numAC slots are AC, last numDC slots are DC
    stationChargerPower{s} = [repmat(AC_power, numAC, 1); repmat(DC_power, numDC, 1)];
end

% Prepare scheduled table (add chargerIdx and chargerPower columns)
scheduled = table(events.evtID, events.stationIdx, events.energy_kWh, ...
    zeros(height(events),1), zeros(height(events),1), strings(height(events),1), zeros(height(events),1), zeros(height(events),1), ...
    'VariableNames', {'evtID','stationIdx','energy_kWh_req','start_hr','end_hr','chargerType','energy_kWh_delivered','chargerIdx'});

% For each event, evaluate all chargers at that station and pick the charger that
% yields maximum delivered energy within the timeWindow_hr. Tie-break by earliest start.
for i = 1:height(events)
    s = events.stationIdx(i);
    if s == 0
        % no station (shouldn't happen) -> leave zeros
        continue;
    end

    nextFree = stationChargerNextFree{s};        % vector of next free times (hr)
    powers = stationChargerPower{s};             % vector of charger powers (kW)
    nCh = numel(nextFree);
    if nCh == 0
        continue;
    end

    reqEnergy = events.energy_kWh(i);            % kWh required

    % Evaluate each charger j
    bestDelivered = -Inf;
    bestIdx = 0;
    bestStart = NaN;
    bestEnd = NaN;
    bestPower = NaN;
    bestFullEnd = NaN; % real end if charger used (may exceed window)
    for j = 1:nCh
        start_j = nextFree(j);
        power_j = powers(j);
        if power_j <= 0
            continue;
        end
        % time to fully satisfy on this charger (hours)
        dur_full = reqEnergy / power_j;
        end_full = start_j + dur_full;

        % energy delivered within time window:
        if start_j >= timeWindow_hr
            delivered_j = 0;
            actual_end = NaN;
        else
            actual_end = min(end_full, timeWindow_hr);
            delivered_j = min(reqEnergy, power_j * max(0, actual_end - start_j)); % kWh
        end

        % tie-break preference: prefer larger delivered_j; if equal prefer earlier start_j; if equal prefer higher power
        if delivered_j > bestDelivered || (abs(delivered_j-bestDelivered)<1e-9 && (start_j < bestStart || (start_j==bestStart && power_j>bestPower)))
            bestDelivered = delivered_j;
            bestIdx = j;
            bestStart = start_j;
            bestEnd = actual_end;
            bestPower = power_j;
            bestFullEnd = end_full;
        end
    end

    % Assign the event to bestIdx (if any delivered > 0 or we still schedule to occupy a charger)
    if bestIdx == 0 || bestDelivered == 0
        % No useful charger within window: mark as not served
        scheduled.start_hr(i) = NaN;
        scheduled.end_hr(i) = NaN;
        scheduled.chargerType(i) = "";
        scheduled.energy_kWh_delivered(i) = 0;
        scheduled.chargerIdx(i) = NaN;
        continue;
    end

    % Record scheduled times and energy delivered (cap at reqEnergy)
    scheduled.start_hr(i) = bestStart;
    scheduled.end_hr(i) = bestEnd;
    scheduled.energy_kWh_delivered(i) = min(reqEnergy, bestDelivered);
    scheduled.chargerIdx(i) = bestIdx;
    if bestIdx <= numAC
        scheduled.chargerType(i) = "AC";
    else
        scheduled.chargerType(i) = "DC";
    end

    % Update the chosen charger nextFree to the full end time (occupied until end_full even if outside window)
    stationChargerNextFree{s}(bestIdx) = bestFullEnd;
end

% Now compute charge times and convert to minutes
scheduled.chargeTime_hr = scheduled.end_hr - scheduled.start_hr;
scheduled.chargeTime_hr(~isfinite(scheduled.chargeTime_hr) | scheduled.chargeTime_hr<0) = 0;
scheduled.chargeTime_min = scheduled.chargeTime_hr * 60;

% Summaries
totalEnergySold_kWh = sum(scheduled.energy_kWh_delivered);
fullyServed = sum(scheduled.energy_kWh_delivered >= scheduled.energy_kWh_req - 1e-6);
partiallyServed = sum(scheduled.energy_kWh_delivered > 0 & scheduled.energy_kWh_delivered < scheduled.energy_kWh_req - 1e-6);
notServed = sum(scheduled.energy_kWh_delivered == 0);

stationServed = zeros(K,1);
stationEnergy = zeros(K,1);
stationTime   = zeros(K,1); % total charging time (hr)
avgChargeTime = zeros(K,1); % average per EV (min)
for s=1:K
    mask = scheduled.stationIdx==s & scheduled.energy_kWh_delivered > 0;
    stationServed(s) = sum(mask);
    stationEnergy(s) = sum(scheduled.energy_kWh_delivered(mask));
    stationTime(s)   = sum(scheduled.chargeTime_hr(mask));
    if stationServed(s) > 0
        avgChargeTime(s) = mean(scheduled.chargeTime_min(mask));
    else
        avgChargeTime(s) = 0;
    end
end

% Print results
fprintf('\n=== Revised Charging Simulation (1h, each station %d AC + %d DC) ===\n', numAC, numDC);
for s=1:K
    fprintf(['Station %d (edge %d->%d): Served %d EVs | Energy sold = %.2f kWh | ' ...
             'Total charging time = %.2f hr | Avg charging time = %.1f min/EV\n'], ...
        s, chosenEdges(s,1), chosenEdges(s,2), ...
        stationServed(s), stationEnergy(s), stationTime(s), avgChargeTime(s));
end
fprintf('Total energy sold = %.2f kWh\n', totalEnergySold_kWh);
fprintf('EVs fully served = %d | Partially served = %d | Not served = %d\n', ...
    fullyServed, partiallyServed, notServed);

% Save logs with charging time in minutes and charger index/type
writetable(scheduled,'charger_schedule.csv');
fprintf('Detailed schedule saved to charger_schedule.csv (includes charge times in minutes and charger index).\n');
