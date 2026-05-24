function [EVevents, EVsummary] = simulateEVTimeline( ...
   OD, taskPaths, chosenEdges, dist, ...
   numEVs_hourly, kmPerUnit, speed_kmh, ...
   minSOC, maxSOC, chargeThreshold, stopSOC, consRate, ...
   battery_kWh, chargerPower_kW, portsPerStation)
% SIMULATEEVTIMELINE
% Simulate arrival, waiting, charging and departure times for EVs starting
% from OD origins and traversing their taskPaths. Uses hourly counts to
% space start times within each hour. Assigns ports at stations and
% computes waiting if all ports busy.
%
% INPUTS (use your existing variables):
%   OD                 - Nx2 origin-destination node list
%   taskPaths          - cell array of node paths per OD (from localPath)
%   chosenEdges        - Kx2 station edges [u v] (station located at upstream node u)
%   dist               - floyd-warshall shortest-path distance matrix (units)
%   numEVs_hourly      - 24x1 vector: EVs per hour (per OD) (may be integers)
%   kmPerUnit          - km per distance unit (e.g. 2)
%   speed_kmh          - vehicle speed (km/h)
%   minSOC,maxSOC      - SOC bounds in percent (e.g. 60,85)
%   chargeThreshold    - SOC threshold (%) below which EVs will charge
%   stopSOC            - SOC forced minimum if "stopped"
%   consRate           - %SOC consumed per distance unit
%   battery_kWh        - vehicle battery capacity (kWh)
%   chargerPower_kW    - assumed charging power (kW)
%   portsPerStation    - Kx1 integer vector (# of ports per station)
%
% OUTPUTS:
%   EVevents  - table of all charging events (one row per charging session)
%   EVsummary - table per EV summarizing if it charged multiple times, times
% -----------------------------------------
% Basic setup
% -----------------------------------------
K = size(chosenEdges,1);
numOD = size(OD,1);
% per-station port next-free times (minutes). Each station has portsPerStation(s) ports
stationPortNextFree = cell(K,1);
for s = 1:K
   stationPortNextFree{s} = zeros(max(1,portsPerStation(s)),1); % minutes since t=0
end
% Event log arrays (grow as needed)
evList = []; % will collect EV-level summary
eventRows = []; % will collect per-charging-event rows
evGlobalID = 0;
% Helper: find station index for an edge (undirected match)
edgeKey = @(u,v) sprintf('%d-%d',u,v);
stationKeys = arrayfun(@(i) edgeKey(chosenEdges(i,1), chosenEdges(i,2)), 1:K, 'UniformOutput', false);
% -----------------------------------------
% Main loop: hours -> OD -> EVs
% -----------------------------------------
for hr = 1:24
   nPerHour = numEVs_hourly(hr);    % EVs per OD this hour (may be 0)
   if nPerHour <= 0
       continue;
   end
   % evenly spaced departure offsets (minutes) within hour: 0, 60/n, 2*60/n, ...
   interval = 60 / nPerHour;
   departOffsets = (0:(nPerHour-1)) * interval;   % minutes after (hr-1)*60
   for odIdx = 1:numOD
       path = taskPaths{odIdx};
       if isempty(path), continue; end
       originNode = OD(odIdx,1);
       destNode   = OD(odIdx,2);
       for kEV = 1:nPerHour
           evGlobalID = evGlobalID + 1;
           depart_min = (hr-1)*60 + departOffsets(kEV);   % absolute minutes since day start
           % initial SOC sampled uniformly between minSOC and maxSOC (as earlier)
% --- START SOC DISTRIBUTION: 95% inside [minSOC,maxSOC], 5% outside ---
pStartMain = 0.95;

if rand() <= pStartMain
    % 95% EVs: normal distribution inside bounds
    muSOC    = (minSOC + maxSOC) / 2;
    sigmaSOC = (maxSOC - minSOC) / (2*1.96);   % 95% inside bounds
    soc_init = muSOC + sigmaSOC * randn();
    soc_init = min(max(soc_init, minSOC), maxSOC);
else
    % 5% EVs: outside bounds
    if rand() < 0.5
        % very low SOC
        soc_init = max(0, minSOC - 5*rand());
    else
        % very high SOC
        soc_init = min(100, maxSOC + 5*rand());
    end
end
           % prepare per-EV bookkeeping
           evID = evGlobalID;
           evChargedTimes = 0;
           evChargeEvents = [];
           % current SOC as percent
           soc = soc_init;
           % cumulative travel time (minutes) and distance units traveled so far
           cumTime_min = 0;
           cumDist_units = 0;
           % start traversing the path edge-by-edge
           for pIdx = 1:(numel(path)-1)
               u = path(pIdx);
               v = path(pIdx+1);
               % distance for this step (graph units)
               d_uv = dist(u,v);            % units
               if ~isfinite(d_uv), break; end
               % travel time for this edge in minutes
               travelTime_hr = (d_uv * kmPerUnit) / speed_kmh;
               travelTime_min = travelTime_hr * 60;
               % arrival time at the head of this edge (before traversing this edge)
               arrival_min = depart_min + cumTime_min;
               % check whether there is a station at this edge (either direction)
               % station is considered at upstream node of chosenEdges (u->v) or reverse
               idxFwd = find(strcmp(stationKeys, edgeKey(u,v)),1);
               idxRev = find(strcmp(stationKeys, edgeKey(v,u)),1);
               if ~isempty(idxFwd)
                   stationIdx = idxFwd;
                   stationNode = chosenEdges(stationIdx,1);
               elseif ~isempty(idxRev)
                   stationIdx = idxRev;
                   stationNode = chosenEdges(stationIdx,1); % station upstream stored that way
               else
                   stationIdx = 0;
               end
               % BEFORE traversing this edge, if there is a station at (u,v), EV arrives at stationNode (u)
               if stationIdx ~= 0
                   % compute remaining distance (units) from station node to destination node
                   rem_units = dist(stationNode, destNode);
                   if ~isfinite(rem_units)
                       rem_units = 0;
                   end
                   % compute SOC at arrival (current soc already reflects previous travel; it hasn't yet subtracted this edge)
                   soc_arrival = soc;   % percent
                   % charging decision: will charge if soc < chargeThreshold
                   if soc_arrival < chargeThreshold
                       % desired SOC to safely finish remaining route (in percent)
                       % --- Target SOC distribution: 95% in [75,80]
% Probability that EV follows target SOC policy
pTarget = 0.95;

if rand() <= pTarget
    % 95% of EVs → target SOC in [75,80]
    muTargetSOC    = 77.5;
    sigmaTargetSOC = (80 - 75)/(2*1.96);

    targetSOC = muTargetSOC + sigmaTargetSOC*randn();
    targetSOC = min(max(targetSOC,75),80);
else
    % 5% of EVs → minimal charging (or legacy behavior)
    targetSOC = min(maxSOC, soc_arrival + rem_units*consRate);
end


deltaSOC_pct = max(0, targetSOC - soc_arrival);

                       if deltaSOC_pct > 0
                           % charging time (minutes) using SOC-based charging curve
chargeTime_min = charging_time(soc_arrival, soc_arrival + deltaSOC_pct);

% energy delivered (still useful for reporting)
energy_req_kWh = (deltaSOC_pct/100) * battery_kWh;

                           % PORT ASSIGNMENT: check stationPortNextFree times
                           ports = stationPortNextFree{stationIdx};
                           % earliest port that is free at or before arrival
                           % Find all free ports at arrival time
freePorts = find(ports <= arrival_min);

if ~isempty(freePorts)
    % Always choose smallest numbered free port
    portUsed = min(freePorts);
    wait_min = 0;
    start_min = arrival_min;
else
    % All ports busy → choose port that becomes free earliest
    [nextFreeTime, portUsed] = min(ports);
    wait_min = nextFreeTime - arrival_min;
    start_min = nextFreeTime;
end


depart_min_event = start_min + chargeTime_min;
ports(portUsed) = depart_min_event;
stationPortNextFree{stationIdx} = ports;

                           
                           % record event
                           evChargedTimes = evChargedTimes + 1;
                           evChargeEvents(end+1).station = stationIdx; %#ok<AGROW>
                           evChargeEvents(end).port          = portUsed;
                           evChargeEvents(end).arrival_min = arrival_min;
                           evChargeEvents(end).wait_min = wait_min;
                           evChargeEvents(end).start_min = start_min;
                           evChargeEvents(end).charge_min = chargeTime_min;
                           evChargeEvents(end).depart_min = depart_min_event;
                           evChargeEvents(end).soc_before = soc_arrival;
                           evChargeEvents(end).soc_after = targetSOC;
                           % update SOC to after charging
                           soc = targetSOC;

                       end
                   end
               end
               % after station (or not), traverse the edge: SOC reduces by d_uv*consRate
               soc = soc - d_uv * consRate;
               if soc < stopSOC
                   soc = stopSOC;
               end
               cumDist_units = cumDist_units + d_uv;
               cumTime_min = cumTime_min + travelTime_min;
           end % end path edges
           % store EV-level summary
           evList(end+1).evID = evID; %#ok<AGROW>
           evList(end).odIndex = odIdx;
           evList(end).depart_min = depart_min;
           evList(end).soc_init = soc_init;
           evList(end).numCharges = evChargedTimes;
           evList(end).chargeEvents = evChargeEvents;
       end % EVs per OD (kEV)
   end % OD loop
end % hour loop
% Convert eventRows / evList into tables for output
% Build EVevents table where each row is a charging event
rows = {};

for i = 1:numel(evList)
    ev = evList(i);

    if isempty(ev.chargeEvents)
        continue;
    end

    for e = 1:numel(ev.chargeEvents)
        ce = ev.chargeEvents(e);

        rows(end+1,:) = { ...
            ev.evID, ...
            ev.odIndex, ...
            ce.station, ...
            ce.port, ...
            ce.arrival_min, ...
            ce.wait_min, ...
            ce.start_min, ...
            ce.charge_min, ...
            ce.depart_min, ...
            ce.soc_before, ...
            ce.soc_after ...
        };
    end
end


EVevents = cell2table(rows, 'VariableNames', ...
   {'evID','odIndex','stationIdx','port', ...
    'arrival_min','wait_min','start_min','charge_min', ...
    'depart_event_min','soc_before','soc_after'});
% ---------------------------------------------------------
% Energy consumed per charging event (kWh)
% ---------------------------------------------------------
EVevents.energy_kWh = ...
    (EVevents.soc_after - EVevents.soc_before) .* battery_kWh / 100;
% =========================================================
% FIX PORT ASSIGNMENT (CHRONOLOGICAL BY ARRIVAL TIME)
% =========================================================

for s = 1:K

    idx = find(EVevents.stationIdx == s);

    if isempty(idx)
        continue;
    end

    Ts = EVevents(idx,:);

    % Sort by arrival time
    Ts = sortrows(Ts,'arrival_min');

    portNextFree = zeros(portsPerStation(s),1);

    for r = 1:height(Ts)

        arrival = Ts.arrival_min(r);

        freePorts = find(portNextFree <= arrival);

        if ~isempty(freePorts)
            portUsed = min(freePorts);   % always smallest port
            startTime = arrival;
        else
            [nextFree, portUsed] = min(portNextFree);
            startTime = nextFree;
        end

        departTime = startTime + Ts.charge_min(r);

        portNextFree(portUsed) = departTime;

        Ts.port(r) = portUsed;
        Ts.start_min(r) = startTime;
        Ts.wait_min(r) = startTime - arrival;
        Ts.depart_event_min(r) = departTime;
    end

    % Write back to main table
    EVevents(idx,:) = Ts;

end

% =========================================================
% WRITE EXCEL FILE (ONE SHEET PER EVFCS)
% =========================================================

excelFile = 'EVFCS_EV_Timeline.xlsx';

if exist(excelFile,'file')
    delete(excelFile);
end

numStations = size(chosenEdges,1);

for s = 1:numStations

    T = EVevents(EVevents.stationIdx == s, :);

    if isempty(T)
        continue;
    end

    sheetName = sprintf('EVFCS_%d_%d_%d', ...
        s, chosenEdges(s,1), chosenEdges(s,2));

    writetable(T, excelFile, 'Sheet', sheetName);

end

% EVsummary: one row per EV
EVsummary = table( ...
   zeros(numel(evList),1), ...   % evID
   zeros(numel(evList),1), ...   % odIndex
   zeros(numel(evList),1), ...   % depart_min
   zeros(numel(evList),1), ...   % soc_init
   zeros(numel(evList),1), ...   % numCharges
   false(numel(evList),1), ...   % chargedMultipleTimes
   'VariableNames', { ...
       'evID','odIndex','depart_min','soc_init', ...
       'numCharges','chargedMultipleTimes'} );
for i = 1:numel(evList)
   e = evList(i);
   EVsummary.evID(i) = e.evID;
   EVsummary.odIndex(i) = e.odIndex;
   EVsummary.depart_min(i) = e.depart_min;
   EVsummary.soc_init(i) = e.soc_init;
   EVsummary.numCharges(i) = e.numCharges;
   EVsummary.chargedMultipleTimes(i) = (e.numCharges > 1);
end


% =========================================================
% PRINT TOTAL ENERGY SOLD PER EVFCS
% =========================================================
% =========================================================
% STATION-LEVEL SUMMARY (PRINT TO COMMAND WINDOW)
% =========================================================

fprintf('\n=== STATION SALES SUMMARY ===\n');

numStations = size(chosenEdges,1);

for s = 1:numStations

    % Events for this station
    Ts = EVevents(EVevents.stationIdx == s, :);

    if isempty(Ts)
        fprintf('Station %d (edge %d->%d): No charging sessions\n', ...
            s, chosenEdges(s,1), chosenEdges(s,2));
        continue;
    end

    % ---- Metrics ----
    totalEnergy_kWh   = sum(Ts.energy_kWh);
    totalTime_min     = sum(Ts.charge_min);
    numSessions       = height(Ts);

    avgChargeTime_min = totalTime_min / numSessions;
    avgEnergyPerEV    = totalEnergy_kWh / numSessions;

    % ---- Print ----
    fprintf(['Station %d (edge %d->%d): ', ...
             'ChargeSold = %.2f kWh | ', ...
             'TotalTime = %.2f min | ', ...
             'AvgChargeTime = %.2f min | ', ...
             'AvgEnergy/EV = %.2f kWh | ', ...
             'Sessions = %d\n'], ...
        s, chosenEdges(s,1), chosenEdges(s,2), ...
        totalEnergy_kWh, totalTime_min, ...
        avgChargeTime_min, avgEnergyPerEV, ...
        numSessions);
end




