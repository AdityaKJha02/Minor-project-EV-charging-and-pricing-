clc; clear; close all;
rng(70);   % fixed seed for reproducibility

%% =========================================================
%% 1. SHORTEST PATHS (YOUR WORKING FILE – UNCHANGED)
%% =========================================================
M = readmatrix('distance_matrix.xlsx');
M(M==0) = Inf;
M(1:size(M,1)+1:end) = 0;
[dist, nextHop] = floydWarshallNextHop(M);

%% =========================================================
%% 2. READ OD PAIRS (UNCHANGED LOGIC)
%% =========================================================
txt = extractFileText('OD nodes.docx');
pairs = regexp(txt,'\((\d+)\s*,\s*(\d+)\)','tokens');

OD = cellfun(@(c)[str2double(c{1}) str2double(c{2})], ...
             pairs, 'UniformOutput', false);
OD = vertcat(OD{:});

%% =========================================================
%% 3. BUILD TASK PATHS (UNCHANGED)
%% =========================================================
taskPaths = cell(size(OD,1),1);
for i = 1:size(OD,1)
    taskPaths{i} = localPath(OD(i,1), OD(i,2), nextHop);
end

%% =========================================================
%% 4. CANDIDATE EDGES (UNCHANGED)
%% =========================================================
candidateEdges = [];
for i = 1:numel(taskPaths)
    p = taskPaths{i};
    if numel(p) > 1
        candidateEdges = [candidateEdges; [p(1:end-1)' p(2:end)']];
    end
end
candidateEdges = unique(candidateEdges,'rows');
fprintf('Found %d candidate edges from tasks.\n', size(candidateEdges,1));

%% =========================================================
%% 5. LOAD POWER DATA (NEW – REQUIRED FOR TEMPORAL VARIANCE)
%% =========================================================

% --- IEEE 69-bus 24h active power data (69 x 24)
baseLoad24h = readmatrix('IEEE69_24h active data.xlsx');

% --- Edge → Bus mapping
% Expected format: [fromNode  toNode  busIndex]
edgeToBus = readmatrix('Bus to edge mapping.xlsx');

%% =========================================================
%% 5B. LOAD HOURLY VEHICLE FLOW (PER-UNIT → ABSOLUTE)
%% =========================================================

% 24 × 1 per-unit vehicle flow (0–1)
vehFlow_pu = readmatrix('24hr_vehicles.xlsx');

if numel(vehFlow_pu) ~= 24
    error('Vehicle flow file must contain 24 hourly values');
end

maxEVsPerHour = 5;                % p.u. = 1 corresponds to 50 EVs
numEVs_hourly = round(maxEVsPerHour * vehFlow_pu(:));

fprintf('Hourly EV flow loaded (max = %d EV/h)\n', maxEVsPerHour);

%% =========================================================
%% 6. PARAMETERS (SAME AS YOUR OLD CODE)
%% =========================================================
consRate = 1;

minSOC = 60;
maxSOC = 85;
chargeThreshold = 55;
stopSOC = 20;

battery_kWh = 59;

numStations = 6;

wCharge = 8.431;
wStop   = 8.709;
wVar    = 1.092;

maxGreedyIter = 1000;
disp(numEVs_hourly.')

%% =========================================================
%% 7. RUN GREEDY (UPDATED CALL SIGNATURE)
%% =========================================================
[chosenEdges, history] = greedyPlacement_minmax_temporalVar( ...
    candidateEdges, taskPaths, OD, dist, ...
    consRate, numEVs_hourly, minSOC, maxSOC, ...
    chargeThreshold, stopSOC, ...
    numStations, wCharge, wStop, wVar, ...
    maxGreedyIter, battery_kWh, ...
    baseLoad24h, edgeToBus);


%% =========================================================
%% 8. FINAL OUTPUT
%% =========================================================
disp('Selected EVFCS edges:');
disp(chosenEdges);

%% =========================================================
%% LOCAL HELPER (UNCHANGED)
%% =========================================================
function path = localPath(u, v, nextHop)
path = u;
while u ~= v
    u = nextHop(u, v);
    if u == 0
        path = [];
        return;
    end
    path(end+1) = u; %#ok<AGROW>
end
end


%% =========================================================
%% 9. CHARGE SOLD & TIME PER STATION
%% =========================================================

%% =========================================================
%% RUN STATION SALES & TIME SIMULATION
%% =========================================================

%{
stationStats = simulateStationSalesAndTime( ...
    chosenEdges, taskPaths, dist, ...
    consRate, numEVs_hourly, ...
    minSOC, maxSOC, ...
    chargeThreshold, stopSOC, ...
    battery_kWh);

%% =========================================================
%% PRINT RESULTS
%% =========================================================

fprintf('\n=== STATION SALES SUMMARY ===\n');
for k = 1:numel(stationStats)
    fprintf(['Station %d (edge %d->%d): ', ...
             'ChargeSold = %.2f kWh | ', ...
             'AvgChargeTime = %.2f min | ', ...
             'AvgEnergy/EV = %.2f kWh | ', ...
             'Sessions = %d\n'], ...
        k, ...
        stationStats(k).edge(1), stationStats(k).edge(2), ...
        stationStats(k).chargeSold_kWh, ...
        stationStats(k).avgChargingTime_min, ...
        stationStats(k).avgEnergyPerEV_kWh, ...
        stationStats(k).numCharges);
end

speed_kmh = 60;
kmPerUnit = 2;
%}

portsPerStation =  [8 5 2 5 4 5];

%% =========================================================
%% 10. COMPUTE REQUIRED NUMBER OF CHARGING PORTS
%% =========================================================

%{
maxPortHours = 10;  
portStats = computeChargingPorts(stationStats, maxPortHours);

fprintf('\n=== CHARGING PORT REQUIREMENTS (Max port usage = %.1f hr) ===\n', ...
        maxPortHours);
for s = 1:numel(portStats)
    fprintf(['Station %d (edge %d->%d): ', ...
             'Ports = %d | Avg port usage = %.2f hr | Utilization = %.1f%%\n'], ...
        portStats(s).stationID, ...
        portStats(s).edge(1), portStats(s).edge(2), ...
        portStats(s).numPorts, ...
        portStats(s).avgPortUsage_hr, ...
        portStats(s).utilization_pct);
end
%}


% ================= EV TIMELINE PARAMETERS =================
kmPerUnit      = 2;        % km per distance unit
speed_kmh      = 60;       % EV speed
chargerPower_kW = 75;      % DC fast charger power

% ports per station (same order as chosenEdges)
% =========================================================
% EV ARRIVAL – WAITING – DEPARTURE TIME SIMULATION
% =========================================================

%% =========================================================
%% 11. EV TIMELINE
%% =========================================================
[EVevents, EVsummary] = simulateEVTimeline( ...
    OD, taskPaths, chosenEdges, dist, ...
    numEVs_hourly, kmPerUnit, speed_kmh, ...
    minSOC, maxSOC, chargeThreshold, stopSOC, consRate, ...
    battery_kWh, chargerPower_kW, portsPerStation);

%% =========================================================
%% APPROXIMATE EV ARRIVALS PER STATION (TRAFFIC-BASED)
%% =========================================================

fprintf('\n============================================\n');
fprintf('APPROXIMATE EV ARRIVAL ANALYSIS\n');
fprintf('============================================\n');

numStations = size(chosenEdges,1);
numOD       = size(OD,1);

% 1️⃣ Total EVs generated in whole network per day
totalEVs_day = sum(numEVs_hourly) * numOD;

fprintf('Total EVs generated in network per day = %d\n', totalEVs_day);

% 2️⃣ Count how many OD paths include each station edge
ODcount_per_station = zeros(numStations,1);

for s = 1:numStations

    u = chosenEdges(s,1);
    v = chosenEdges(s,2);

    for i = 1:numOD

        p = taskPaths{i};
        if isempty(p), continue; end

        edges = [p(1:end-1)' p(2:end)'];

        if any(ismember(edges,[u v],'rows')) || ...
           any(ismember(edges,[v u],'rows'))

            ODcount_per_station(s) = ...
                ODcount_per_station(s) + 1;
        end
    end
end

% 3️⃣ Approx EV arrivals per station per day
EVarrivals_perStation = ...
    ODcount_per_station * sum(numEVs_hourly);

% 4️⃣ Actual EVs that charged (from simulation)
actualCharged_perStation = zeros(numStations,1);

for s = 1:numStations
    actualCharged_perStation(s) = ...
        numel(unique(EVevents.evID(EVevents.stationIdx==s)));
end

% 5️⃣ Display comparison
fprintf('\nStation Comparison:\n');
fprintf('------------------------------------------------------------\n');
fprintf('Station | OD_paths | Approx_Arrivals | Actual_Charged\n');
fprintf('------------------------------------------------------------\n');

for s = 1:numStations
    fprintf('%7d | %8d | %15d | %14d\n', ...
        s, ...
        ODcount_per_station(s), ...
        EVarrivals_perStation(s), ...
        actualCharged_perStation(s));
end

fprintf('------------------------------------------------------------\n');
%% =========================================================
%% 12. EVFCS POWER TIMELINE
%% =========================================================

generateEVFCSArrivalAndPowerTimeline_10min( ...
    EVevents, chosenEdges, portsPerStation, battery_kWh);

%% =========================================================
%% 12(b) Complete data
%% =========================================================

sheetNames_EV = {
    'EVFCS_1_6_2'
    'EVFCS_2_24_21'
    'EVFCS_3_15_22'
    'EVFCS_4_13_12'
    'EVFCS_5_7_18'
    'EVFCS_6_10_15'
};

generateFCS_Summary( ...
    'EVFCS_10min_Arrival_Power.xlsx', ...
    'EVFCS_EV_Timeline.xlsx', ...
    sheetNames_EV);

%% =========================================================
% USER INPUT: EV IDs (comma separated)
%% =========================================================

% =========================================================
% AUTOMATICALLY PROCESS ALL EV IDs
% =========================================================

totalEVs_day = sum(numEVs_hourly) * size(OD,1);

% Generate EV IDs from 1 to totalEVs_day
evIDs = 1:totalEVs_day;

fprintf('\nProcessing ALL EVs from 1 to %d\n', totalEVs_day);
outputExcelTrace = 'EV_Full_Travel_Trace.xlsx';

if exist(outputExcelTrace,'file')
    delete(outputExcelTrace);
end

sheetName = 'EV_Travel_Summary';

rowPtr = 1;   % keeps track of Excel row writing
totalEVs_day = sum(numEVs_hourly) * size(OD,1);

writecell({'Total EVs generated in network per day', totalEVs_day}, ...
          outputExcelTrace, ...
          'Sheet', sheetName, ...
          'Range', sprintf('A%d', rowPtr));

rowPtr = rowPtr + 3;   % leave gap
% Column headers
writecell({'EV_ID','Type','Node_or_Event','Arrival_min',...
           'SOC_pct','Distance_units',...
           'Station','Port','Start_min','Charge_min',...
           'Depart_min','SOC_before','SOC_after','Energy_kWh'}, ...
          outputExcelTrace, ...
          'Sheet', sheetName, ...
          'Range', sprintf('A%d', rowPtr));

rowPtr = rowPtr + 1;
%% =========================================================
% LOOP OVER REQUESTED EVs
%% =========================================================

for id = evIDs

    

    % ---- Find EV summary ----
    idx_summary = find(EVsummary.evID == id);

    if isempty(idx_summary)
        fprintf('EV %d not found.\n', id);
        continue;
    end

    odIndex    = EVsummary.odIndex(idx_summary);
depart_min = EVsummary.depart_min(idx_summary);
soc_init   = EVsummary.soc_init(idx_summary);

% Get actual OD pair
originNode = OD(odIndex,1);
destNode   = OD(odIndex,2);


writecell({id,'INFO',...
          sprintf('%d --> %d',originNode,destNode),...
          depart_min,soc_init}, ...
          outputExcelTrace, ...
          'Sheet', sheetName, ...
          'Range', sprintf('A%d', rowPtr));

rowPtr = rowPtr + 1;
    % ---- Get path ----
    path = taskPaths{odIndex};
    destNode = OD(odIndex,2);

    % ---- Get charging events ----
    EVev = EVevents(EVevents.evID == id, :);

    % -----------------------------------------
    % TRACE TRAVEL
    % -----------------------------------------
soc = soc_init;
cumTime = 0;
cumDist = 0;


for p = 1:(length(path)-1)

    u = path(p);
    v = path(p+1);

    % ---- Print NODE FIRST ----
    arrivalTime = depart_min + cumTime;


writecell({id,'NODE',...
          u,arrivalTime,soc,cumDist}, ...
          outputExcelTrace, ...
          'Sheet', sheetName, ...
          'Range', sprintf('A%d', rowPtr));

rowPtr = rowPtr + 1;
    % ---- Check if charging at this node ----
    if ~isempty(EVev)

        chargeIdx = find(abs(EVev.arrival_min - arrivalTime) < 1e-6);

        if ~isempty(chargeIdx)

            ce = EVev(chargeIdx,:);

            
writecell({id,'CHARGING',...
          u,arrivalTime,[],[],...
          ce.stationIdx,ce.port,...
          ce.start_min,ce.charge_min,...
          ce.depart_event_min,...
          ce.soc_before,ce.soc_after,...
          ce.energy_kWh}, ...
          outputExcelTrace, ...
          'Sheet', sheetName, ...
          'Range', sprintf('A%d', rowPtr));

rowPtr = rowPtr + 1;
            % Update SOC
            soc = ce.soc_after;

            % Update cumulative time after charging
            cumTime = ce.depart_event_min - depart_min;

            arrivalTime = depart_min + cumTime;
        end
    end

    % ---- Now print EDGE travel ----
    d_uv = dist(u,v);
    distance_km = d_uv * kmPerUnit;

    travelTime_hr = distance_km / speed_kmh;
    travelTime_min = travelTime_hr * 60;

    

    % ---- Update SOC and time after travel ----
    soc = soc - d_uv * consRate;
    if soc < stopSOC
        soc = stopSOC;
    end

    cumTime = cumTime + travelTime_min;
    cumDist = cumDist + d_uv;

end
    % ---- Final destination ----
    
writecell({id,'DESTINATION',...
          destNode,depart_min + cumTime,soc,cumDist}, ...
          outputExcelTrace, ...
          'Sheet', sheetName, ...
          'Range', sprintf('A%d', rowPtr));

rowPtr = rowPtr + 2;   % blank row before next EV
end
