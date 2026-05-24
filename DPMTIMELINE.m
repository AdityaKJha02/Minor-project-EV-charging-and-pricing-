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
%
%% =========================================================
%% 5C. LOAD HOURLY VEHICLE FLOW DPM (FROM EXCEL – SHEET 2, ROW 2)
%% =========================================================

%% =========================================================
%% 5C. LOAD HOURLY VEHICLE FLOW DPM (ALL FCS FILE)
%% =========================================================

weightsFile = 'PED_Normalized_Hourly_Weights.xlsx';

% Read weights from first sheet (Sheet1)
vehFlow_pu = readmatrix(weightsFile, ...
    'Range', 'C2:C25');

% Convert to column vector (24×1)
vehFlow_pu = vehFlow_pu(:);

% Validation
if numel(vehFlow_pu) ~= 24
    error('Expected 24 hourly weights in Sheet1');
end

vehFlow_pu = vehFlow_pu(:);   % force 24×1 column vector


disp(vehFlow_pu');
% Convert per-unit flow to absolute EVs
maxEVsPerHour = 5;            % p.u. = 1 corresponds to 5 EV/h
% Scale per-unit weights
exactEVs = maxEVsPerHour * vehFlow_pu;   % 24×1 (fractional)

% Step 1: floor
EV_floor = floor(exactEVs);

% Step 2: fractional residuals
residual = exactEVs - EV_floor;

% Step 3: how many EVs are missing?
missingEVs = round(sum(exactEVs) - sum(EV_floor));

% Step 4: assign missing EVs to largest residuals
[~, idx] = sort(residual, 'descend');

numEVs_hourly1 = EV_floor;
numEVs_hourly1(idx(1:missingEVs)) = ...
    numEVs_hourly1(idx(1:missingEVs)) + 1;


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
disp(numEVs_hourly1)

%% =========================================================
%% 7. RUN GREEDY (UPDATED CALL SIGNATURE)
%% =========================================================
[chosenEdges, history] = greedyPlacement_minmax_temporalVar( ...
    candidateEdges, taskPaths, OD, dist, ...
    consRate, numEVs_hourly1, minSOC, maxSOC, ...
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
speed_kmh = 60;
kmPerUnit = 2;
portsPerStation = [8 5 2 5 4 5];


%% =========================================================

% ================= EV TIMELINE PARAMETERS =================
kmPerUnit      = 2;        % km per distance unit
speed_kmh      = 60;       % EV speed
chargerPower_kW = 75;      % DC fast charger power

% ports per station (same order as chosenEdges)
% =========================================================
% EV ARRIVAL – WAITING – DEPARTURE TIME SIMULATION
% =========================================================

%% =========================================================
%% 11.(b) EV TIMELINE DPM
%% =========================================================
[EVevents1, EVsummary1] = simulateEVTimeline( ...
    OD, taskPaths, chosenEdges, dist, ...
    numEVs_hourly1, kmPerUnit, speed_kmh, ...
    minSOC, maxSOC, chargeThreshold, stopSOC, consRate, ...
    battery_kWh, chargerPower_kW, portsPerStation);


generateEVFCSArrivalAndPowerTimeline_10min( ...
    EVevents1, chosenEdges, portsPerStation, battery_kWh);

sheetNames_EV = {    %dpm
    'EVFCS_1_2_6'
    'EVFCS_2_24_21'
    'EVFCS_3_15_22'
    'EVFCS_4_12_13'
    'EVFCS_5_7_18'
    'EVFCS_6_10_15'
};
generateFCS_Summary( ...
    'EVFCS_10min_Arrival_Power.xlsx', ...
    'EVFCS_EV_Timeline.xlsx', ...
    sheetNames_EV);

filename = 'EVFCS_10min_Arrival_Power.xlsx';


for s = 1:length(sheetNames_EV)

    sheet = sheetNames_EV{s};

    % Read sheet
    data = readcell(filename,'Sheet',sheet);

    colG = 7; % Column G

    firstRow = [];

    % Find first non-zero value in column G
    for r = 2:size(data,1)
        val = data{r,colG};

        if isnumeric(val) && ~isnan(val) && val ~= 0
            firstRow = r;
            break
        end
    end

    % Move first valid row to row 2
    if ~isempty(firstRow) && firstRow > 2
        data = [data(1,:); data(firstRow:end,:)];
    end
    % Number of data rows (excluding header)
n = size(data,1) - 1;

% Generate time sequence
time_min = (0:10:(n-1)*10)';

hour = mod(floor(time_min/60),24);
minute = mod(time_min,60);

% Rewrite first three columns
data(2:end,1) = num2cell(hour);
data(2:end,2) = num2cell(minute);
data(2:end,3) = num2cell(time_min);

    % --- Fix for writecell ---
    for i = 1:numel(data)
        if ismissing(data{i})
            data{i} = [];
        end
    end
    % -------------------------

    % Write back to Excel
    writecell(data, filename,'Sheet', sheet);

end

disp('All sheets processed successfully.');
